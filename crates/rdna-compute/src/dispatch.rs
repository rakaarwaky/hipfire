//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::kernels;
use hip_bridge::{DeviceBuffer, HipResult, HipRuntime};
use std::collections::HashMap;
use std::ffi::c_void;

/// Tensor stored on the GPU. Tracks shape and element type.
pub struct GpuTensor {
    pub buf: DeviceBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q4K,  // 144 bytes per 256 elements
    Q6K,  // 210 bytes per 256 elements
    Q8_0, // 34 bytes per 32 elements
    Raw,  // raw bytes, no element interpretation
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Raw => 1, // byte-level
        }
    }
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    compiler: KernelCompiler,
    modules: HashMap<String, hip_bridge::Module>,
    functions: HashMap<String, hip_bridge::Function>,
    pool: crate::pool::GpuPool,
}

impl Gpu {
    pub fn init() -> HipResult<Self> {
        let hip = HipRuntime::load()?;
        let count = hip.device_count()?;
        if count == 0 {
            return Err(hip_bridge::HipError::new(0, "no GPU devices found"));
        }
        hip.set_device(0)?;

        let compiler = KernelCompiler::new("gfx1010")?;

        Ok(Self {
            hip,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
        })
    }

    /// Compile and load a kernel, caching the result.
    fn ensure_kernel(&mut self, module_name: &str, source: &str, func_name: &str) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }

        let obj_path = self.compiler.compile(module_name, source)?;
        let obj_path_str = obj_path.to_str().unwrap().to_string();

        if !self.modules.contains_key(module_name) {
            let module = self.hip.module_load(&obj_path_str)?;
            self.modules.insert(module_name.to_string(), module);
        }

        let module = &self.modules[module_name];
        let func = self.hip.module_get_function(module, func_name)?;
        self.functions.insert(func_name.to_string(), func);
        Ok(())
    }

    // ── Tensor allocation ───────────────────────────────────────

    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        let numel: usize = shape.iter().product();
        let byte_size = numel * dtype.size();
        let buf = self.pool.alloc(&self.hip, byte_size)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    pub fn upload_f32(&mut self, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
        let tensor = self.alloc_tensor(shape, DType::F32)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(tensor)
    }

    pub fn download_f32(&self, tensor: &GpuTensor) -> HipResult<Vec<f32>> {
        let numel = tensor.numel();
        let mut data = vec![0.0f32; numel];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4)
        };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)?;
        Ok(data)
    }

    pub fn zeros(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        let tensor = self.alloc_tensor(shape, dtype)?;
        self.hip.memset(&tensor.buf, 0, tensor.byte_size())?;
        Ok(tensor)
    }

    /// GPU-side embedding lookup: copy row `token_id` from embedding table to output.
    /// Avoids downloading the entire embedding table to CPU.
    pub fn embedding_lookup(
        &self,
        table: &GpuTensor,  // [vocab_size * dim] F32
        output: &GpuTensor, // [dim] F32
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        let byte_offset = (token_id as usize) * dim * 4;
        let byte_size = dim * 4;
        self.hip.memcpy_dtod_offset(&output.buf, &table.buf, byte_offset, byte_size)
    }

    /// Upload raw bytes to GPU (for quantized weights).
    pub fn upload_raw(&self, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
        let buf = self.hip.malloc(data.len())?;
        self.hip.memcpy_htod(&buf, data)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype: DType::Raw,
        })
    }

    pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.pool.free(tensor.buf);
        Ok(())
    }

    // ── Kernel operations ───────────────────────────────────────

    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])
    pub fn gemv_f32(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv", kernels::GEMV_SRC, "gemv_f32")?;
        let func = &self.functions["gemv_f32"];

        let m = a.shape[0] as i32;
        let k = a.shape[1] as i32;
        let alpha = 1.0f32;
        let beta = 0.0f32;

        let mut a_ptr = a.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m;
        let mut k_val = k;
        let mut alpha_val = alpha;
        let mut beta_val = beta;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut alpha_val as *mut _ as *mut c_void,
            &mut beta_val as *mut _ as *mut c_void,
        ];

        // One block per row, 256 threads per block with shared memory reduction
        let block_size = 256u32.min(k as u32);
        let shared_mem = block_size * 4; // one float per thread
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }

    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)
    pub fn gemv_q4k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q4k", kernels::GEMV_Q4K_SRC, "gemv_q4k")?;
        let func = &self.functions["gemv_q4k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 32u32; // single warp — no shared memory needed
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                0,
                None,
                &mut params,
            )
        }
    }

    /// y = A_q8_0 * x (quantized GEMV for Q8_0)
    pub fn gemv_q8_0(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q8_0", kernels::GEMV_Q8_0_SRC, "gemv_q8_0")?;
        let func = &self.functions["gemv_q8_0"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }

    /// y = A_q6k * x (quantized GEMV for Q6_K)
    pub fn gemv_q6k(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("gemv_q6k", kernels::GEMV_Q6K_SRC, "gemv_q6k")?;
        let func = &self.functions["gemv_q6k"];

        let mut a_ptr = a_raw.buf.as_ptr();
        let mut x_ptr = x.buf.as_ptr();
        let mut y_ptr = y.buf.as_ptr();
        let mut m_val = m as i32;
        let mut k_val = k as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32;
        let shared_mem = block_size * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }

    /// out = rmsnorm(x, weight, eps)
    pub fn rmsnorm_f32(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rmsnorm", kernels::RMSNORM_SRC, "rmsnorm_f32")?;
        let func = &self.functions["rmsnorm_f32"];

        let batch = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let mut x_ptr = x.buf.as_ptr();
        let mut w_ptr = weight.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(n as u32);
        let shared_mem = block_size * 4; // float per thread

        unsafe {
            self.hip.launch_kernel(
                func,
                [batch as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }

    /// c = a + b (element-wise)
    pub fn add_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("add", kernels::ADD_SRC, "add_f32")?;
        let func = &self.functions["add_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// c = a * b (element-wise)
    pub fn mul_f32(&mut self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("mul", kernels::MUL_SRC, "mul_f32")?;
        let func = &self.functions["mul_f32"];

        let n = a.numel() as i32;
        let mut a_ptr = a.buf.as_ptr();
        let mut b_ptr = b.buf.as_ptr();
        let mut c_ptr = c.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// out = silu(x)
    pub fn silu_f32(&mut self, x: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("silu", kernels::SILU_SRC, "silu_f32")?;
        let func = &self.functions["silu_f32"];

        let n = x.numel() as i32;
        let mut x_ptr = x.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// out = silu(gate) * up — fused to avoid intermediate buffer
    pub fn silu_mul_f32(&mut self, gate: &GpuTensor, up: &GpuTensor, out: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("silu_mul", kernels::SILU_MUL_SRC, "silu_mul_f32")?;
        let func = &self.functions["silu_mul_f32"];

        let n = gate.numel() as i32;
        let mut gate_ptr = gate.buf.as_ptr();
        let mut up_ptr = up.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        unsafe { self.hip.launch_kernel(func, [grid, 1, 1], [block, 1, 1], 0, None, &mut params) }
    }

    /// In-place softmax over last dimension
    pub fn softmax_f32(&mut self, x: &GpuTensor) -> HipResult<()> {
        self.ensure_kernel("softmax", kernels::SOFTMAX_SRC, "softmax_f32")?;
        let func = &self.functions["softmax_f32"];

        let rows = if x.shape.len() > 1 { x.shape[0] } else { 1 };
        let n = x.shape.last().copied().unwrap() as i32;

        let mut x_ptr = x.buf.as_ptr();
        let mut n_val = n;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];

        let block = 256u32.min(n as u32);
        let shared_mem = block * 4;

        unsafe {
            self.hip.launch_kernel(
                func,
                [rows as u32, 1, 1],
                [block, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }

    /// GPU-side RoPE (rotary positional embedding) applied in-place to Q and K.
    pub fn rope_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos: usize,
        n_heads_q: usize,
        n_heads_k: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> HipResult<()> {
        self.ensure_kernel("rope", kernels::ROPE_SRC, "rope_f32")?;
        let func = &self.functions["rope_f32"];

        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k.buf.as_ptr();
        let mut pos_val = pos as i32;
        let mut nhq = n_heads_q as i32;
        let mut nhk = n_heads_k as i32;
        let mut hd = head_dim as i32;
        let mut fb = freq_base;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut pos_val as *mut _ as *mut c_void,
            &mut nhq as *mut _ as *mut c_void,
            &mut nhk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut fb as *mut _ as *mut c_void,
        ];

        let half = (head_dim / 2) as u32;
        let block = 256u32.min(half);
        let grid = (half + block - 1) / block;

        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                None,
                &mut params,
            )
        }
    }

    /// GPU-side GQA attention.
    pub fn attention_f32(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        self.ensure_kernel("attention", kernels::ATTENTION_SRC, "attention_f32")?;
        let func = &self.functions["attention_f32"];

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut q_ptr = q.buf.as_ptr();
        let mut k_ptr = k_cache.buf.as_ptr();
        let mut v_ptr = v_cache.buf.as_ptr();
        let mut out_ptr = out.buf.as_ptr();
        let mut seq_val = seq_len as i32;
        let mut nh = n_heads as i32;
        let mut nkv = n_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut ms = max_seq as i32;
        let mut sc = scale;

        let mut params: Vec<*mut c_void> = vec![
            &mut q_ptr as *mut _ as *mut c_void,
            &mut k_ptr as *mut _ as *mut c_void,
            &mut v_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut seq_val as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ms as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];

        let block_size = 256u32.min(seq_len.max(head_dim) as u32);
        let shared_mem = ((seq_len + block_size as usize) * 4) as u32;

        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, 1, 1],
                [block_size, 1, 1],
                shared_mem,
                None,
                &mut params,
            )
        }
    }
}
