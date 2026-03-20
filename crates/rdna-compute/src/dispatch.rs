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
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
pub struct Gpu {
    pub hip: HipRuntime,
    compiler: KernelCompiler,
    modules: HashMap<String, hip_bridge::Module>,
    functions: HashMap<String, hip_bridge::Function>,
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

    pub fn alloc_tensor(&self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        let numel: usize = shape.iter().product();
        let byte_size = numel * dtype.size();
        let buf = self.hip.malloc(byte_size)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    pub fn upload_f32(&self, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
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

    pub fn zeros(&self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        let tensor = self.alloc_tensor(shape, dtype)?;
        self.hip.memset(&tensor.buf, 0, tensor.byte_size())?;
        Ok(tensor)
    }

    pub fn free_tensor(&self, tensor: GpuTensor) -> HipResult<()> {
        self.hip.free(tensor.buf)
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
}
