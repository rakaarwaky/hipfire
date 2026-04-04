//! High-level compute dispatch — builds PM4 internally, handles kernarg layout.
//!
//! Users load a module, get kernels, and dispatch without touching PM4 packets.

use crate::device::{Device, GpuBuffer};
use crate::hsaco::{HsacoModule, KernelMeta};
use crate::queue::ComputeQueue;
use crate::{RedlineError, Result};

/// A loaded GPU module — ELF uploaded to VRAM with parsed kernels.
pub struct LoadedModule {
    pub kernels: Vec<Kernel>,
    pub code_buf: GpuBuffer,
}

/// A compute kernel ready for dispatch.
pub struct Kernel {
    pub name: String,
    pub code_va: u64,
    pub pgm_rsrc1: u32,
    pub pgm_rsrc2: u32,
    pub kernarg_size: u64,
    pub group_segment_size: u32,
    /// Total user SGPRs (private seg buf + dispatch ptr + kernarg ptr + ...)
    user_sgpr_count: u32,
    /// Index within user SGPRs where kernarg pointer goes (None if no kernarg)
    kernarg_sgpr_idx: Option<u32>,
}

/// A command buffer that accumulates PM4 dispatch packets.
pub struct CommandBuffer {
    pub(crate) dwords: Vec<u32>,
}

// PM4 helpers
fn pkt3(opcode: u32, body_count: u32) -> u32 {
    (3u32 << 30) | ((body_count - 1) << 16) | (opcode << 8) | (1 << 1) // SHADER_TYPE=1 (compute)
}

const SET_SH_REG: u32 = 0x76;
const DISPATCH_DIRECT: u32 = 0x15;
const RELEASE_MEM: u32 = 0x49;
const ACQUIRE_MEM: u32 = 0x58;

impl Device {
    /// Load a .hsaco module: parse ELF, upload code to VRAM, return ready-to-dispatch kernels.
    pub fn load_module(&self, hsaco_bytes: &[u8]) -> Result<LoadedModule> {
        let module = HsacoModule::from_bytes(hsaco_bytes.to_vec())?;
        let code_buf = self.alloc_vram(module.elf.len() as u64)?;
        self.upload(&code_buf, &module.elf)?;

        let kernels: Vec<Kernel> = module.kernels.iter().map(|km| {
            let kd_off = km.kd_offset as usize;
            let kcp = if kd_off + 58 <= module.elf.len() {
                u16::from_le_bytes([module.elf[kd_off + 56], module.elf[kd_off + 57]])
            } else {
                0
            };
            Kernel::from_meta(km, code_buf.gpu_addr, kcp)
        }).collect();

        Ok(LoadedModule { kernels, code_buf })
    }

    /// Load a .hsaco file from disk.
    pub fn load_module_file(&self, path: &str) -> Result<LoadedModule> {
        let data = std::fs::read(path)
            .map_err(|e| RedlineError { code: -1, message: format!("read {path}: {e}") })?;
        self.load_module(&data)
    }
}

impl Kernel {
    fn from_meta(km: &KernelMeta, code_buf_base: u64, kcp: u16) -> Self {
        let code_va = code_buf_base + km.code_offset;

        // Decode kernel_code_properties to determine user SGPR layout
        let mut count = 0u32;
        let mut kernarg_idx = None;

        if kcp & (1 << 0) != 0 { count += 4; } // private segment buffer
        if kcp & (1 << 1) != 0 { count += 2; } // dispatch ptr
        if kcp & (1 << 2) != 0 { count += 2; } // queue ptr
        if kcp & (1 << 3) != 0 {
            kernarg_idx = Some(count);
            count += 2; // kernarg segment ptr
        }
        if kcp & (1 << 4) != 0 { count += 2; } // dispatch id
        if kcp & (1 << 5) != 0 { count += 2; } // flat scratch init
        if kcp & (1 << 6) != 0 { count += 1; } // private segment size

        Kernel {
            name: km.name.clone(),
            code_va,
            pgm_rsrc1: km.pgm_rsrc1,
            pgm_rsrc2: km.pgm_rsrc2,
            kernarg_size: km.kernarg_size,
            group_segment_size: km.group_segment_size,
            user_sgpr_count: count,
            kernarg_sgpr_idx: kernarg_idx,
        }
    }

    /// Find a kernel by name in a loaded module.
    pub fn find<'a>(module: &'a LoadedModule, name: &str) -> Option<&'a Kernel> {
        module.kernels.iter().find(|k| k.name == name)
    }
}

impl CommandBuffer {
    pub fn new() -> Self {
        Self { dwords: Vec::with_capacity(512) }
    }

    /// Append a single dispatch to this command buffer.
    /// `kernarg_va`: GPU virtual address of the kernarg buffer for this dispatch.
    pub fn dispatch(&mut self, k: &Kernel, grid: [u32; 3], block: [u32; 3], kernarg_va: u64) {
        let d = &mut self.dwords;

        // COMPUTE_PGM_LO/HI
        d.push(pkt3(SET_SH_REG, 3));
        d.push(0x020C);
        d.push((k.code_va >> 8) as u32);
        d.push((k.code_va >> 40) as u32);

        // COMPUTE_PGM_RSRC1/RSRC2
        d.push(pkt3(SET_SH_REG, 3));
        d.push(0x0212);
        d.push(k.pgm_rsrc1);
        d.push(k.pgm_rsrc2);

        // COMPUTE_PGM_RSRC3 (GFX10 required)
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0228);
        d.push(0);

        // COMPUTE_TMPRING_SIZE = 0 (no scratch)
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0218);
        d.push(0);

        // COMPUTE_NUM_THREAD_X/Y/Z
        d.push(pkt3(SET_SH_REG, 4));
        d.push(0x0207);
        d.push(block[0]);
        d.push(block[1]);
        d.push(block[2]);

        // COMPUTE_RESOURCE_LIMITS = 0
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0215);
        d.push(0);

        // USER_DATA — fill with zeros, place kernarg pointer at the right index
        if k.user_sgpr_count > 0 {
            d.push(pkt3(SET_SH_REG, 1 + k.user_sgpr_count));
            d.push(0x0240); // COMPUTE_USER_DATA_0
            for i in 0..k.user_sgpr_count {
                if Some(i) == k.kernarg_sgpr_idx {
                    d.push(kernarg_va as u32);
                } else if Some(i) == k.kernarg_sgpr_idx.map(|x| x + 1) {
                    d.push((kernarg_va >> 32) as u32);
                } else {
                    d.push(0);
                }
            }
        }

        // DISPATCH_DIRECT
        // CS_EN=1 | CS_W32_EN=1 (HIP on RDNA always wave32)
        let di = (1u32 << 0) | (1 << 15);
        d.push(pkt3(DISPATCH_DIRECT, 4));
        d.push(grid[0]);
        d.push(grid[1]);
        d.push(grid[2]);
        d.push(di);
    }

    /// Append a dispatch with explicit dynamic LDS (shared memory) size.
    /// `lds_bytes` is the dynamic shared memory in bytes (added to kernel's static LDS).
    pub fn dispatch_with_lds(&mut self, k: &Kernel, grid: [u32; 3], block: [u32; 3],
                              kernarg_va: u64, lds_bytes: u32) {
        let d = &mut self.dwords;

        // COMPUTE_PGM_LO/HI
        d.push(pkt3(SET_SH_REG, 3));
        d.push(0x020C);
        d.push((k.code_va >> 8) as u32);
        d.push((k.code_va >> 40) as u32);

        // COMPUTE_PGM_RSRC1
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0212);
        d.push(k.pgm_rsrc1);

        // COMPUTE_PGM_RSRC2 with LDS_SIZE override
        // LDS_SIZE field is bits [20:14] — number of 512-byte blocks
        // Total LDS = kernel static + dynamic lds_bytes
        let total_lds = k.group_segment_size + lds_bytes;
        let lds_blocks = (total_lds + 511) / 512; // round up to 512-byte blocks
        let rsrc2_base = k.pgm_rsrc2 & !(0x7F << 14); // clear existing LDS_SIZE
        let rsrc2 = rsrc2_base | ((lds_blocks & 0x7F) << 14);

        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0213); // COMPUTE_PGM_RSRC2 offset (0x0212 + 1)
        d.push(rsrc2);

        // PGM_RSRC3
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0228);
        d.push(0);

        // TMPRING_SIZE
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0218);
        d.push(0);

        // NUM_THREAD
        d.push(pkt3(SET_SH_REG, 4));
        d.push(0x0207);
        d.push(block[0]);
        d.push(block[1]);
        d.push(block[2]);

        // RESOURCE_LIMITS
        d.push(pkt3(SET_SH_REG, 2));
        d.push(0x0215);
        d.push(0);

        // USER_DATA
        if k.user_sgpr_count > 0 {
            d.push(pkt3(SET_SH_REG, 1 + k.user_sgpr_count));
            d.push(0x0240);
            for i in 0..k.user_sgpr_count {
                if Some(i) == k.kernarg_sgpr_idx {
                    d.push(kernarg_va as u32);
                } else if Some(i) == k.kernarg_sgpr_idx.map(|x| x + 1) {
                    d.push((kernarg_va >> 32) as u32);
                } else {
                    d.push(0);
                }
            }
        }

        // DISPATCH_DIRECT
        let di = (1u32 << 0) | (1 << 15); // CS_EN | CS_W32_EN
        d.push(pkt3(DISPATCH_DIRECT, 4));
        d.push(grid[0]);
        d.push(grid[1]);
        d.push(grid[2]);
        d.push(di);
    }

    /// Insert a barrier between dispatches — waits for previous compute to finish.
    /// Uses CS_PARTIAL_FLUSH event. On same-context dispatch, L2 coherency is
    /// maintained by the hardware for write→read dependencies.
    pub fn barrier(&mut self) {
        let d = &mut self.dwords;
        // CS_PARTIAL_FLUSH: wait for all outstanding compute shaders on this context
        d.push(pkt3(0x46, 1));
        d.push(7); // EVENT_TYPE=CS_PARTIAL_FLUSH
    }

    /// Number of PM4 dwords in this command buffer.
    pub fn len_dwords(&self) -> u32 {
        self.dwords.len() as u32
    }

    /// Serialize to bytes for upload.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.dwords.iter().flat_map(|d| d.to_le_bytes()).collect()
    }
}

/// Convenience wrapper: dispatch queue with persistent IB + kernarg buffers.
pub struct DispatchQueue {
    pub queue: ComputeQueue,
    ib_buf: GpuBuffer,
    ka_buf: GpuBuffer,
}

const IB_SIZE: u64 = 64 * 1024; // 64KB IB buffer (plenty for hundreds of dispatches)
const KA_SIZE: u64 = 64 * 1024; // 64KB kernarg buffer

impl DispatchQueue {
    pub fn new(dev: &Device) -> Result<Self> {
        let queue = ComputeQueue::new(dev)?;
        let ib_buf = dev.alloc_vram(IB_SIZE)?;
        let ka_buf = dev.alloc_vram(KA_SIZE)?;
        Ok(Self { queue, ib_buf, ka_buf })
    }

    /// Single dispatch: upload args, build PM4, submit, wait.
    ///
    /// `args` should contain only the explicit kernel arguments.
    /// Hidden arguments (block counts, group sizes) are populated automatically.
    pub fn dispatch(
        &self,
        dev: &Device,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[u8],
        extra_bos: &[&GpuBuffer],
    ) -> Result<()> {
        // Build kernarg: explicit args + hidden args
        let ka_size = std::cmp::max(kernel.kernarg_size as usize, args.len());
        let mut ka_data = vec![0u8; std::cmp::max(ka_size, 256)];
        ka_data[..args.len()].copy_from_slice(args);

        // Populate hidden args if kernarg_size > explicit args
        // Layout (code object V5): block_count_x/y/z (u32×3), group_size_x/y/z (u16×3),
        // remainder_x/y/z (u16×3), then global_offset_x/y/z (u64×3), grid_dims (u16)
        let hidden_off = (args.len() + 7) & !7; // 8-byte aligned after explicit args
        if ka_size > hidden_off {
            let mut w = |off: usize, val: &[u8]| {
                if off + val.len() <= ka_data.len() {
                    ka_data[off..off + val.len()].copy_from_slice(val);
                }
            };
            // block_count_x/y/z (u32 each)
            w(hidden_off, &grid[0].to_le_bytes());
            w(hidden_off + 4, &grid[1].to_le_bytes());
            w(hidden_off + 8, &grid[2].to_le_bytes());
            // group_size_x/y/z (u16 each)
            w(hidden_off + 12, &(block[0] as u16).to_le_bytes());
            w(hidden_off + 14, &(block[1] as u16).to_le_bytes());
            w(hidden_off + 16, &(block[2] as u16).to_le_bytes());
            // remainder = 0 for uniform work groups (already zeroed)
            // grid_dims
            let ndims = if grid[2] > 1 { 3u16 } else if grid[1] > 1 { 2 } else { 1 };
            w(hidden_off + 64, &ndims.to_le_bytes());
        }
        dev.upload(&self.ka_buf, &ka_data)?;

        // Build PM4
        let mut cb = CommandBuffer::new();
        cb.dispatch(kernel, grid, block, self.ka_buf.gpu_addr);

        // Upload IB
        let ib_bytes = cb.as_bytes();
        dev.upload(&self.ib_buf, &ib_bytes)?;

        // Collect BO list
        let mut bos: Vec<&GpuBuffer> = vec![&self.ib_buf, &self.ka_buf];
        bos.extend_from_slice(extra_bos);

        self.queue.submit_and_wait(dev, &self.ib_buf, cb.len_dwords(), &bos)
    }

    /// Submit a pre-built command buffer. Caller manages kernarg separately.
    pub fn submit(
        &self,
        dev: &Device,
        cb: &CommandBuffer,
        bos: &[&GpuBuffer],
    ) -> Result<()> {
        let ib_bytes = cb.as_bytes();
        if ib_bytes.len() as u64 > IB_SIZE {
            return Err(RedlineError { code: -1, message: "command buffer exceeds IB size".into() });
        }
        dev.upload(&self.ib_buf, &ib_bytes)?;

        let mut all_bos: Vec<&GpuBuffer> = vec![&self.ib_buf];
        all_bos.extend_from_slice(bos);

        self.queue.submit_and_wait(dev, &self.ib_buf, cb.len_dwords(), &all_bos)
    }

    /// Get a reference to the persistent kernarg buffer.
    pub fn kernarg_buf(&self) -> &GpuBuffer {
        &self.ka_buf
    }

    pub fn destroy(self, dev: &Device) {
        self.queue.destroy(dev);
    }
}

/// Build a kernarg byte buffer from typed arguments.
/// Each arg is written at the correct alignment.
pub struct KernargBuilder {
    data: Vec<u8>,
}

impl KernargBuilder {
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0u8; capacity] }
    }

    pub fn write_u32(&mut self, offset: usize, val: u32) -> &mut Self {
        self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        self
    }

    pub fn write_u64(&mut self, offset: usize, val: u64) -> &mut Self {
        self.data[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
        self
    }

    pub fn write_f32(&mut self, offset: usize, val: f32) -> &mut Self {
        self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        self
    }

    pub fn write_ptr(&mut self, offset: usize, gpu_addr: u64) -> &mut Self {
        self.write_u64(offset, gpu_addr)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}
