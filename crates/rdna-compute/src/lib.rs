//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

mod compiler;
mod dispatch;
mod kernels;

pub use compiler::KernelCompiler;
pub use dispatch::{DType, Gpu, GpuTensor};
pub use kernels::GEMV_SRC;
