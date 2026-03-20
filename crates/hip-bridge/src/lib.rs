//! hip-bridge: Safe Rust FFI to AMD HIP runtime via dlopen.
//! Modeled after rustane's ane-bridge — no link-time dependency on libamdhip64.

mod ffi;
mod error;

pub use error::{HipError, HipResult};
pub use ffi::{Event, Function, HipRuntime, Module, Stream};

/// Re-export memory copy direction for callers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// Opaque GPU buffer handle. Tracks pointer + size for safety.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

// DeviceBuffer is Send — GPU pointers can be sent between threads.
// They are NOT Sync — concurrent access requires stream synchronization.
unsafe impl Send for DeviceBuffer {}
