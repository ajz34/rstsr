//! The device workspace of the cuTENSOR execute calls.

use crate::buffer::CudaBuffer;
use crate::prelude_dev::*;
use std::sync::Arc;

/// An owning device workspace passed to the cuTENSOR execute calls.
///
/// `cuMemAlloc`-aligned (256 bytes), stream-ordered free on the allocation stream. cuTENSOR
/// documents the workspace as optional — "it is still recommended to provide some small workspace
/// (e.g., 128 MB)" — and non-GEMM `f64` contractions use 32-36 MiB of it. Every execute call of
/// this module must see a workspace at least as large as the plan's
/// `required_workspace` (the planning of [`crate::cutensor_wrap`] takes the workspace size as the
/// plan limit).
///
/// The default is [`crate::flags::DEFAULT_CUTENSOR_WORKSPACE_BYTES`] per stream
/// ([`CudaStreamObj::cutensor_workspace`][crate::stream::CudaStreamObj::cutensor_workspace], sized
/// by `CudaRuntimeFlags::cutensor_workspace_bytes`); `CutensorWorkspace` is for callers that want
/// their own size or lifetime.
#[derive(Clone)]
pub struct CutensorWorkspace {
    buf: Arc<CudaBuffer<u8>>,
}

impl CutensorWorkspace {
    /// Fallible allocation of `bytes` on `stream` (the module has no panicking twin).
    pub fn new_f(stream: &Arc<CudaStreamObj>, bytes: usize) -> Result<Self> {
        Ok(Self { buf: Arc::new(CudaBuffer::new_f(stream, bytes)?) })
    }

    /// Wrap the per-stream default slot buffer.
    pub fn from_slot(buf: Arc<CudaBuffer<u8>>) -> Self {
        Self { buf }
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the workspace is empty (zero bytes).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The allocation stream (owns the free ordering).
    pub fn alloc_stream(&self) -> Arc<CudaStreamObj> {
        self.buf.alloc_stream()
    }

    /// Device pointer for an execute call on `exec`, with the read-event guard handed back: the
    /// caller keeps it alive until after the launch, so a free on the allocation stream (a
    /// caller-owned workspace dropped right after the call) cannot race the kernel.
    pub(crate) fn ptr_on<'a>(
        &'a self,
        exec: &'a Arc<CudaStreamObj>,
    ) -> (*mut core::ffi::c_void, cudarc::driver::SyncOnDrop<'a>) {
        let (ptr, guard) = self.buf.read_ptr_on(exec);
        (ptr as *mut core::ffi::c_void, guard)
    }
}

impl core::fmt::Debug for CutensorWorkspace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CutensorWorkspace").field("len", &self.len()).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::DEFAULT_CUTENSOR_WORKSPACE_BYTES;

    fn device_or_skip() -> Option<DeviceCuda> {
        if !crate::registry::is_available() {
            eprintln!("skipping: no CUDA device visible");
            return None;
        }
        if !crate::cutensor_wrap::handle::libcutensor_available() {
            eprintln!("skipping: libcutensor is not loadable");
            return None;
        }
        Some(DeviceCuda::new(0))
    }

    #[test]
    fn workspace_allocates_on_hardware() {
        let Some(device) = device_or_skip() else { return };
        let ws = CutensorWorkspace::new_f(device.stream(), 1 << 20).expect("allocate 1 MiB");
        assert_eq!(ws.len(), 1 << 20);
        assert!(!ws.is_empty());

        // The per-stream default slot: sized by the device's flag (128 MiB by default), canonical
        // per stream, and wrapped into a `CutensorWorkspace` for the execute calls.
        let bytes = device.flags().cutensor_workspace_bytes;
        assert_eq!(bytes, DEFAULT_CUTENSOR_WORKSPACE_BYTES);
        let slot = device.stream().cutensor_workspace(bytes).expect("the default workspace slot");
        assert_eq!(slot.len(), DEFAULT_CUTENSOR_WORKSPACE_BYTES);
        let again = device.stream().cutensor_workspace(bytes).expect("the slot is canonical");
        assert!(Arc::ptr_eq(slot, again));
        let wrapped = CutensorWorkspace::from_slot(slot.clone());
        assert_eq!(wrapped.len(), DEFAULT_CUTENSOR_WORKSPACE_BYTES);
    }

    /// With the flag at `0`, the default slot is `None` (run without workspace). The test uses a
    /// fresh stream and a device-value flag override, so it races nothing global.
    #[test]
    fn zero_flag_means_no_default_workspace() {
        let Some(device) = device_or_skip() else { return };
        let flags = CudaRuntimeFlags { cutensor_workspace_bytes: 0, ..Default::default() };
        let device = device.with_flags(flags);
        let stream = Arc::new(CudaStreamObj::new_f(&device).expect("fresh stream"));
        assert!(stream.cutensor_workspace(0).is_none(), "the 0 flag is no workspace");
    }
}
