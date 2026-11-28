//! The cuTENSOR library handle.

use crate::cutensor_wrap::error::{self, Stage};
use crate::prelude_dev::*;
use cudarc::cutensor::sys;
use std::sync::{Arc, Mutex, MutexGuard};

/// Whether `libcutensor` can be loaded on this machine (no panic). Under `dynamic-linking` the
/// library is a link-time dependency and this is always `true`.
pub(crate) fn libcutensor_available() -> bool {
    #[cfg(cutensor_dynamic_loading)]
    {
        // SAFETY: a loadability probe; `Library::new` is `unsafe` because of constructor functions
        // in dlopened objects, which this library does not use.
        unsafe { sys::is_culib_present() }
    }
    #[cfg(not(cutensor_dynamic_loading))]
    {
        true
    }
}

/// A cuTENSOR library handle bound to one ordinal's device, plus the lock that serializes the
/// planning calls the header leaves without a thread-safety remark.
///
/// cuTENSOR ties a handle (and every plan created through it) to the CUDA device that was active
/// at `cutensorCreate`; executing such a plan on another device fails with
/// `CUTENSOR_STATUS_ARCH_MISMATCH`. Handles are therefore per ordinal
/// ([`CudaOrdinalObj::cutensor`][crate::registry::CudaOrdinalObj::cutensor]), and one is shared by
/// every device value of that ordinal.
///
/// The header marks the execute calls and the
/// `cutensorCreate{Contraction,Reduction,Permutation,Elementwise*}` calls "thread-safe";
/// `cutensorCreatePlan`, `cutensorEstimateWorkspaceSize` and the attribute get/set entries carry no
/// remark, and the handle's plan cache is shared by all threads. [`CutensorHandle::lock`] is held
/// across the planning sequence and the attribute calls; the execute calls run unlocked, which is
/// what multi-stream overlap needs.
pub struct CutensorHandle {
    ptr: sys::cutensorHandle_t,
    plan_lock: Mutex<()>,
    version: (u32, u32, u32),
}

impl CutensorHandle {
    /// Create a handle on the device of `stream` (its context is made current first, as cuTENSOR
    /// binds the handle to the active device).
    pub fn new(stream: &Arc<CudaStreamObj>) -> Result<Self> {
        if !libcutensor_available() {
            return Err(error::not_loadable());
        }
        stream.inner().context().bind_to_thread().map_err(crate::error::map_driver_error)?;
        let mut handle = core::mem::MaybeUninit::<sys::cutensorHandle_t>::uninit();
        // SAFETY: `handle` is a valid output slot for the one-pointer `cutensorHandle_t`.
        let status = unsafe { sys::cutensorCreate(handle.as_mut_ptr()) };
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            return Err(error::status_error(Stage::CreateHandle, status, ""));
        }
        // SAFETY: the call wrote the handle on success.
        let ptr = unsafe { handle.assume_init() };
        Ok(Self { ptr, plan_lock: Mutex::new(()), version: Self::query_version() })
    }

    /// The cuTENSOR runtime version as `(major, minor, patch)` of the loaded library. Diagnostic
    /// (tests and benches print it); the wrapper gates nothing on it.
    pub fn version(&self) -> (u32, u32, u32) {
        self.version
    }

    /// The raw handle for the rest of the module. Every call the header leaves without a
    /// thread-safety remark must hold [`Self::lock`].
    #[allow(dead_code)] // wired by the B2 descriptors/plan
    pub(crate) fn ptr(&self) -> sys::cutensorHandle_t {
        self.ptr
    }

    /// Serialize one planning / attribute sequence.
    #[allow(dead_code)] // wired by the B2 planning sequence
    pub(crate) fn lock(&self) -> MutexGuard<'_, ()> {
        self.plan_lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn query_version() -> (u32, u32, u32) {
        // SAFETY: a pure version query; the library is loaded by the time a handle exists.
        let v = unsafe { sys::cutensorGetVersion() };
        ((v / 10000) as u32, ((v % 10000) / 100) as u32, (v % 100) as u32)
    }
}

impl Drop for CutensorHandle {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` came from `cutensorCreate` and is destroyed exactly once, before the
        // lock is torn down (fields drop in declaration order, so `plan_lock` — declared after
        // `ptr` — is still alive here, though no other thread can reach `self` any more).
        let status = unsafe { sys::cutensorDestroy(self.ptr) };
        debug_assert_eq!(status, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
    }
}

impl core::fmt::Debug for CutensorHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CutensorHandle").field("version", &self.version).finish_non_exhaustive()
    }
}

// SAFETY: the header marks the handle "thread-safe" (`cutensorCreate`: "blocking, no reentrant,
// and thread-safe"), and its plan cache is shared across threads by design. The entries it leaves
// unremarked are serialized by `plan_lock`, which is what makes sharing the handle sound.
unsafe impl Send for CutensorHandle {}
unsafe impl Sync for CutensorHandle {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Option<DeviceCuda> {
        if !crate::registry::is_available() {
            eprintln!("skipping: no CUDA device visible");
            return None;
        }
        Some(DeviceCuda::new(0))
    }

    /// A standalone handle is created and destroyed, and reports the library version.
    #[test]
    fn handle_creates_and_reports_version() {
        let Some(device) = test_device() else { return };
        if !libcutensor_available() {
            eprintln!("skipping: libcutensor is not loadable");
            return;
        }
        let handle = CutensorHandle::new(device.stream()).expect("cutensorCreate");
        let (major, _, _) = handle.version();
        assert_eq!(major, 2, "cuTENSOR 2.x runtime, got {:?}", handle.version());
    }

    /// The ordinal slot resolves to the same handle on every call, and declines to `None` (not a
    /// panic) when the library is absent.
    #[test]
    fn ordinal_slot_is_canonical() {
        let Some(device) = test_device() else { return };
        let obj = device.ordinal_obj();
        let first = obj.cutensor();
        let second = obj.cutensor();
        match (first, second) {
            (Some(a), Some(b)) => {
                assert!(Arc::ptr_eq(a, b), "the ordinal slot is canonical");
                assert_eq!(a.version(), b.version());
            },
            (None, None) => eprintln!("skipping: libcutensor is not loadable"),
            _ => panic!("the ordinal slot must be stable"),
        }
    }
}
