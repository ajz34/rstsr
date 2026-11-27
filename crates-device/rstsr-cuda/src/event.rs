//! Public event handle for user-side cross-stream ordering.

use crate::error::{map_driver_error, map_driver_result};
use crate::prelude_dev::*;
use crate::stream::CudaStreamObj;
use cudarc::driver::{sys, CudaContext, CudaEvent as DriverEvent};
use std::sync::Arc;

/// A CUDA event: a point in a stream that other streams can wait for.
///
/// - `record(stream)` captures "everything enqueued on `stream` so far"; `wait(stream)` makes
///   `stream` wait for the recorded point.
/// - `synchronize` blocks the host until the recorded point completes.
///
/// Created without timing support (`CU_EVENT_DISABLE_TIMING`), like the
/// crate's internal ordering events.
pub struct CudaEvent {
    inner: DriverEvent,
}

impl CudaEvent {
    pub(crate) fn new(inner: DriverEvent) -> Self {
        Self { inner }
    }

    /// Create an event on `device`'s context.
    pub fn new_f(device: &DeviceCuda) -> Result<Self> {
        device
            .ordinal_obj()
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .map(|inner| Self { inner })
            .map_err(map_driver_error)
    }

    /// The context this event belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.inner.context()
    }

    pub(crate) fn raw(&self) -> &DriverEvent {
        &self.inner
    }

    /// Record the event on `stream`. Panics through the error mapping; use
    /// [`CudaEvent::record_f`] for the fallible form.
    pub fn record(&self, stream: &CudaStreamObj) {
        self.record_f(stream).rstsr_unwrap()
    }

    /// Fallible form of [`CudaEvent::record`].
    pub fn record_f(&self, stream: &CudaStreamObj) -> Result<()> {
        map_driver_result(self.inner.record(stream.inner()))
    }

    /// Make `stream` wait for this event. Panics through the error mapping;
    /// use [`CudaEvent::wait_f`] for the fallible form.
    pub fn wait(&self, stream: &CudaStreamObj) {
        self.wait_f(stream).rstsr_unwrap()
    }

    /// Fallible form of [`CudaEvent::wait`].
    pub fn wait_f(&self, stream: &CudaStreamObj) -> Result<()> {
        map_driver_result(stream.inner().wait(&self.inner))
    }

    /// Block the host until the recorded point completes. Panics through the
    /// error mapping; use [`CudaEvent::synchronize_f`] for the fallible form.
    pub fn synchronize(&self) {
        self.synchronize_f().rstsr_unwrap()
    }

    /// Fallible form of [`CudaEvent::synchronize`].
    pub fn synchronize_f(&self) -> Result<()> {
        map_driver_result(self.inner.synchronize())
    }

    /// Whether the recorded point has completed without waiting; an event that
    /// was never recorded counts as complete. Panics through the error
    /// mapping; use [`CudaEvent::is_complete_f`] for the fallible form.
    pub fn is_complete(&self) -> bool {
        self.is_complete_f().rstsr_unwrap()
    }

    /// Fallible form of [`CudaEvent::is_complete`].
    pub fn is_complete_f(&self) -> Result<bool> {
        map_driver_result(self.inner.try_is_complete())
    }
}

impl core::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaEvent").finish_non_exhaustive()
    }
}
