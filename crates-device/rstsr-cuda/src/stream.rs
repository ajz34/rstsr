//! Stream wrapper for `DeviceCuda`.

use crate::error::{map_driver_error, map_driver_result};
use crate::prelude_dev::*;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use std::sync::Arc;

/// A CUDA stream of one ordinal. Identity is (handle, context); the device
/// value's stream participates in `same_device` strictly.
#[derive(Debug)]
pub struct CudaStreamObj {
    pub(crate) stream: Arc<CudaStream>,
    /// Cached ordinal of the owning context (for equality and display).
    pub(crate) ordinal: usize,
}

impl CudaStreamObj {
    /// Create a new stream on `ctx`.
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.new_stream().map_err(map_driver_error)?;
        Ok(Self { stream, ordinal: ctx.ordinal() })
    }

    /// The underlying stream handle (launches, copies, synchronization).
    pub fn inner(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Wrap an existing cudarc stream of the given ordinal (test/interop
    /// helper; production code goes through `CudaOrdinalObj::new`).
    pub fn wrap(ordinal: usize, stream: Arc<CudaStream>) -> Self {
        Self { stream, ordinal }
    }

    /// The ordinal of this stream's context.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Block the host until all work submitted to this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        map_driver_result(self.stream.synchronize())
    }
}

impl PartialEq for CudaStreamObj {
    fn eq(&self, other: &Self) -> bool {
        self.stream == other.stream
    }
}
impl Eq for CudaStreamObj {}
