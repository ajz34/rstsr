//! Host-side readiness: [`CudaSyncAPI::synchronize`] for devices and tensors.

use crate::buffer::CudaBuffer;
use crate::device::DeviceCuda;
use crate::error::map_driver_result;
use crate::pinned::DeviceCudaPinned;
use crate::pinned::PinnedBuffer;
use crate::prelude_dev::*;

/// Block the host until the data of this device or tensor is observable.
pub trait CudaSyncAPI {
    /// Fallible form of [`CudaSyncAPI::synchronize`].
    fn synchronize_f(&self) -> Result<()>;

    /// Panicking form of [`CudaSyncAPI::synchronize_f`].
    fn synchronize(&self) {
        self.synchronize_f().rstsr_unwrap()
    }
}

impl CudaSyncAPI for DeviceCuda {
    /// Wait for this device value's execution stream.
    fn synchronize_f(&self) -> Result<()> {
        DeviceCuda::synchronize_f(self)
    }
}

impl CudaSyncAPI for DeviceCudaPinned {
    /// Nothing is pending on the device itself; the buffers order themselves.
    fn synchronize_f(&self) -> Result<()> {
        Ok(())
    }
}

impl<R, T, D> CudaSyncAPI for TensorAny<R, T, DeviceCuda, D>
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataAPI<Data = CudaBuffer<T>>,
{
    /// Wait for the execution stream, then for the buffer's allocation stream
    /// (host reads only observe the allocation stream).
    fn synchronize_f(&self) -> Result<()> {
        self.storage().device().synchronize_f()?;
        map_driver_result(self.storage().raw().alloc_stream().inner().synchronize())
    }
}

impl<R, T, D> CudaSyncAPI for TensorAny<R, T, DeviceCudaPinned, D>
where
    T: Send + Sync + 'static,
    D: DimAPI,
    R: DataAPI<Data = PinnedBuffer<T>>,
{
    /// Wait for every copy recorded on the pinned buffer.
    fn synchronize_f(&self) -> Result<()> {
        self.storage().raw().sync_f()
    }
}

/// Block the host until all work on `device`'s execution stream completes.
/// Panics through the error mapping; use [`synchronize_f`] for the fallible
/// form.
pub fn synchronize(device: &DeviceCuda) {
    device.synchronize()
}

/// Fallible form of [`synchronize`].
pub fn synchronize_f(device: &DeviceCuda) -> Result<()> {
    device.synchronize_f()
}

/// Block until the data of `tensor` (CUDA or pinned) is observable on the
/// host. Panics through the error mapping; use [`synchronize_tensor_f`] for
/// the fallible form.
pub fn synchronize_tensor<T: CudaSyncAPI>(tensor: &T) {
    tensor.synchronize()
}

/// Fallible form of [`synchronize_tensor`].
pub fn synchronize_tensor_f<T: CudaSyncAPI>(tensor: &T) -> Result<()> {
    tensor.synchronize_f()
}
