//! Zero-copy host views over pinned tensors ([`TensorPinnedHostViewAPI`]).

use crate::pinned::buffer::PinnedBuffer;
use crate::pinned::device::DeviceCudaPinned;
use crate::prelude_dev::*;
use core::mem::ManuallyDrop;

/// Read-only host views over a pinned tensor.
///
/// The views borrow the pinned bytes (no copy) and carry the tensor's layout.
/// Each call synchronizes the buffer's event first, so the host never reads
/// bytes a DMA is still writing.
pub trait TensorPinnedHostViewAPI<'a, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
{
    /// `DeviceCpuSerial` view over the pinned bytes.
    fn host_view(&'a self) -> TensorView<'a, T, DeviceCpuSerial, D>;

    /// View over the pinned bytes on any `Raw = Vec<T>` device.
    fn host_view_on<B>(&'a self, device: &B) -> TensorView<'a, T, B, D>
    where
        B: DeviceAPI<T, Raw = Vec<T>>;
}

/// Mutable host views over a pinned tensor (see [`TensorPinnedHostViewAPI`]).
pub trait TensorPinnedHostViewMutAPI<'a, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
{
    /// Mutable `DeviceCpuSerial` view over the pinned bytes.
    fn host_view_mut(&'a mut self) -> TensorViewMut<'a, T, DeviceCpuSerial, D>;

    /// Mutable view over the pinned bytes on any `Raw = Vec<T>` device.
    fn host_view_mut_on<B>(&'a mut self, device: &B) -> TensorViewMut<'a, T, B, D>
    where
        B: DeviceAPI<T, Raw = Vec<T>>;
}

impl<'a, R, T, D> TensorPinnedHostViewAPI<'a, T, D> for TensorAny<R, T, DeviceCudaPinned, D>
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataAPI<Data = PinnedBuffer<T>>,
{
    fn host_view(&'a self) -> TensorView<'a, T, DeviceCpuSerial, D> {
        self.host_view_on(self.device().base())
    }

    fn host_view_on<B>(&'a self, device: &B) -> TensorView<'a, T, B, D>
    where
        B: DeviceAPI<T, Raw = Vec<T>>,
    {
        self.raw().sync();
        // SAFETY: the `Vec` aliases the whole pinned buffer (`ptr`, `len`, capacity `len`) for
        // `'a`, which the borrowed `PinnedBuffer` guarantees; the event sync above means no DMA
        // touches the bytes while the view lives. It lives in a `ManuallyDrop` and is never
        // deallocated or resized (the `asarray(&[T])` contract).
        let raw = unsafe { Vec::from_raw_parts(self.raw().ptr(), self.raw().len(), self.raw().len()) };
        let storage = Storage::new(DataRef::from_manually_drop(ManuallyDrop::new(raw)), device.clone());
        let layout = self.layout().clone();
        TensorAny::new_f(storage, layout).rstsr_unwrap()
    }
}

impl<'a, R, T, D> TensorPinnedHostViewMutAPI<'a, T, D> for TensorAny<R, T, DeviceCudaPinned, D>
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataMutAPI<Data = PinnedBuffer<T>>,
{
    fn host_view_mut(&'a mut self) -> TensorViewMut<'a, T, DeviceCpuSerial, D> {
        let base = self.device().base().clone();
        self.host_view_mut_on(&base)
    }

    fn host_view_mut_on<B>(&'a mut self, device: &B) -> TensorViewMut<'a, T, B, D>
    where
        B: DeviceAPI<T, Raw = Vec<T>>,
    {
        self.raw_mut().sync();
        // `as_mut_slice` keeps the `&mut` provenance for later mutable re-derivations.
        let ptr = self.raw_mut().ptr();
        let len = self.raw().len();
        // SAFETY: as `fabricate_vec`, over the exclusive `&mut` borrow; wrapped in `ManuallyDrop`
        // so it is never dropped or resized.
        let raw = unsafe { Vec::from_raw_parts(ptr, len, len) };
        let storage = Storage::new(DataMut::from_manually_drop(ManuallyDrop::new(raw)), device.clone());
        let layout = self.layout().clone();
        TensorAny::new_f(storage, layout).rstsr_unwrap()
    }
}
