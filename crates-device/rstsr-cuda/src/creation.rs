//! Creation traits for `DeviceCuda`.

use crate::buffer::CudaBuffer;
use crate::prelude_dev::*;
use crate::transfer;
use num::Num;

impl<T> DeviceCreationAnyAPI<T> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: Clone + Send + Sync + 'static,
{
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        let raw = CudaBuffer::new(self.stream(), len)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>>
    where
        T: Clone,
    {
        // DEFER: device fill kernel; correctness does not depend on this
        let mut raw = CudaBuffer::new(self.stream(), len)?;
        transfer::fill_naive(&mut raw, &fill)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        let raw = transfer::h2d_full(self, vec)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>>
    where
        T: Clone,
    {
        self.outof_cpu_vec(vec.to_vec())
    }

    fn uninit_impl(&self, len: usize) -> Result<Storage<DataOwned<CudaBuffer<MaybeUninit<T>>>, MaybeUninit<T>, Self>> {
        let raw = CudaBuffer::<MaybeUninit<T>>::new(self.stream(), len)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    unsafe fn assume_init_impl(
        storage: Storage<DataOwned<CudaBuffer<MaybeUninit<T>>>, MaybeUninit<T>, Self>,
    ) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        // Layout-identical rebuild (no transmute of internals): leak the
        // pointer out of the MaybeUninit wrapper and re-upgrade it as `u8`
        // bytes under a fresh event set.
        let (data, device) = storage.into_raw_parts();
        let buf: CudaBuffer<MaybeUninit<T>> = data.into_raw();
        let len = buf.len();
        let stream = buf.alloc_stream().inner().clone();
        let ptr = buf.leak_ptr();
        let slice = stream.upgrade_device_ptr::<u8>(ptr, len * core::mem::size_of::<T>());
        let raw = CudaBuffer::<T> { slice, len, _m: core::marker::PhantomData };
        Ok(Storage::new(DataOwned::from(raw), device))
    }
}

impl<T> DeviceCreationNumAPI<T> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>,
    T: Num + Clone + Send + Sync + 'static,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        let mut raw = CudaBuffer::new(self.stream(), len)?;
        transfer::memset_zero(&mut raw)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>>
    where
        T: Clone,
    {
        self.full_impl(len, T::one())
    }
}
