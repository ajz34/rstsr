//! Creation traits for `DeviceCuda`.

use crate::buffer::CudaBuffer;
use crate::prelude_dev::*;
use crate::transfer;
use num::Num;

impl<T> DeviceCreationAnyAPI<T> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: Copy + Send + Sync + 'static,
{
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        let raw = CudaBuffer::new(self.stream(), len)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>>
    where
        T: Clone + Send + Sync + 'static,
    {
        // kernel fill for the TypeIds `is_fill_supported` lists; other T
        // (half, complex, i128/u128) keep the host geometric-doubling path
        if crate::kernels_ops::is_fill_supported(&fill) {
            let mut raw = CudaBuffer::new(self.stream(), len)?;
            let layout = Layout::new(vec![len], vec![1_isize], 0)?;
            crate::kernels_ops::fill_bytes_erased(self, &mut raw, &layout, &fill)?;
            return Ok(Storage::new(DataOwned::from(raw), self.clone()));
        }
        // DEFER: extend the TypeId list to all CudaDType members or move the
        // bound onto a dedicated creation trait fork
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
    T: Num + Copy + Send + Sync + 'static,
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

impl<T> DeviceCreationArangeAPI<T> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>,
    T: crate::dtype::CudaDType
        + PartialOrd
        + Clone
        + core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Div<Output = T>
        + num::ToPrimitive
        + num::Zero
        + Send
        + Sync
        + 'static,
{
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        // length: ceil((end - start) / step) in f64 — one division, matching
        // the accelerated CPU path; elements [i] = start + step*i in-kernel
        let n = {
            let zero = T::zero();
            rstsr_assert!(step != zero, InvalidValue, "arange step must be nonzero")?;
            let fs = start.to_f64().unwrap_or(f64::NAN);
            let fe = end.to_f64().unwrap_or(f64::NAN);
            let fst = step.to_f64().unwrap_or(f64::NAN);
            let span = (fe - fs) / fst;
            if span <= 0.0 {
                0usize
            } else {
                span.ceil() as usize
            }
        };
        // DEFER: step sign/zero validation mirrors the tensor layer; the
        // kernel writes start + step*i directly.
        let buf = crate::kernels_ops::sequence(self, n, start, step)?;
        Ok(Storage::new(DataOwned::from(buf), self.clone()))
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>,
    T: crate::dtype::CudaDType + num::complex::ComplexFloat + Clone + Send + Sync + 'static,
{
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>> {
        let buf = crate::kernels_ops::linspace(self, n, start, end, endpoint)?;
        Ok(Storage::new(DataOwned::from(buf), self.clone()))
    }
}
