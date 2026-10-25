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
        let raw = CudaBuffer::new_f(self.stream(), len)?;
        Ok(Storage::new(DataOwned::from(raw), self.clone()))
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<DataOwned<CudaBuffer<T>>, T, Self>>
    where
        T: Clone + Send + Sync + 'static,
    {
        // kernel fill for the TypeIds `is_fill_supported` lists; other T
        // (half, complex, i128/u128) keep the host geometric-doubling path
        if crate::kernels_ops::is_fill_supported(&fill) {
            let mut raw = CudaBuffer::new_f(self.stream(), len)?;
            let layout = Layout::new(vec![len], vec![1_isize], 0)?;
            crate::kernels_ops::fill_bytes_erased(self, &mut raw, &layout, &fill)?;
            return Ok(Storage::new(DataOwned::from(raw), self.clone()));
        }
        // DEFER: extend the TypeId list to all CudaDType members or move the
        // bound onto a dedicated creation trait fork
        let mut raw = CudaBuffer::new_f(self.stream(), len)?;
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
        let raw = CudaBuffer::<MaybeUninit<T>>::new_f(self.stream(), len)?;
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
        let mut raw = CudaBuffer::new_f(self.stream(), len)?;
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
        // length: ceil((end - start) / step) computed in the operand dtype
        // (NumPy `_calc_length`); elements [i] = start + step*i in-kernel
        let n = arange_len(&start, &end, &step)?;
        let buf = crate::kernels_ops::sequence(self, n, start, step)?;
        Ok(Storage::new(DataOwned::from(buf), self.clone()))
    }
}

/// NumPy `arange` length, computed without a widening round-trip: integer
/// ranges use exact i128 ceil-division (f64 loses all precision above
/// 2^53), float ranges use the operand width.
fn arange_len<T>(start: &T, end: &T, step: &T) -> Result<usize>
where
    T: crate::dtype::CudaDType
        + PartialOrd
        + Clone
        + core::ops::Sub<Output = T>
        + core::ops::Div<Output = T>
        + num::ToPrimitive
        + num::Zero,
{
    rstsr_assert!(*step != T::zero(), InvalidValue, "arange step must be nonzero")?;
    if crate::dtype::is_int_cname(T::CNAME) {
        let (s, e, st) = match (start.to_i128(), end.to_i128(), step.to_i128()) {
            (Some(s), Some(e), Some(st)) => (s, e, st),
            _ => return Ok(0),
        };
        let span = e.wrapping_sub(s);
        if span == 0 || (span > 0) != (st > 0) {
            return Ok(0);
        }
        let n = (span.unsigned_abs() + st.unsigned_abs() - 1) / st.unsigned_abs();
        return Ok(n.min(usize::MAX as u128) as usize);
    }
    let n = match T::CNAME {
        "float" => {
            let (s, e, st) = (start.to_f32(), end.to_f32(), step.to_f32());
            match (s, e, st) {
                (Some(s), Some(e), Some(st)) if st != 0.0 => float_len_f32((e - s) / st),
                _ => 0,
            }
        },
        // NumPy rounds the quotient in the scalar dtype before the ceil; a
        // non-finite half quotient (NumPy raises) falls back to the widened
        // span
        "__half" | "__nv_bfloat16" => {
            let span = (*end - *start) / *step;
            match span.to_f32() {
                Some(s) if s.is_finite() => float_len_f32(s),
                _ => match (start.to_f32(), end.to_f32(), step.to_f32()) {
                    (Some(s), Some(e), Some(st)) if st != 0.0 => float_len_f32((e - s) / st),
                    _ => 0,
                },
            }
        },
        _ => {
            let (s, e, st) = (start.to_f64(), end.to_f64(), step.to_f64());
            match (s, e, st) {
                (Some(s), Some(e), Some(st)) if st != 0.0 => float_len_f64((e - s) / st),
                _ => 0,
            }
        },
    };
    Ok(n)
}

fn float_len_f32(span: f32) -> usize {
    if span <= 0.0 {
        0
    } else {
        span.ceil() as usize
    }
}

fn float_len_f64(span: f64) -> usize {
    if span <= 0.0 {
        0
    } else {
        span.ceil() as usize
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
