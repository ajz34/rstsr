//! Creation methods for `Tensor` struct.
//!
//! This module relates to the [Python array API standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/creation_functions.html).
//!
//! Todo list:
//! - [x] `arange`: [`arange`], [`arange_int`]
//! - [ ] `asarray`
//! - [x] `empty`: [`empty`]
//! - [x] `empty_like`: [`empty_like`]
//! - [ ] `eye`
//! - [ ] ~`from_dlpack`~
//! - [x] `full`: [`full`]
//! - [x] `full_like`: [`full_like`]
//! - [x] `linspace`: [`linspace`]
//! - [ ] `meshgrid`
//! - [x] `ones`: [`ones`]
//! - [x] `ones_like`: [`ones_like`]
//! - [ ] `tril`
//! - [ ] `triu`
//! - [x] `zeros`: [`zeros`]
//! - [x] `zeros_like`: [`zeros_like`]

use crate::cpu_backend::device::CpuDevice;
use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{Float, Num};

/* #region arange */

/// Evenly spaced values within the half-open interval `[start, stop)` as
/// one-dimensional array.
///
/// # See also
///
/// - [`arange_cpu`] for CPU-only version.
/// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html#array_api.arange)
/// - [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
pub fn arange<T, B>(start: T, stop: T, step: T, device: &B) -> Tensor<T, Ix1, B>
where
    T: Float,
    B: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    let data = B::arange_impl(device, start, stop, step).unwrap();
    let layout = [data.len()].into();
    Tensor::new(data.into(), layout).unwrap()
}

/// Evenly spaced values within the half-open interval `[start, stop)` as
/// one-dimensional array.
///
/// # See also
///
/// - [`arange`] for other devices.
/// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html#array_api.arange)
/// - [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
pub fn arange_cpu<T>(start: T, stop: T, step: T) -> Tensor<T, Ix1, CpuDevice>
where
    T: Float,
    CpuDevice: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    let data = CpuDevice::arange_impl(&CpuDevice {}, start, stop, step).unwrap();
    let layout = [data.len()].into();
    Tensor::new(data.into(), layout).unwrap()
}

/* #endregion */

/* #region arange_int */

/// Evenly spaced values within the half-open interval `[0, len)` as
/// one-dimensional array, each step 1.
pub fn arange_int<T, B>(len: usize, device: &B) -> Tensor<T, Ix1, B>
where
    T: Float,
    B: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    let data = B::arange_int_impl(device, len).unwrap();
    let layout = [data.len()].into();
    Tensor::new(data.into(), layout).unwrap()
}

/// Evenly spaced values within the half-open interval `[0, len)` as
/// one-dimensional array, each step 1.
pub fn arange_int_cpu<T>(len: usize) -> Tensor<T, Ix1, CpuDevice>
where
    T: Float,
    CpuDevice: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    arange_int(len, &CpuDevice {})
}

/* #endregion arange_int */

/* #region empty */

/// Uninitialized tensor having a specified shape.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
///
/// # See also
///
/// - [Python array API standard: `empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html)
pub unsafe fn empty<T, D, B>(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let layout = layout.into();
    let (_, idx_max) = layout.bounds_index().unwrap();
    let data: Storage<T, B> = B::empty_impl(device, idx_max).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// Uninitialized tensor having a specified shape.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor with uninitialized.
///
/// # See also
///
/// - [Python array API standard: `empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html)
pub unsafe fn empty_cpu<T, D>(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug,
    D: DimAPI,
{
    empty(layout, &CpuDevice {})
}

/* #endregion */

/* #region empty_like */

/// Uninitialized tensor with the same shape as an input tensor.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor withuninitialized.
pub unsafe fn empty_like<R, T, D, B>(tensor: &TensorBase<R, D>, device: &B) -> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let shape = tensor.layout().shape();
    let layout = shape.new_contig(0);
    let idx_max = layout.size();
    let data: Storage<T, _> = device.empty_impl(idx_max).unwrap();
    unsafe { Tensor::new_unchecked(data.into(), layout) }
}

/// Uninitialized tensor with the same shape as an input tensor.
///
/// # Safety
///
/// This function is unsafe because it creates a tensor withuninitialized.
pub unsafe fn empty_like_cpu<R, T, D>(tensor: &TensorBase<R, D>) -> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug,
    D: DimAPI,
{
    empty_like(tensor, &CpuDevice {})
}

/* #endregion */

/* #region full */

/// New tensor having a specified shape and filled with given value.
///
/// # See also
///
/// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
pub fn full<T, D, B>(layout: impl Into<Layout<D>>, fill: T, device: &B) -> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let layout = layout.into();
    let idx_max = layout.size();
    let data: Storage<T, _> = device.full_impl(idx_max, fill).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// New tensor having a specified shape and filled with given value.
///
/// # See also
///
/// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
pub fn full_cpu<T, D>(layout: impl Into<Layout<D>>, fill: T) -> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug,
    D: DimAPI,
{
    full(layout, fill, &CpuDevice {})
}

/* #endregion */

/* #region full_like */

/// New tensor filled with given value and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
pub fn full_like<R, T, D, B>(tensor: &TensorBase<R, D>, fill: T, device: &B) -> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    let shape = tensor.layout().shape();
    let layout = shape.new_contig(0);
    let idx_max = layout.size();
    let data: Storage<T, _> = device.full_impl(idx_max, fill).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// New tensor filled with given value and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
pub fn full_like_cpu<R, T, D>(tensor: &TensorBase<R, D>, fill: T) -> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug,
    D: DimAPI,
{
    full_like(tensor, fill, &CpuDevice {})
}

/* #endregion */

/* #region linspace */

/// Evenly spaced numbers over a specified interval.
///
/// For boundary condition, current implementation is similar to numpy,
/// where `n = 0` will return an empty array, and `n = 1` will return an
/// array with starting value.
///
/// # See also
///
/// - [numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
/// - [Python array API standard: `linspace`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.linspace.html)
pub fn linspace<T, B>(start: T, end: T, n: usize, device: &B) -> Tensor<T, Ix1, B>
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    let data = B::linspace_impl(device, start, end, n).unwrap();
    let layout = [data.len()].into();
    Tensor::new(data.into(), layout).unwrap()
}

/// Evenly spaced numbers over a specified interval.
///
/// For boundary condition, current implementation is similar to numpy,
/// where `n = 0` will return an empty array, and `n = 1` will return an
/// array with starting value.
///
/// # See also
///
/// - [numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
/// - [Python array API standard: `linspace`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.linspace.html)
pub fn linspace_cpu<T>(start: T, end: T, n: usize) -> Tensor<T, Ix1, CpuDevice>
where
    T: ComplexFloat,
    CpuDevice: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    linspace(start, end, n, &CpuDevice {})
}

/* #endregion */

/* #region ones */

/// New tensor filled with ones and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
pub fn ones<T, D, B>(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let layout = layout.into();
    let (_, idx_max) = layout.bounds_index().unwrap();
    let data: Storage<T, _> = B::ones_impl(device, idx_max).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// New tensor filled with ones and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
pub fn ones_cpu<T, D>(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    ones(layout, &CpuDevice {})
}

/* #endregion */

/* #region ones_like */

/// New tensor filled with ones and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
pub fn ones_like<R, T, D, B>(tensor: &TensorBase<R, D>, device: &B) -> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let shape = tensor.layout().shape();
    let layout = shape.new_contig(0);
    let idx_max = layout.size();
    let data: Storage<T, _> = B::ones_impl(device, idx_max).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// New tensor filled with ones and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
pub fn ones_like_cpu<R, T, D>(tensor: &TensorBase<R, D>) -> Tensor<T, D, CpuDevice>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    ones_like(tensor, &CpuDevice {})
}

/* #endregion */

/* #region zeros */

/// New tensor filled with zeros and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
pub fn zeros<T, D, B>(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let layout = layout.into();
    let (_, idx_max) = layout.bounds_index().unwrap();
    let data: Storage<T, _> = B::zeros_impl(device, idx_max).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// New tensor filled with zeros and having a specified shape.
///
/// # See also
///
/// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
pub fn zeros_cpu<T, D>(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    zeros(layout, &CpuDevice {})
}

/* #endregion */

/* #region zeros_like */

/// New tensor filled with zeros and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
pub fn zeros_like<R, T, D, B>(tensor: &TensorBase<R, D>, device: &B) -> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    let shape = tensor.layout().shape();
    let layout = shape.new_contig(0);
    let idx_max = layout.size();
    let data: Storage<T, _> = B::zeros_impl(device, idx_max).unwrap();
    Tensor::new(data.into(), layout).unwrap()
}

/// New tensor filled with zeros and having the same shape as an input
/// tensor.
///
/// # See also
///
/// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
pub fn zeros_like_cpu<R, T, D>(tensor: &TensorBase<R, D>) -> Tensor<T, D, CpuDevice>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    zeros_like(tensor, &CpuDevice {})
}

/* #endregion */

/// Methods for array creation.
impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Uninitialized tensor with the same shape as an input tensor.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor withuninitialized.
    pub unsafe fn empty_like(&self) -> Tensor<T, D, B>
    where
        B: DeviceCreationAnyAPI<T>,
    {
        empty_like(self, &self.data().as_storage().device())
    }

    /// New tensor filled with given value and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
    pub fn full_like(&self, fill: T) -> Tensor<T, D, B>
    where
        B: DeviceCreationAnyAPI<T>,
    {
        full_like(self, fill, &self.data().as_storage().device())
    }

    /// New tensor filled with ones and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
    pub fn ones_like(&self) -> Tensor<T, D, B>
    where
        T: Num,
        B: DeviceCreationNumAPI<T>,
    {
        ones_like(self, &self.data().as_storage().device())
    }

    /// New tensor filled with zeros and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
    pub fn zeros_like(&self) -> Tensor<T, D, B>
    where
        T: Num,
        B: DeviceCreationNumAPI<T>,
    {
        zeros_like(self, &self.data().as_storage().device())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn playground() {
        let a = arange_cpu(2.5, 3.2, 0.02);
        println!("{a:6.3?}");
        let a = arange_int_cpu::<f64>(15);
        println!("{a:6.3?}");
        let a = unsafe { empty_cpu::<f64, _>([15, 18].f()) };
        println!("{a:6.3?}");
        let a = unsafe { a.empty_like() };
        println!("{a:6.3?}");
        let a = full_cpu([2, 2].f(), 3.16);
        println!("{a:6.3?}");
        let a = full_like_cpu(&a, 2.71);
        println!("{a:6.3?}");
        let a = linspace_cpu(3.2, 4.7, 12);
        println!("{a:6.3?}");
        let a = linspace_cpu(Complex32::new(1.8, 7.5), Complex32::new(-8.9, 1.6), 12);
        println!("{a:6.3?}");
        let a = ones_cpu::<f64, _>([2, 2]);
        println!("{a:6.3?}");
        let a = a.ones_like();
        println!("{a:6.3?}");
        let a = zeros_cpu::<f64, _>([2, 2]);
        println!("{a:6.3?}");
        let a = a.zeros_like();
        println!("{a:6.3?}");
    }
}
