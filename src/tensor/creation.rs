//! Creation methods for `Tensor` struct.
//!
//! This module relates to the [Python array API standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/creation_functions.html).
//!
//! Todo list:
//! - [x] `arange`: [`Tensor::arange`], [`Tensor::arange_int`]
//! - [ ] `asarray`
//! - [x] `empty`: [`Tensor::empty`]
//! - [x] `empty_like`: [`Tensor::empty_like`]
//! - [ ] `eye`
//! - [ ] ~`from_dlpack`~
//! - [x] `full`: [`Tensor::full`]
//! - [x] `full_like`: [`Tensor::full_like`]
//! - [x] `linspace`: [`Tensor::linspace`]
//! - [ ] `meshgrid`
//! - [x] `ones`: [`Tensor::ones`]
//! - [x] `ones_like`: [`Tensor::ones_like`]
//! - [ ] `tril`
//! - [ ] `triu`
//! - [x] `zeros`: [`Tensor::zeros`]
//! - [x] `zeros_like`: [`Tensor::zeros_like`]

use crate::cpu_backend::device::CpuDevice;
use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::{Float, Num};

/* #region arange */

impl<T, B> Tensor<T, Ix1, B>
where
    T: Float,
    B: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[start, stop)` as
    /// one-dimensional array.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html#array_api.arange)
    /// - [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
    pub fn arange(start: T, stop: T, step: T, device: &B) -> Tensor<T, Ix1, B> {
        let data = B::arange_impl(device, start, stop, step).unwrap();
        let layout = [data.len()].into();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T> Tensor<T, Ix1, CpuDevice>
where
    T: Float,
    CpuDevice: DeviceAPI<T> + DeviceCreationFloatAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[start, stop)` as
    /// one-dimensional array.
    ///
    /// # See also
    ///
    /// - [`Tensor::arange`]
    pub fn arange_cpu(start: T, stop: T, step: T) -> Tensor<T, Ix1, CpuDevice> {
        let data = CpuDevice::arange_impl(&CpuDevice {}, start, stop, step).unwrap();
        let layout = [data.len()].into();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

/* #endregion */

/* #region arange_int */

impl<T, B> Tensor<T, Ix1, B>
where
    T: Num,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[0, len)` as
    /// one-dimensional array, each step 1.
    pub fn arange_int(len: usize, device: &B) -> Tensor<T, Ix1, B> {
        let data = B::arange_int_impl(device, len).unwrap();
        let layout = [data.len()].into();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T> Tensor<T, Ix1, CpuDevice>
where
    T: Num,
    CpuDevice: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// Evenly spaced values within the half-open interval `[0, len)` as
    /// one-dimensional array, each step 1.
    pub fn arange_int_cpu(len: usize) -> Tensor<T, Ix1, CpuDevice> {
        Tensor::arange_int(len, &CpuDevice {})
    }
}

/* #endregion arange_int */

/* #region empty */

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Uninitialized tensor having a specified shape.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html)
    pub unsafe fn empty(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, B> = B::empty_impl(device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T, D> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug,
    D: DimAPI,
{
    /// Uninitialized tensor having a specified shape.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    ///
    /// # See also
    ///
    /// - [`Tensor::empty`]
    pub unsafe fn empty_cpu(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice> {
        Tensor::empty(layout, &CpuDevice {})
    }
}

/* #endregion */

/* #region empty_like */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Uninitialized tensor with the same shape as an input tensor.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor withuninitialized.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `empty_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty_like.html)
    pub unsafe fn empty_like(&self) -> Tensor<T, D, B> {
        let shape = self.layout().shape();
        let layout = shape.new_contig(0);
        let idx_max = layout.size();
        let device = self.data().storage().device();
        let data: Storage<T, _> = device.empty_impl(idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

/* #endregion */

/* #region full */

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// New tensor having a specified shape and filled with given value.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
    pub fn full(layout: impl Into<Layout<D>>, fill: T, device: &B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let idx_max = layout.size();
        let data: Storage<T, _> = device.full_impl(idx_max, fill).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T, D> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug,
    D: DimAPI,
{
    /// New tensor having a specified shape and filled with given value.
    ///
    /// # See also
    ///
    /// [`Tensor::full`]
    pub fn full_cpu(layout: impl Into<Layout<D>>, fill: T) -> Tensor<T, D, CpuDevice> {
        Tensor::full(layout, fill, &CpuDevice {})
    }
}

/* #endregion */

/* #region full_like */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// New tensor filled with given value and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
    pub fn full_like(&self, fill: T) -> Tensor<T, D, B> {
        let shape = self.layout().shape();
        let layout = shape.new_contig(0);
        let idx_max = layout.size();
        let device = self.data().storage().device();
        let data: Storage<T, _> = device.full_impl(idx_max, fill).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

/* #endregion */

/* #region linspace */

impl<T, B> Tensor<T, Ix1, B>
where
    T: ComplexFloat,
    B: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
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
    pub fn linspace(start: T, end: T, n: usize, device: &B) -> Tensor<T, Ix1, B> {
        let data = B::linspace_impl(device, start, end, n).unwrap();
        let layout = [data.len()].into();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T> Tensor<T, Ix1, CpuDevice>
where
    T: ComplexFloat,
    CpuDevice: DeviceAPI<T> + DeviceCreationComplexFloatAPI<T>,
{
    /// Evenly spaced numbers over a specified interval.
    ///
    /// # See also
    ///
    /// - [`Tensor::linspace`]
    pub fn linspace_cpu(start: T, end: T, n: usize) -> Tensor<T, Ix1, CpuDevice> {
        Tensor::linspace(start, end, n, &CpuDevice {})
    }
}

/* #endregion */

/* #region ones */

impl<T, D, B> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with ones and having a specified shape.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
    pub fn ones(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, _> = B::ones_impl(device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T, D> Tensor<T, D, CpuDevice>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    /// New tensor filled with ones and having a specified shape.
    ///
    /// # See also
    ///
    /// [`Tensor::ones`]
    pub fn ones_cpu(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice>
    where
        T: Num + Clone + Debug,
        D: DimAPI,
    {
        Tensor::ones(layout, &CpuDevice {})
    }
}

/* #endregion */

/* #region ones_like */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with ones and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
    pub fn ones_like(&self) -> Tensor<T, D, B> {
        let shape = self.layout().shape();
        let layout = shape.new_contig(0);
        let idx_max = layout.size();
        let device = self.data().storage().device();
        let data: Storage<T, B> = B::ones_impl(device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

/* #endregion */

/* #region zeros */

impl<T, D, B> Tensor<T, D, B>
where
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with zeros and having a specified shape.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
    pub fn zeros(layout: impl Into<Layout<D>>, device: &B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, _> = B::zeros_impl(device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

impl<T, D> Tensor<T, D, CpuDevice>
where
    T: Num + Clone + Debug,
    D: DimAPI,
{
    /// New tensor filled with zeros and having a specified shape.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
    pub fn zeros_cpu(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice> {
        Tensor::zeros(layout, &CpuDevice {})
    }
}

/* #endregion */

/* #region zeros_like */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Num,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationNumAPI<T>,
{
    /// New tensor filled with zeros and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
    pub fn zeros_like(&self) -> Tensor<T, D, B> {
        let shape = self.layout().shape();
        let layout = shape.new_contig(0);
        let idx_max = layout.size();
        let device = self.data().storage().device();
        let data: Storage<T, _> = B::zeros_impl(device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn playground() {
        let a = Tensor::arange_cpu(2.5, 3.2, 0.02);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::arange_int_cpu(15);
        println!("{a:6.3?}");
        let a = unsafe { Tensor::<f64, _>::empty_cpu([15, 18].f()) };
        println!("{a:6.3?}");
        let a = unsafe { a.empty_like() };
        println!("{a:6.3?}");
        let a = unsafe { Tensor::empty_like(&a) };
        println!("{a:6.3?}");
        let a = Tensor::full_cpu([2, 2].f(), 3.16);
        println!("{a:6.3?}");
        let a = Tensor::full_like(&a, 2.71);
        println!("{a:6.3?}");
        let a = a.full_like(2.71);
        println!("{a:6.3?}");
        let a = Tensor::linspace_cpu(3.2, 4.7, 12);
        println!("{a:6.3?}");
        let a = Tensor::linspace_cpu(Complex32::new(1.8, 7.5), Complex32::new(-8.9, 1.6), 12);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::ones_cpu([2, 2]);
        println!("{a:6.3?}");
        let a = a.ones_like();
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::zeros_cpu([2, 2]);
        println!("{a:6.3?}");
        let a = a.zeros_like();
        println!("{a:6.3?}");
    }
}
