//! Creation methods for `Tensor` struct.
//!
//! This module relates to the [Python array API standard v2023.12](https://data-apis.org/array-api/2023.12/API_specification/creation_functions.html).
//!
//! Todo list:
//! - [x] `arange`: [`Tensor::arange_with_device`],
//!   [`Tensor::arange_int_with_device`]
//! - [ ] `asarray`
//! - [x] `empty`: [`Tensor::empty_with_device`]
//! - [x] `empty_like`: [`Tensor::empty_like_with_device`]
//! - [ ] `eye`
//! - [ ] ~`from_dlpack`~
//! - [x] `full`: [`Tensor::full_with_device`]
//! - [x] `full_like`: [`Tensor::full_like_with_device`]
//! - [x] `linspace`: [`Tensor::linspace_with_device`]
//! - [ ] `meshgrid`
//! - [x] `ones`: [`Tensor::ones_with_device`]
//! - [x] `ones_like`: [`Tensor::ones_like_with_device`]
//! - [ ] `tril`
//! - [ ] `triu`
//! - [x] `zeros`: [`Tensor::zeros_with_device`]
//! - [x] `zeros_like`: [`Tensor::zeros_like_with_device`]

use crate::cpu_backend::device::CpuDevice;
use crate::layout::{DimAPI, Ix1, Layout};
use crate::storage::{DataAPI, Storage, StorageBaseAPI, StorageCreationAPI};
use crate::{Tensor, TensorBase};
use core::fmt::Debug;
use num::complex::ComplexFloat;
use num::{Float, Num};

/* #region with backend */

/// Tensor creation methods for one-dimension array.
impl<T, B> Tensor<T, Ix1, B>
where
    Storage<T, B>: StorageBaseAPI<DType = T, Device = B> + StorageCreationAPI,
{
    /// Evenly spaced values within the half-open interval `[start, stop)` as a
    /// one-dimensional array.
    ///
    /// # See also
    ///
    /// - [`Tensor::arange_int_with_device`] for simplified arange function,
    ///   where step is of size 1.
    /// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html#array_api.arange)
    /// - [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
    pub fn arange_with_device(start: T, stop: T, step: T, device: B) -> Tensor<T, Ix1, B>
    where
        T: Float,
    {
        let data =
            <Storage<T, B> as StorageCreationAPI>::arange_impl(&device, start, stop, step).unwrap();
        let layout = [data.len()].into();
        Tensor::new(data.into(), layout).unwrap()
    }

    /// Evenly spaced values within the half-open interval `[0, len)` as a
    /// one-dimensional array. Each step is of size 1.
    ///
    /// Note that `int` here means input parameters is integer, not the output
    /// type.
    ///
    /// # See also
    ///
    /// - [`Tensor::arange_with_device`] for floating-point numbers and arbitary
    ///   start, stop, step.
    /// - [Python array API standard: `arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.arange.html#array_api.arange)
    /// - [`numpy.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
    pub fn arange_int_with_device(len: usize, device: B) -> Tensor<T, Ix1, B>
    where
        T: Num,
    {
        let data: Storage<T, B> = StorageCreationAPI::arange_int_impl(&device, len).unwrap();
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
    pub fn linspace_with_device(start: T, end: T, n: usize, device: B) -> Tensor<T, Ix1, B>
    where
        T: ComplexFloat,
    {
        let data =
            <Storage<T, B> as StorageCreationAPI>::linspace_impl(&device, start, end, n).unwrap();
        let layout = [data.len()].into();
        Tensor::new(data.into(), layout).unwrap()
    }
}

/// Tensor creation methods for arbitary-dimension.
impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    Storage<T, B>: StorageBaseAPI<DType = T, Device = B> + StorageCreationAPI,
{
    /// Uninitialized tensor having a specified shape.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    ///
    /// # See also
    ///
    /// - [`Tensor::zeros_with_device`] for creating a tensor with zeros.
    /// - [Python array API standard: `empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html)
    pub unsafe fn empty_with_device(layout: impl Into<Layout<D>>, device: B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, B> = StorageCreationAPI::empty_impl(&device, idx_max).unwrap();
        Tensor::new(data.into(), layout).unwrap()
    }

    /// Uninitialized tensor with the same shape as an input tensor.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    ///
    /// # See also
    ///
    /// - [`Tensor::empty_with_device`]
    /// - [`Tensor::zeros_with_device`] for creating a tensor with zeros.
    /// - [Python array API standard: `empty_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty_like.html)
    pub unsafe fn empty_like_with_device(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, B>>, D>,
        device: B,
    ) -> Tensor<T, D, B> {
        let shape = tensor.layout().shape();
        let stride = shape.stride_contig();
        let layout = Layout::new(shape, stride, 0);
        let idx_max = layout.size();
        let data: Storage<T, B> = StorageCreationAPI::empty_impl(&device, idx_max).unwrap();
        Tensor::new_unchecked(data.into(), layout)
    }

    /// New tensor having a specified shape and filled with given value.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html)
    pub fn full_with_device(
        layout: impl Into<Layout<D>>,
        fill_value: T,
        device: B,
    ) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, B> =
            StorageCreationAPI::full_impl(&device, idx_max, fill_value).unwrap();
        Tensor::new(data.into(), layout).unwrap()
    }

    /// New tensor filled with given value and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [`Tensor::full_with_device`] for creating a tensor with the same shape
    ///   as an input tensor.
    /// - [Python array API standard: `full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html)
    pub fn full_like_with_device(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, B>>, D>,
        fill_value: T,
        device: B,
    ) -> Tensor<T, D, B> {
        let shape = tensor.layout().shape();
        let stride = shape.stride_contig();
        let layout = Layout::new(shape, stride, 0);
        let idx_max = layout.size();
        let data: Storage<T, B> =
            StorageCreationAPI::full_impl(&device, idx_max, fill_value).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }

    /// New tensor filled with ones and having a specified shape.
    ///
    /// # See also
    ///
    /// - [`Tensor::zeros_with_device`] for creating a tensor with zeros.
    /// - [Python array API standard: `ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html)
    pub fn ones_with_device(layout: impl Into<Layout<D>>, device: B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, B> = StorageCreationAPI::ones_impl(&device, idx_max).unwrap();
        Tensor::new(data.into(), layout).unwrap()
    }

    /// New tensor filled with ones and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [`Tensor::ones_with_device`] for creating a tensor with ones.
    /// - [Python array API standard: `ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html)
    pub fn ones_like_with_device(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, B>>, D>,
        device: B,
    ) -> Tensor<T, D, B> {
        let shape = tensor.layout().shape();
        let stride = shape.stride_contig();
        let layout = Layout::new(shape, stride, 0);
        let idx_max = layout.size();
        let data: Storage<T, B> = StorageCreationAPI::ones_impl(&device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }

    /// New tensor filled with zeros and having a specified shape.
    ///
    /// # See also
    ///
    /// - [`Tensor::empty_with_device`] for creating a tensor with
    ///   uninitialized.
    /// - [Python array API standard: `zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html)
    pub fn zeros_with_device(layout: impl Into<Layout<D>>, device: B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index().unwrap();
        let data: Storage<T, B> = StorageCreationAPI::zeros_impl(&device, idx_max).unwrap();
        Tensor::new(data.into(), layout).unwrap()
    }

    /// New tensor filled with zeros and having the same shape as an input
    /// tensor.
    ///
    /// # See also
    ///
    /// - [`Tensor::zeros_with_device`] for creating a tensor with zeros.
    /// - [Python array API standard: `zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html)
    pub fn zeros_like_with_device(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, B>>, D>,
        device: B,
    ) -> Tensor<T, D, B> {
        let shape = tensor.layout().shape();
        let stride = shape.stride_contig();
        let layout = Layout::new(shape, stride, 0);
        let idx_max = layout.size();
        let data: Storage<T, B> = StorageCreationAPI::zeros_impl(&device, idx_max).unwrap();
        unsafe { Tensor::new_unchecked(data.into(), layout) }
    }
}

/* #endregion */

/* #region cpu device only */

/// Tensor creation methods for one-dimension array with CPU device.
impl<T> Tensor<T, Ix1, CpuDevice>
where
    T: Clone + Debug + Num,
{
    /// Evenly spaced values within the half-open interval `[start, stop)` as a
    /// one-dimensional array.
    ///
    /// This function is CPU versioin of [`Tensor::arange_with_device`].
    pub fn arange(start: T, stop: T, step: T) -> Tensor<T, Ix1, CpuDevice>
    where
        T: Float,
    {
        Tensor::<T, Ix1, CpuDevice>::arange_with_device(start, stop, step, CpuDevice)
    }

    /// Evenly spaced values within the half-open interval `[0, len)` as a
    /// one-dimensional array. Each step is of size 1.
    ///
    /// This function is CPU versioin of [`Tensor::arange_int_with_device`].
    pub fn arange_int(len: usize) -> Tensor<T, Ix1, CpuDevice> {
        Tensor::<T, Ix1, CpuDevice>::arange_int_with_device(len, CpuDevice)
    }

    /// Evenly spaced numbers over a specified interval.
    ///
    /// This function is CPU versioin of [`Tensor::linspace_with_device`].
    pub fn linspace(start: T, end: T, n: usize) -> Tensor<T, Ix1, CpuDevice>
    where
        T: ComplexFloat,
    {
        Tensor::<T, Ix1, CpuDevice>::linspace_with_device(start, end, n, CpuDevice)
    }
}

/// Tensor creation methods for arbitary-dimension with CPU device.
impl<T, D> Tensor<T, D, CpuDevice>
where
    T: Clone + Debug + Num,
    D: DimAPI,
{
    /// Uninitialized tensor having a specified shape.
    ///
    /// This function is CPU versioin of [`Tensor::empty_with_device`].
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    pub unsafe fn empty(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::empty_with_device(layout, CpuDevice)
    }

    /// Uninitialized tensor with the same shape as an input tensor.
    ///
    /// This function is CPU versioin of [`Tensor::empty_like_with_device`].
    ///
    /// # Safety
    ///
    /// This function is unsafe because it creates a tensor with uninitialized.
    pub unsafe fn empty_like(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, CpuDevice>>, D>,
    ) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::empty_like_with_device(tensor, CpuDevice)
    }

    /// New tensor having a specified shape and filled with given value.
    ///
    /// This function is CPU versioin of [`Tensor::full_with_device`].
    pub fn full(layout: impl Into<Layout<D>>, fill_value: T) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::full_with_device(layout, fill_value, CpuDevice)
    }

    /// New tensor filled with given value and having the same shape as an input
    /// tensor.
    ///
    /// This function is CPU versioin of [`Tensor::full_like_with_device`].
    pub fn full_like(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, CpuDevice>>, D>,
        fill_value: T,
    ) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::full_like_with_device(tensor, fill_value, CpuDevice)
    }

    /// New tensor filled with ones and having a specified shape.
    ///
    /// This function is CPU versioin of [`Tensor::ones_with_device`].
    pub fn ones(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::ones_with_device(layout, CpuDevice)
    }

    /// New tensor filled with ones and having the same shape as an input
    /// tensor.
    ///
    /// This function is CPU versioin of [`Tensor::ones_like_with_device`].
    pub fn ones_like(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, CpuDevice>>, D>,
    ) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::ones_like_with_device(tensor, CpuDevice)
    }

    /// New tensor filled with zeros and having a specified shape.
    ///
    /// This function is CPU versioin of [`Tensor::zeros_with_device`].
    pub fn zeros(layout: impl Into<Layout<D>>) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::zeros_with_device(layout, CpuDevice)
    }

    /// New tensor filled with zeros and having the same shape as an input
    /// tensor.
    ///
    /// This function is CPU versioin of [`Tensor::zeros_like_with_device`].
    pub fn zeros_like(
        tensor: &TensorBase<impl DataAPI<Data = Storage<T, CpuDevice>>, D>,
    ) -> Tensor<T, D, CpuDevice> {
        Tensor::<T, D, CpuDevice>::zeros_like_with_device(tensor, CpuDevice)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;
    use crate::layout::*;
    use num::complex::Complex32;

    #[test]
    fn playground() {
        use crate::cpu_backend::device::CpuDevice;
        let a = Tensor::<f64, _>::arange(2.5, 3.2, 0.02);
        println!("{a:6.3?}");
        let a = Tensor::<Complex32, _>::arange_int(15);
        println!("{a:6.3?}");
        let a = unsafe { Tensor::<Complex32, _>::empty([15, 18].f()) };
        println!("{a:6.3?}");
        let a = unsafe { Tensor::empty_like(&a) };
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::full([2, 2].f(), 3.16);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::full_like(&a, 2.71);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::linspace(3.2, 4.7, 12);
        println!("{a:6.3?}");
        let a = Tensor::<Complex32, _>::linspace(
            Complex32::new(1.8, 7.5),
            Complex32::new(-8.9, 1.6),
            12,
        );
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::ones_with_device([2, 2], CpuDevice);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::ones_like_with_device(&a, CpuDevice);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::zeros_with_device([2, 2], CpuDevice);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _>::zeros_with_device([2, 2].f(), CpuDevice);
        println!("{a:6.3?}");
    }
}
