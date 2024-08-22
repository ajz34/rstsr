//! This module handles view and ownership of tensor data.
//!
//! Functions defined in this module shall not explicitly copy any value.
//!
//! Some functions in Python array API will be implemented here:
//! - [x] expand_dims [`TensorBase::expand_dims`]
//! - [x] flip
//! - [ ] moveaxis
//! - [ ] permute_dims
//! - [ ] squeeze

use crate::prelude_dev::*;
use core::num::TryFromIntError;

/// Get a view of tensor.
pub fn view<R, D>(tensor: &TensorBase<R, D>) -> TensorBase<DataRef<'_, R::Data>, D>
where
    R: DataAPI,
    D: DimAPI,
{
    let data = tensor.data().as_ref();
    let layout = tensor.layout().clone();
    unsafe { TensorBase::new_unchecked(data, layout) }
}

/// Get a mutable view of tensor.
pub fn view_mut<R, D>(tensor: &mut TensorBase<R, D>) -> TensorBase<DataRefMut<'_, R::Data>, D>
where
    R: DataMutAPI,
    D: DimAPI,
{
    let layout = tensor.layout().clone();
    let data = tensor.data_mut().as_ref_mut();
    unsafe { TensorBase::new_unchecked(data, layout) }
}

/// Expand the shape of tensor.
///
/// Insert a new axis that will appear at the `axis` position in the
/// expanded tensor shape.
///
/// # Arguments
///
/// * `axis` - The position in the expanded axes where the new axis is placed.
///
/// # Example
///
/// ```
/// use rstsr::Tensor;
/// let a = Tensor::<f64, _>::zeros([4, 9, 8]);
/// let b = a.expand_dims(2);
/// assert_eq!(b.shape(), &[4, 9, 1, 8]);
/// ```
///
/// # Panics
///
/// - If `axis` is greater than the number of axes in the original tensor.
///
/// # See also
///
/// - [numpy.expand_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
/// - [Python array API standard: `expand_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.expand_dims.html)
pub fn expand_dims<R, D, I>(
    tensor: &TensorBase<R, D>,
    axis: I,
) -> TensorBase<DataRef<'_, R::Data>, D::LargerOne>
where
    R: DataAPI,
    D: DimLargerOneAPI,
    Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into().unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_insert(axis).unwrap();
    let layout = layout.try_into().unwrap(); // safe to unwrap
    let data = tensor.data().as_ref();
    unsafe { TensorBase::new_unchecked(data, layout) }
}

/// Expand the shape of tensor. See also [`expand_dims`].
pub fn into_expand_dims<R, D, I>(tensor: TensorBase<R, D>, axis: I) -> TensorBase<R, D::LargerOne>
where
    D: DimLargerOneAPI,
    Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into().unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_insert(axis).unwrap();
    let layout = layout.try_into().unwrap(); // safe to unwrap
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
}

/// Reverses the order of elements in an array along the given axis.
///
/// The shape of the array must be preserved.
///
/// # Arguments
///
/// * `axis` - The axis to flip on.
///
/// # Panics
///
/// - If `axis` is greater than the number of axes in the original tensor.
///
/// # See also
///
/// - [Python array API standard: `flip`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.flip.html)
pub fn flip<R, D, I>(tensor: &TensorBase<R, D>, axis: I) -> TensorBase<DataRef<'_, R::Data>, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into().unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_narrow(axis, slice!(None, None, -1)).unwrap();
    let data = tensor.data().as_ref();
    unsafe { TensorBase::new_unchecked(data, layout) }
}

/// Reverses the order of elements in an array along the given axis. See also
/// [`flip`].
pub fn into_flip<R, D, I>(tensor: TensorBase<R, D>, axis: I) -> TensorBase<R, D>
where
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into().unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_narrow(axis, slice!(None, None, -1)).unwrap();
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Get a view of tensor. See also [`view`].
    pub fn view(&self) -> TensorBase<DataRef<'_, R::Data>, D> {
        view(self)
    }

    /// Expand the shape of tensor. See also [`expand_dims`].
    pub fn expand_dims<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D::LargerOne>
    where
        D: DimLargerOneAPI,
        Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
        I: TryInto<isize, Error = TryFromIntError>,
    {
        expand_dims(self, axis)
    }

    /// Expand the shape of tensor. See also [`into_expand_dims`].
    pub fn into_expand_dims<I>(self, axis: I) -> TensorBase<R, D::LargerOne>
    where
        D: DimLargerOneAPI,
        Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
        I: TryInto<isize, Error = TryFromIntError>,
    {
        into_expand_dims(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis. See
    /// also [`flip`].
    pub fn flip<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        flip(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis. See
    /// also [`into_flip`].
    pub fn into_flip<I>(self, axis: I) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        into_flip(self, axis)
    }
}

impl<R, D> TensorBase<R, D>
where
    R: DataMutAPI,
    D: DimAPI,
{
    /// Get a mutable view of tensor.
    pub fn view_mut(&mut self) -> TensorBase<DataRefMut<'_, R::Data>, D> {
        view_mut(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::{cpu_backend::device::CpuDevice, storage::Storage};

    // fn test_expand_dims() {
    //     let a = Tensor::<f64, _>::zeros([4, 9, 8]);
    //     let b = a.expand_dims(2);
    //     assert_eq!(b.shape(), &[4, 9, 1, 8]);
    // }
    #[test]
    fn test_flip() {
        let device = CpuDevice {};
        let a = Tensor::<f64, _>::new(
            Storage::<f64, CpuDevice>::new(
                (0..24).map(|v| v as f64).collect::<Vec<_>>(),
                device.clone(),
            )
            .into(),
            [2, 3, 4].c(),
        )
        .unwrap();
        println!("{:?}", a);

        let b = a.flip(1);
        println!("{:?}", b);
        assert_eq!(b.shape(), &[2, 3, 4]);
        let c = a.flip(2);
        println!("{:?}", c);
        assert_eq!(c.shape(), &[2, 3, 4]);
    }
}
