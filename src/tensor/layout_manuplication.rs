//! This module handles view and ownership of tensor data.
//!
//! Functions defined in this module shall not explicitly copy any value.
//!
//! Some functions in Python array API will be implemented here:
//! - [x] expand_dims [`expand_dims`]
//! - [x] flip [`flip`]
//! - [ ] moveaxis
//! - [x] permute_dims [`transpose`], [`permute_dims`], [`swapaxes`]
//! - [x] squeeze [`squeeze`]

use crate::prelude_dev::*;
use core::num::TryFromIntError;

/// View of reshaped tensor.
///
/// This function will raise error when
/// - The number of elements in the reshaped tensor is not same as the original
///   tensor.
/// - The layout of the original tensor is not contiguous.
pub fn to_shape_assume_contig<R, D, D2>(
    tensor: &TensorBase<R, D>,
    shape: impl Into<Shape<D2>>,
) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
where
    R: DataAPI,
    R::Data: Clone,
    D: DimAPI,
    D2: DimAPI,
{
    into_shape_assume_contig(tensor.view(), shape)
}

/// View of reshaped tensor. See also [`to_shape_assume_contig`].
pub fn into_shape_assume_contig<R, D, D2>(
    tensor: TensorBase<R, D>,
    shape: impl Into<Shape<D2>>,
) -> Result<TensorBase<R, D2>>
where
    R: DataAPI,
    D: DimAPI,
    D2: DimAPI,
{
    let layout = tensor.layout();
    let is_c_contig = layout.is_c_contig();
    let is_f_contig = layout.is_f_contig();

    let shape: Shape<D2> = shape.into();
    rstsr_assert_eq!(layout.size(), shape.size(), InvalidLayout, "Number of elements not same.")?;

    let new_layout = match (is_c_contig, is_f_contig) {
        (true, true) => match Order::default() {
            Order::C => shape.new_c_contig(layout.offset),
            Order::F => shape.new_f_contig(layout.offset),
        },
        (true, false) => shape.new_c_contig(layout.offset),
        (false, true) => shape.new_f_contig(layout.offset),
        (false, false) => rstsr_raise!(InvalidLayout, "Assumes contiguous layout.")?,
    };
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, new_layout)) }
}

/// Convert layout to another dimension.
///
/// This is mostly used when converting static dimension to dynamic
/// dimension or vice versa.
pub fn to_dim<R, D, D2>(tensor: &TensorBase<R, D>) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
where
    R: DataAPI,
    R::Data: Clone,
    D: DimAPI,
    D2: DimAPI,
    Layout<D>: TryInto<Layout<D2>, Error = Error>,
{
    into_dim(tensor.view())
}

/// Convert layout to another dimension. See also [`to_dim`].
pub fn into_dim<R, D, D2>(tensor: TensorBase<R, D>) -> Result<TensorBase<R, D2>>
where
    R: DataAPI,
    D: DimAPI,
    D2: DimAPI,
    Layout<D>: TryInto<Layout<D2>, Error = Error>,
{
    let layout = tensor.layout().clone().into_dim::<D2>()?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
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
    into_expand_dims(tensor.view(), axis)
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

/// Permutes the axes (dimensions) of an array.
///
/// # See also
///
/// - [Python array API standard: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
pub fn transpose<'a, R, D, I>(
    tensor: &'a TensorBase<R, D>,
    axes: &[I],
) -> Result<TensorBase<DataRef<'a, R::Data>, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError> + Copy,
{
    into_transpose(tensor.view(), axes)
}

/// Permutes the axes (dimensions) of an array. See also [`transpose`].
pub fn into_transpose<R, D, I>(tensor: TensorBase<R, D>, axes: &[I]) -> Result<TensorBase<R, D>>
where
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError> + Copy,
{
    let axes = axes.iter().map(|&x| x.try_into().unwrap()).collect::<Vec<isize>>();
    let layout = tensor.layout().transpose(&axes)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

/// Permutes the axes (dimensions) of an array. See also [`transpose`].
pub fn permute_dims<'a, R, D, I>(
    tensor: &'a TensorBase<R, D>,
    axes: &[I],
) -> Result<TensorBase<DataRef<'a, R::Data>, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError> + Copy,
{
    into_permute_dims(tensor.view(), axes)
}

/// Permutes the axes (dimensions) of an array. See also [`transpose`].
pub fn into_permute_dims<R, D, I>(tensor: TensorBase<R, D>, axes: &[I]) -> Result<TensorBase<R, D>>
where
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError> + Copy,
{
    into_transpose(tensor, axes)
}

/// Reverse the order of elements in an array along the given axis.
pub fn reverse_axes<R, D>(tensor: &TensorBase<R, D>) -> TensorBase<DataRef<'_, R::Data>, D>
where
    R: DataAPI,
    R::Data: Clone,
    D: DimAPI,
{
    into_reverse_axes(tensor.view())
}

/// Reverse the order of elements in an array along the given axis. See also
/// [`reverse_axes`].
pub fn into_reverse_axes<R, D>(tensor: TensorBase<R, D>) -> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    let layout = tensor.layout().reverse_axes();
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
}

/// Interchange two axes of an array.
///
/// # See also
///
/// - [numpy `swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html)
pub fn swapaxes<R, D, I>(
    tensor: &TensorBase<R, D>,
    axis1: I,
    axis2: I,
) -> TensorBase<DataRef<'_, R::Data>, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    into_swapaxes(tensor.view(), axis1, axis2)
}

/// Interchange two axes of an array. See also [`swapaxes`].
pub fn into_swapaxes<R, D, I>(tensor: TensorBase<R, D>, axis1: I, axis2: I) -> TensorBase<R, D>
where
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis1 = axis1.try_into().unwrap();
    let axis2 = axis2.try_into().unwrap();
    let layout = tensor.layout().swapaxes(axis1, axis2).unwrap();
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
}

/// Removes singleton dimensions (axes).
///
/// # See also
///
/// - [Python array API standard: `squeeze`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.squeeze.html)
pub fn squeeze<R, D, I>(
    tensor: &TensorBase<R, D>,
    axis: I,
) -> Result<TensorBase<DataRef<'_, R::Data>, D::SmallerOne>>
where
    R: DataAPI,
    D: DimSmallerOneAPI,
    Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
    I: TryInto<isize, Error = TryFromIntError>,
{
    into_squeeze(tensor.view(), axis)
}

/// Removes singleton dimensions (axes). See also [`squeeze`].
pub fn into_squeeze<R, D, I>(
    tensor: TensorBase<R, D>,
    axis: I,
) -> Result<TensorBase<R, D::SmallerOne>>
where
    D: DimSmallerOneAPI,
    Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into().unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_eliminate(axis)?;
    let layout = layout.try_into().unwrap(); // safe to unwrap
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
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
    into_flip(tensor.view(), axis)
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

/// Methods for tensor manipulation.
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// View of reshaped tensor. See also [`to_shape_assume_contig`].
    pub fn to_shape_assume_contig<D2>(
        &self,
        shape: impl Into<Shape<D2>>,
    ) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        R::Data: Clone,
        D2: DimAPI,
    {
        to_shape_assume_contig(self, shape)
    }

    /// View of reshaped tensor. See also [`into_shape_assume_contig`].
    pub fn into_shape_assume_contig<D2>(
        self,
        shape: impl Into<Shape<D2>>,
    ) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self, shape)
    }

    /// Convert layout to another dimension. See also [`to_dim`].
    pub fn to_dim<D2>(&self) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        R::Data: Clone,
        D2: DimAPI,
        Layout<D>: TryInto<Layout<D2>, Error = Error>,
    {
        to_dim(self)
    }

    /// Convert layout to another dimension. See also [`to_dim`].
    pub fn into_dim<D2>(self) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
        Layout<D>: TryInto<Layout<D2>, Error = Error>,
    {
        into_dim(self)
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

    /// Permutes the axes (dimensions) of an array. See also [`transpose`].
    pub fn transpose<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose(self, axes)
    }

    /// Permutes the axes (dimensions) of an array. See also [`transpose`].
    pub fn into_transpose<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        into_transpose(self, axes)
    }

    /// Permutes the axes (dimensions) of an array. See also [`transpose`].
    pub fn permute_dims<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        permute_dims(self, axes)
    }

    /// Permutes the axes (dimensions) of an array. See also [`transpose`].
    pub fn into_permute_dims<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        into_permute_dims(self, axes)
    }

    /// Reverse the order of elements in an array along the given axis. See
    /// also [`reverse_axes`].
    pub fn reverse_axes(&self) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        R::Data: Clone,
    {
        reverse_axes(self)
    }

    /// Reverse the order of elements in an array along the given axis. See
    /// also [`reverse_axes`].
    pub fn into_reverse_axes(self) -> TensorBase<R, D>
    where
        R::Data: Clone,
    {
        into_reverse_axes(self)
    }

    /// Interchange two axes of an array. See also [`swapaxes`].
    pub fn swapaxes<I>(&self, axis1: I, axis2: I) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        swapaxes(self, axis1, axis2)
    }

    /// Interchange two axes of an array. See also [`swapaxes`].
    pub fn into_swapaxes<I>(self, axis1: I, axis2: I) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        into_swapaxes(self, axis1, axis2)
    }

    /// Removes singleton dimensions (axes). See also [`squeeze`].
    pub fn squeeze<I>(&self, axis: I) -> Result<TensorBase<DataRef<'_, R::Data>, D::SmallerOne>>
    where
        D: DimSmallerOneAPI,
        Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
        I: TryInto<isize, Error = TryFromIntError>,
    {
        squeeze(self, axis)
    }

    /// Removes singleton dimensions (axes). See also [`into_squeeze`].
    pub fn into_squeeze<I>(self, axis: I) -> Result<TensorBase<R, D::SmallerOne>>
    where
        D: DimSmallerOneAPI,
        Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
        I: TryInto<isize, Error = TryFromIntError>,
    {
        into_squeeze(self, axis)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::{cpu_backend::device::CpuDevice, storage::Storage};

    #[test]
    fn test_to_shape_assume_contig() {
        let a = linspace_cpu(2.5, 3.2, 16);
        let b = a.to_shape_assume_contig([4, 4]).unwrap();
        println!("{:.3?}", b);
    }

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
