//! This module handles tensor data manipulation.
//!
//! Functions that shall not explicitly copy any value:
//!
//! Some functions in Python array API will be implemented here:
//! - [x] expand_dims [`Tensor::expand_dims`]
//! - [x] flip [`Tensor::flip`]
//! - [ ] moveaxis
//! - [x] permute_dims [`Tensor::transpose`], [`Tensor::permute_dims`],
//!   [`Tensor::swapaxes`]
//! - [x] squeeze [`Tensor::squeeze`]
//!
//! Functions that will/may explicitly copy value:
//!
//! - [x] reshape [`Tensor::reshape`], [`Tensor::to_shape`]

use crate::prelude_dev::*;
use core::num::TryFromIntError;

/// Methods for tensor manipulation (dimension expanded).
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
{
    /// Expand the shape of tensor.
    ///
    /// Insert a new axis that will appear at the `axis` position in the
    /// expanded tensor shape.
    ///
    /// # Arguments
    ///
    /// * `axis` - The position in the expanded axes where the new axis is
    ///   placed.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::Tensor;
    /// let a = Tensor::<f64, _>::zeros_cpu([4, 9, 8]);
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
    pub fn expand_dims<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D::LargerOne>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        self.view().into_expand_dims(axis)
    }

    /// Expand the shape of tensor.
    ///
    /// # See also
    ///
    /// [`Tensor::expand_dims`]
    pub fn into_expand_dims<I>(self, axis: I) -> TensorBase<R, D::LargerOne>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        let axis = axis.try_into().unwrap(); // almost safe to unwrap
        let layout = self.layout().dim_insert(axis).unwrap();
        let layout = layout.try_into().unwrap(); // safe to unwrap
        unsafe { TensorBase::new_unchecked(self.data, layout) }
    }
}

/// Methods for tensor manipulation (dimension preserved).
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Permutes the axes (dimensions) of an array.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
    pub fn transpose<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        self.view().into_transpose(axes)
    }

    /// Permutes the axes (dimensions) of an array.
    ///
    /// # See also
    ///
    /// [`Tensor::transpose`]
    pub fn into_transpose<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        let axes = axes.iter().map(|&x| x.try_into().unwrap()).collect::<Vec<isize>>();
        let layout = self.layout().transpose(&axes)?;
        unsafe { Ok(TensorBase::new_unchecked(self.data, layout)) }
    }

    /// Permutes the axes (dimensions) of an array.
    ///
    /// # See also
    ///
    /// [`Tensor::transpose`]
    pub fn permute_dims<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        self.view().into_transpose(axes)
    }

    /// Permutes the axes (dimensions) of an array.
    ///
    /// # See also
    ///
    /// [`Tensor::transpose`]
    pub fn into_permute_dims<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        self.into_transpose(axes)
    }

    /// Reverse the order of elements in an array along the given axis.
    pub fn reverse_axes(&self) -> TensorBase<DataRef<'_, R::Data>, D> {
        self.view().into_reverse_axes()
    }

    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`Tensor::reverse_axes`]
    pub fn into_reverse_axes(self) -> TensorBase<R, D>
    where
        R: DataAPI,
        D: DimAPI,
    {
        let layout = self.layout().reverse_axes();
        unsafe { TensorBase::new_unchecked(self.data, layout) }
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// - [numpy `swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html)
    pub fn swapaxes<I>(&self, axis1: I, axis2: I) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        self.view().into_swapaxes(axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// [`Tensor::swapaxes`].
    pub fn into_swapaxes<I>(self, axis1: I, axis2: I) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        let axis1 = axis1.try_into().unwrap();
        let axis2 = axis2.try_into().unwrap();
        let layout = self.layout().swapaxes(axis1, axis2).unwrap();
        unsafe { TensorBase::new_unchecked(self.data, layout) }
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
    pub fn flip<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        self.view().into_flip(axis)
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`Tensor::flip`]
    pub fn into_flip<I>(self, axis: I) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        let axis = axis.try_into().unwrap(); // almost safe to unwrap
        let layout = self.layout().dim_narrow(axis, slice!(None, None, -1)).unwrap();
        unsafe { TensorBase::new_unchecked(self.data, layout) }
    }
}

/// Methods for tensor manipulation (dimension shrinked).
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
{
    /// Removes singleton dimensions (axes).
    ///
    /// # See also
    ///
    /// - [Python array API standard: `squeeze`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.squeeze.html)
    pub fn squeeze<I>(&self, axis: I) -> Result<TensorBase<DataRef<'_, R::Data>, D::SmallerOne>>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        self.view().into_squeeze(axis)
    }

    /// Removes singleton dimensions (axes).
    ///
    /// # See also
    ///
    /// [`Tensor::squeeze`]
    pub fn into_squeeze<I>(self, axis: I) -> Result<TensorBase<R, D::SmallerOne>>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        let axis = axis.try_into().unwrap(); // almost safe to unwrap
        let layout = self.layout().dim_eliminate(axis)?;
        let layout = layout.try_into().unwrap(); // safe to unwrap
        unsafe { Ok(TensorBase::new_unchecked(self.data, layout)) }
    }
}

/// Methods for tensor shape change without data clone.
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// View of reshaped tensor.
    ///
    /// This function will raise error when
    /// - The number of elements in the reshaped tensor is not same as the
    ///   original tensor.
    /// - The layout of the original tensor is not contiguous.
    pub fn to_shape_assume_contig<D2>(
        &self,
        shape: D2,
    ) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
    {
        self.view().into_shape_assume_contig(shape)
    }

    /// View of reshaped tensor.
    ///
    /// # See also
    ///
    /// [`Tensor::to_shape_assume_contig`]
    pub fn into_shape_assume_contig<D2>(self, shape: impl Into<D2>) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
    {
        let layout = self.layout();
        let is_c_contig = layout.is_c_contig();
        let is_f_contig = layout.is_f_contig();

        let shape: D2 = shape.into();
        rstsr_assert_eq!(
            layout.size(),
            shape.shape_size(),
            InvalidLayout,
            "Number of elements not same."
        )?;

        let new_layout = match (is_c_contig, is_f_contig) {
            (true, true) => match TensorOrder::default() {
                TensorOrder::C => shape.new_c_contig(Some(layout.offset)),
                TensorOrder::F => shape.new_f_contig(Some(layout.offset)),
            },
            (true, false) => shape.new_c_contig(Some(layout.offset)),
            (false, true) => shape.new_f_contig(Some(layout.offset)),
            (false, false) => rstsr_raise!(InvalidLayout, "Assumes contiguous layout.")?,
        };
        unsafe { Ok(TensorBase::new_unchecked(self.data, new_layout)) }
    }

    /// Convert layout to another dimension.
    ///
    /// This is mostly used when converting static dimension to dynamic
    /// dimension or vice versa.
    pub fn to_dim<D2>(&self) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        Layout<D>: TryInto<Layout<D2>, Error = Error>,
    {
        self.view().into_dim()
    }

    /// Convert layout to another dimension.
    ///
    /// # See also
    ///
    /// [`Tensor::to_dim`]
    pub fn into_dim<D2>(self) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
        Layout<D>: TryInto<Layout<D2>, Error = Error>,
    {
        let layout = self.layout().clone().into_dim::<D2>()?;
        unsafe { Ok(TensorBase::new_unchecked(self.data, layout)) }
    }
}

/// Methods for tensor shape change with possible data clone.
impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Reshapes an array.
    ///
    /// Values of data may not be changed, but explicit copy of data may occur
    /// depending on whether layout is c/f-contiguous.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `reshape`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.reshape.html)
    pub fn reshape<D2>(&self, shape: D2) -> TensorBase<DataCow<'_, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignAPI<T, D2, D>,
    {
        self.to_shape(shape)
    }

    /// Reshapes an array.
    ///
    /// # See also
    ///
    /// [`Tensor::reshape`]
    pub fn to_shape<D2>(&self, shape: D2) -> TensorBase<DataCow<'_, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignAPI<T, D2, D>,
    {
        rstsr_assert_eq!(self.size(), shape.shape_size(), InvalidLayout).unwrap();
        let result = self.to_shape_assume_contig(shape.clone());
        if let Ok(result) = result {
            // contiguous, no data cloned
            let layout = result.layout().clone();
            let data = result.data.into();
            return TensorBase { data, layout };
        } else {
            // non-contiguous, clone data if necessary
            let device = self.data.storage().device();
            let layout_new = shape.new_contig(None);
            let mut storage_new = unsafe { device.empty_impl(layout_new.size()).unwrap() };
            device
                .assign_arbitary_layout(
                    &mut storage_new,
                    &layout_new,
                    self.storage(),
                    self.layout(),
                )
                .unwrap();
            let data_new = DataCow::Owned(storage_new.into());
            TensorBase { data: data_new, layout: layout_new }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::{cpu_backend::device::CpuDevice, storage::Storage};

    #[test]
    fn test_to_shape_assume_contig() {
        let a = Tensor::linspace_cpu(2.5, 3.2, 16);
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

    #[test]
    fn test_to_shape() {
        let a = Tensor::<f64, _>::linspace_cpu(0.0, 15.0, 16);
        let mut a = a.to_shape([4, 4]);
        a.layout = Layout::new([2, 2], [2, 4], 0);
        println!("{:?}", a);
        let b = a.to_shape([2, 2]);
        println!("{:?}", b);
    }
}
