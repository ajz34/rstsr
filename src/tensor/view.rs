//! This module handles view and ownership of tensor data.
//!
//! Functions defined in this module shall not explicitly copy any value.
//!
//! Some functions in Python array API will be implemented here:
//! - [x] expand_dims [`TensorBase::expand_dims`]
//! - [ ] flip
//! - [ ] moveaxis
//! - [ ] permute_dims
//! - [ ] squeeze

use crate::layout::{DimAPI, DimLargerOneAPI, DimSmallerOneAPI, IndexerDynamic, IxD, Layout};
use crate::storage::{DataAPI, DataRef, StorageBaseAPI};
use crate::{Error, TensorBase};
use core::fmt::Debug;
use std::num::TryFromIntError;

impl<S, D> TensorBase<S, D>
where
    D: DimAPI,
    S: DataAPI,
    S::Data: StorageBaseAPI + Debug + Clone,
{
    /// Get a view of tensor.
    pub fn view(&self) -> TensorBase<DataRef<'_, S::Data>, D> {
        let data = self.data().as_ref();
        let layout = self.layout().clone();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }

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
    /// # Returns
    ///
    /// A view of tensor with more axes than the original tensor.
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
    pub fn expand_dims<I>(&self, axis: I) -> TensorBase<DataRef<'_, S::Data>, D::LargerOne>
    where
        D: DimLargerOneAPI,
        Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
        I: TryInto<isize, Error = TryFromIntError>,
    {
        let axis = axis.try_into().unwrap(); // almost safe to unwrap
        let layout = self.layout().dim_insert(axis).unwrap();
        let layout = layout.try_into().unwrap(); // safe to unwrap
        let data = self.data().as_ref();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}
