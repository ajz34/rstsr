use crate::prelude_dev::*;

/// Methods for tensor ownership conversion.
impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{
    /// Get a view of tensor.
    pub fn view(&self) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        R: DataAPI,
    {
        let data = self.data().as_ref();
        let layout = self.layout().clone();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }

    /// Get a mutable view of tensor.
    pub fn view_mut(&mut self) -> TensorBase<DataRefMut<'_, R::Data>, D>
    where
        R: DataMutAPI,
    {
        let layout = self.layout().clone();
        let data = self.data_mut().as_ref_mut();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }

    /// Convert tensor into owned tensor.
    ///
    /// Data is either moved or cloned.
    /// Layout is not involved; i.e. all underlying data is moved or cloned
    /// without changing layout.
    pub fn into_owned_keep_layout(self) -> TensorBase<DataOwned<R::Data>, D>
    where
        R: DataAPI,
    {
        let TensorBase { data, layout } = self;
        let data = data.into_owned();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

/* #region DataCow */

impl<S, D> From<TensorBase<DataOwned<S>, D>> for TensorBase<DataCow<'_, S>, D>
where
    D: DimAPI,
{
    #[inline]
    fn from(tensor: TensorBase<DataOwned<S>, D>) -> Self {
        let TensorBase { data, layout } = tensor;
        let data = DataCow::from(data);
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

impl<'a, S, D> From<TensorBase<DataRef<'a, S>, D>> for TensorBase<DataCow<'a, S>, D>
where
    D: DimAPI,
{
    #[inline]
    fn from(tensor: TensorBase<DataRef<'a, S>, D>) -> Self {
        let TensorBase { data, layout } = tensor;
        let data = DataCow::from(data);
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

/* #endregion */
