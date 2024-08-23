use crate::prelude_dev::*;

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

/// Convert tensor into owned tensor.
///
/// Data is either moved or cloned.
/// Layout is not involved; i.e. all underlying data is moved or cloned without
/// changing layout.
pub fn into_owned_keep_layout<R, D>(tensor: TensorBase<R, D>) -> TensorBase<DataOwned<R::Data>, D>
where
    R: DataAPI,
    R::Data: Clone,
    D: DimAPI,
{
    let TensorBase { data, layout } = tensor;
    let data = data.into_owned();
    unsafe { TensorBase::new_unchecked(data, layout) }
}

/// Methods for tensor ownership conversion.
impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{
    /// Get a view of tensor. See also [`view`].
    pub fn view(&self) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        R: DataAPI,
    {
        view(self)
    }

    /// Get a mutable view of tensor.
    pub fn view_mut(&mut self) -> TensorBase<DataRefMut<'_, R::Data>, D>
    where
        R: DataMutAPI,
    {
        view_mut(self)
    }

    /// Convert tensor into owned tensor.
    pub fn into_owned_keep_layout(self) -> TensorBase<DataOwned<R::Data>, D>
    where
        R: DataAPI,
        R::Data: Clone,
    {
        into_owned_keep_layout(self)
    }
}
