use crate::prelude_dev::*;

/* #region iterator definitions */

pub struct TensorLayoutIteratorRef<'t, 'a, R, D, It>
where
    D: DimAPI,
    It: LayoutIterAPI<Dim = D>,
{
    pub(crate) tensor: &'t TensorBase<R, D>,
    pub(crate) layout_iterator: It,
    _phantom: PhantomData<&'a R>,
}

pub struct TensorLayoutIteratorMut<'t, 'a, R, D, It>
where
    R: DataMutAPI,
    D: DimAPI,
    It: LayoutIterAPI<Dim = D>,
{
    pub(crate) tensor: &'t mut TensorBase<R, D>,
    pub(crate) layout_iterator: It,
    _phantom: PhantomData<&'a mut R>,
}

/* #endregion */

/* #region impl TensorLayoutIteratorRef */

impl<'t, R, D, It> TensorLayoutIteratorRef<'t, '_, R, D, It>
where
    D: DimAPI,
    It: LayoutIterAPI<Dim = D>,
{
    pub fn new(tensor: &'t TensorBase<R, D>) -> Self {
        let layout_iterator = It::new(tensor.layout());
        Self { tensor, layout_iterator, _phantom: PhantomData }
    }
}

impl<'t, 'a, R, T, D, B, It> Iterator for TensorLayoutIteratorRef<'t, 'a, R, D, It>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceStorageAPI<T>,
    It: LayoutIterAPI<Dim = D>,
    T: 'a,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.layout_iterator.next()?;
        let ptr = self.tensor.data().as_storage().get_index_ptr(index);
        unsafe { Some(&*ptr) }
    }
}

impl<R, D, It> ExactSizeIterator for TensorLayoutIteratorRef<'_, '_, R, D, It>
where
    D: DimAPI,
    It: LayoutIterAPI<Dim = D>,
    Self: Iterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.tensor.size()
    }
}

/* #endregion */

/* #region impl TensorLayoutIteratorMut */

impl<'t, R, D, It> TensorLayoutIteratorMut<'t, '_, R, D, It>
where
    R: DataMutAPI,
    D: DimAPI,
    It: LayoutIterAPI<Dim = D>,
{
    pub fn new(tensor: &'t mut TensorBase<R, D>) -> Self {
        let layout_iterator = It::new(tensor.layout());
        Self { tensor, layout_iterator, _phantom: PhantomData }
    }
}

impl<'t, 'a, R, T, D, B, It> Iterator for TensorLayoutIteratorMut<'t, 'a, R, D, It>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceStorageAPI<T>,
    It: LayoutIterAPI<Dim = D>,
    T: 'a,
{
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.layout_iterator.next()?;
        let ptr = self.tensor.data_mut().as_storage_mut().get_index_mut_ptr(index);
        unsafe { Some(&mut *ptr) }
    }
}

impl<R, D, It> ExactSizeIterator for TensorLayoutIteratorMut<'_, '_, R, D, It>
where
    R: DataMutAPI,
    D: DimAPI,
    It: LayoutIterAPI<Dim = D>,
    Self: Iterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.tensor.size()
    }
}

/* #endregion */

/* #region tensor iter object */

pub fn iter_ref_c_prefer<R, D>(
    tensor: &TensorBase<R, D>,
) -> TensorLayoutIteratorRef<'_, '_, R, D, IterLayoutRowMajor<D>>
where
    D: DimAPI,
    IterLayoutRowMajor<D>: LayoutIterAPI<Dim = D>,
{
    TensorLayoutIteratorRef::new(tensor)
}

pub fn iter_mut_c_prefer<R, D>(
    tensor: &mut TensorBase<R, D>,
) -> TensorLayoutIteratorMut<'_, '_, R, D, IterLayoutRowMajor<D>>
where
    R: DataMutAPI,
    D: DimAPI,
    IterLayoutRowMajor<D>: LayoutIterAPI<Dim = D>,
{
    TensorLayoutIteratorMut::new(tensor)
}

pub fn iter_ref_f_prefer<R, D>(
    tensor: &TensorBase<R, D>,
) -> TensorLayoutIteratorRef<'_, '_, R, D, IterLayoutColMajor<D>>
where
    D: DimAPI,
    IterLayoutColMajor<D>: LayoutIterAPI<Dim = D>,
{
    TensorLayoutIteratorRef::new(tensor)
}

pub fn iter_mut_f_prefer<R, D>(
    tensor: &mut TensorBase<R, D>,
) -> TensorLayoutIteratorMut<'_, '_, R, D, IterLayoutColMajor<D>>
where
    R: DataMutAPI,
    D: DimAPI,
    IterLayoutColMajor<D>: LayoutIterAPI<Dim = D>,
{
    TensorLayoutIteratorMut::new(tensor)
}

/// Methods for element-wise iteration.
impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{

    pub fn iter_ref_c_prefer(&self) -> TensorLayoutIteratorRef<'_, '_, R, D, IterLayoutRowMajor<D>>
    where
        IterLayoutRowMajor<D>: LayoutIterAPI<Dim = D>,
    {
        iter_ref_c_prefer(self)
    }

    pub fn iter_mut_c_prefer(
        &mut self,
    ) -> TensorLayoutIteratorMut<'_, '_, R, D, IterLayoutRowMajor<D>>
    where
        IterLayoutRowMajor<D>: LayoutIterAPI<Dim = D>,
        R: DataMutAPI,
    {
        iter_mut_c_prefer(self)
    }

    pub fn iter_ref_f_prefer(&self) -> TensorLayoutIteratorRef<'_, '_, R, D, IterLayoutColMajor<D>>
    where
        IterLayoutColMajor<D>: LayoutIterAPI<Dim = D>,
    {
        iter_ref_f_prefer(self)
    }

    pub fn iter_mut_f_prefer(
        &mut self,
    ) -> TensorLayoutIteratorMut<'_, '_, R, D, IterLayoutColMajor<D>>
    where
        IterLayoutColMajor<D>: LayoutIterAPI<Dim = D>,
        R: DataMutAPI,
    {
        iter_mut_f_prefer(self)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_iter_ref_c_prefer() {
        let tensor = arange_cpu(2.5, 3.2, 0.1);

        let mut iter = iter_ref_c_prefer(&tensor);
        assert_eq!(iter.len(), 7);
        assert_eq!(iter.next(), Some(&2.5));
        assert_eq!(iter.next(), Some(&2.6));
        assert_eq!(iter.next(), Some(&2.7));
        println!("{:?}", iter.collect::<Vec<_>>());

        let tensor = linspace_cpu(0.0, 15.0, 16);
        let tensor = tensor.to_shape_assume_contig([4, 4]).unwrap();
        let mut iter = iter_ref_c_prefer(&tensor);
        assert_eq!(iter.len(), 16);
        assert_eq!(iter.next(), Some(&0.0));
        assert_eq!(iter.next(), Some(&1.0));
        assert_eq!(iter.next(), Some(&2.0));
        println!("{:?}", iter.collect::<Vec<_>>());

        let tensor = linspace_cpu(0.0, 15.0, 16);
        let tensor = tensor.to_shape_assume_contig([4, 4]).unwrap();
        let tensor = tensor.transpose(&[1, 0]).unwrap();
        let mut iter = iter_ref_c_prefer(&tensor);
        assert_eq!(iter.len(), 16);
        assert_eq!(iter.next(), Some(&0.0));
        assert_eq!(iter.next(), Some(&4.0));
        assert_eq!(iter.next(), Some(&8.0));
        println!("{:?}", iter.collect::<Vec<_>>());
    }
}
