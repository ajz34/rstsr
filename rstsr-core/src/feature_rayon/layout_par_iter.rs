//! Layout parallel iterator

use crate::prelude_dev::*;
use rayon::{
    iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    prelude::*,
};

/* #region layout iterator (col-major) */

pub struct ParIterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    pub layout_iter: IterLayoutColMajor<D>,
}

impl<D> IntoParallelIterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;
    type Iter = ParIterLayoutColMajor<D>;

    fn into_par_iter(self) -> Self::Iter {
        ParIterLayoutColMajor::<D> { layout_iter: self }
    }
}

impl<D> Producer for ParIterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;
    type IntoIter = IterLayoutColMajor<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.layout_iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs, rhs) = self.layout_iter.split_at(index).unwrap();
        let lhs = ParIterLayoutColMajor::<D> { layout_iter: lhs };
        let rhs = ParIterLayoutColMajor::<D> { layout_iter: rhs };
        return (lhs, rhs);
    }
}

impl<D> ParallelIterator for ParIterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.layout_iter.len())
    }
}

impl<D> IndexedParallelIterator for ParIterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

/* #endregion */

/* #region layout iterator (row-major) */

pub struct ParIterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    pub layout_iter: IterLayoutRowMajor<D>,
}

impl<D> IntoParallelIterator for IterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;
    type Iter = ParIterLayoutRowMajor<D>;

    fn into_par_iter(self) -> Self::Iter {
        ParIterLayoutRowMajor::<D> { layout_iter: self }
    }
}

impl<D> Producer for ParIterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;
    type IntoIter = IterLayoutRowMajor<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.layout_iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (lhs, rhs) = self.layout_iter.split_at(index).unwrap();
        let lhs = ParIterLayoutRowMajor::<D> { layout_iter: lhs };
        let rhs = ParIterLayoutRowMajor::<D> { layout_iter: rhs };
        return (lhs, rhs);
    }
}

impl<D> ParallelIterator for ParIterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.layout_iter.len())
    }
}

impl<D> IndexedParallelIterator for ParIterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_col_major() {
        let layout = [10, 10, 10].c();
        let iter_ser = IterLayoutColMajor::new(&layout).unwrap();
        let iter_par = IterLayoutColMajor::new(&layout).unwrap().into_par_iter();
        let vec_ser: Vec<usize> = iter_ser.collect();
        let mut vec_par = vec![];
        iter_par.collect_into_vec(&mut vec_par);
        assert_eq!(vec_ser, vec_par);
    }

    #[test]
    fn test_row_major() {
        let layout = [10, 10, 10].c();
        let iter_ser = IterLayoutRowMajor::new(&layout).unwrap();
        let iter_par = IterLayoutRowMajor::new(&layout).unwrap().into_par_iter();
        let vec_ser: Vec<usize> = iter_ser.collect();
        let mut vec_par = vec![];
        iter_par.collect_into_vec(&mut vec_par);
        assert_eq!(vec_ser, vec_par);
    }
}
