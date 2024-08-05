use super::layout::*;
use super::*;
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    /// Slice the tensor by a range, denoted by slice instead of
    /// std::ops::Range.
    Slice(SliceI),
    /// Indexing via a 1d tensor. Currently not applied.
    IndexSelect(Vec<isize>),
    /// Marginalize one dimension out by index.
    Select(isize),
    /// Insert dimension at index, something like unsqueeze. Currently not
    /// applied.
    Insert,
    /// Expand dimensions. Currently not applied.
    Eclipse,
}

impl<R> From<R> for Indexer
where
    R: Into<SliceI>,
{
    fn from(slice: R) -> Self {
        Self::Slice(slice.into())
    }
}

impl From<usize> for Indexer {
    fn from(index: usize) -> Self {
        Self::Select(index as isize)
    }
}

impl From<isize> for Indexer {
    fn from(index: isize) -> Self {
        Self::Select(index)
    }
}

impl From<Vec<usize>> for Indexer {
    fn from(index: Vec<usize>) -> Self {
        Self::IndexSelect(index.iter().map(|&v| v as isize).collect())
    }
}

impl From<Vec<isize>> for Indexer {
    fn from(index: Vec<isize>) -> Self {
        Self::IndexSelect(index)
    }
}

impl From<Option<usize>> for Indexer {
    fn from(opt: Option<usize>) -> Self {
        match opt {
            Some(v) => Self::Select(v as isize),
            None => Self::Insert,
        }
    }
}

impl From<Option<isize>> for Indexer {
    fn from(opt: Option<isize>) -> Self {
        match opt {
            Some(v) => Self::Select(v),
            None => Self::Insert,
        }
    }
}

pub trait IndexerPreserve: Sized {
    /// Narrowing tensor by slicing at a specific dimension. Number of dimension
    /// will not change after slicing.
    fn slice_at_dim(&self, dim: usize, slice: SliceI) -> Result<Self>;
}

impl<D> IndexerPreserve for Layout<D>
where
    D: DimBaseAPI + DimLayoutAPI,
{
    fn slice_at_dim(&self, dim: usize, slice: SliceI) -> Result<Self> {
        // dimension check
        if dim >= self.ndim() {
            return Err(Error::ValueOutOfRange {
                value: dim as isize,
                min: 0,
                max: self.ndim() as isize - 1,
            });
        }

        // get essential information
        let mut shape = self.shape();
        let mut stride = self.stride();
        let shape_mut = shape.as_mut();
        let stride_mut = stride.as_mut();

        // previous shape length
        let len_prev = shape_mut[dim] as isize;

        // handle cases of step > 0 and step < 0
        let step = slice.step().unwrap_or(1);
        if step == 0 {
            return Err(Error::InvalidInteger { value: step, msg: "step cannot be 0".to_string() });
        }

        if step > 0 {
            // default start = 0 and stop = len_prev
            let mut start = slice.start().unwrap_or(0);
            let mut stop = slice.stop().unwrap_or(len_prev);

            // handle negative slice
            if start < 0 {
                start = (len_prev + start).max(0);
            }
            if stop < 0 {
                stop = (len_prev + stop).max(0);
            }

            if start > len_prev || start > stop {
                // zero size slice caused by inproper start and stop
                start = 0;
                stop = 0;
            } else if stop > len_prev {
                // stop is out of bound, set it to len_prev
                stop = len_prev;
            }

            let offset = (self.offset() as isize + stride_mut[dim] * start) as usize;
            shape_mut[dim] = ((stop - start + step - 1) / step).max(0) as usize;
            stride_mut[dim] = stride_mut[dim] * step;
            return Ok(Self::new(shape, stride, offset));
        } else {
            // step < 0
            // default start = len_prev and stop = 0
            let mut start = slice.start().unwrap_or(len_prev);
            let mut stop = slice.stop().unwrap_or(0);

            // handle negative slice
            if start < 0 {
                start = (len_prev + start).max(0);
            }
            if stop < 0 {
                stop = (len_prev + stop).max(0);
            }

            if stop > len_prev || stop > start {
                // zero size slice caused by inproper start and stop
                start = 0;
                stop = 0;
            } else if start > len_prev {
                // stop is out of bound, set it to len_prev
                start = len_prev;
            }

            let offset = (self.offset() as isize + len_prev * start) as usize;
            shape_mut[dim] = ((stop - start - step - 1) / step).max(0) as usize;
            stride_mut[dim] = stride_mut[dim] * step;
            return Ok(Self::new(shape, stride, offset));
        }
    }
}

pub trait IndexerDynamic {
    /// Select dimension at index. Number of dimension will decrease by 1.
    fn select_at_dim(&self, dim: usize, index: usize) -> Layout<IxD>;
}

impl<D> IndexerDynamic for Layout<D>
where
    D: DimLayoutAPI,
{
    /// Select dimension at index. Number of dimension will decrease by 1.
    fn select_at_dim(&self, dim: usize, index: usize) -> Layout<IxD> {
        // dimension check
        if dim >= self.ndim() {
            panic!("Index out of bound: index {}, shape {}", dim, self.ndim());
        }

        // get essential information
        let Shape(shape) = self.shape_ref();
        let Stride(stride) = self.stride_ref();
        let mut offset = self.offset() as isize;
        let mut shape_new: Vec<usize> = vec![];
        let mut stride_new: Vec<isize> = vec![];

        // change everything
        for (i, (&d, &s)) in shape.as_ref().iter().zip(stride.as_ref().iter()).enumerate() {
            if i == dim {
                offset += s * index as isize;
            } else {
                shape_new.push(d);
                stride_new.push(s);
            }
        }

        let offset = offset as usize;
        return Layout::<IxD>::new(Shape(shape_new), Stride(stride_new), offset);
    }
}

/// Generate slice with into support and optional parameters.
#[macro_export]
macro_rules! slice {
    ($stop:expr) => {
        Slice::<isize>::from(Slice { start: None, stop: $stop.into(), step: None })
    };
    ($start:expr, $stop:expr) => {
        Slice::<isize>::from(Slice { start: $start.into(), stop: $stop.into(), step: None })
    };
    ($start:expr, $stop:expr, $step:expr) => {
        Slice::<isize>::from(Slice { start: $start.into(), stop: $stop.into(), step: $step.into() })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice() {
        let t = 3 as usize;
        let s = slice!(1, 2, t);
        assert_eq!(s.start(), Some(1));
        assert_eq!(s.stop(), Some(2));
        assert_eq!(s.step(), Some(3));
    }

    #[test]
    fn test_slice_at_dim() {
        let l = Layout::<Ix3>::new(Shape([2, 3, 4]), Stride([1, 10, 100]), 0);
        let s = slice!(10, 1, -1);
        let l = l.slice_at_dim(1, s).unwrap();
        // let l = [1, 0, 2].c();
        // let l = l.slice_by_slices(&[(1..2).into(), (2..10).into(), (3..4).into()]);
        println!("{:?}", l);
    }
}
