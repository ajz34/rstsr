use super::layout::*;
use super::*;
use crate::{Error, Result};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    /// Slice the tensor by a range, denoted by slice instead of
    /// std::ops::Range.
    Slice(SliceI),
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

impl From<Option<usize>> for Indexer {
    fn from(opt: Option<usize>) -> Self {
        match opt {
            Some(_) => panic!("Option<T> should not be used in Indexer."),
            None => Self::Insert,
        }
    }
}

macro_rules! impl_from_int_into_indexer {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Indexer {
                fn from(index: $t) -> Self {
                    Self::Select(index as isize)
                }
            }
        )*
    };
}

impl_from_int_into_indexer!(usize, isize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

pub trait IndexerPreserve: Sized {
    /// Narrowing tensor by slicing at a specific dimension. Number of dimension
    /// will not change after slicing.
    fn dim_narrow(&self, dim: usize, slice: SliceI) -> Result<Self>;
}

impl<D> IndexerPreserve for Layout<D>
where
    D: DimBaseAPI + DimLayoutAPI,
{
    fn dim_narrow(&self, dim: usize, slice: SliceI) -> Result<Self> {
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

        // fast return if slice is empty
        if slice == (Slice { start: None, stop: None, step: None }) {
            return Ok(self.clone());
        }

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

pub trait IndexerDynamic: IndexerPreserve {
    /// Select dimension at index. Number of dimension will decrease by 1.
    fn dim_select(&self, dim: usize, index: isize) -> Result<Layout<IxD>>;

    /// Insert dimension after, with shape 1. Number of dimension will increase
    /// by 1.
    fn dim_insert(&self, dim: usize) -> Result<Layout<IxD>>;

    /// Index tensor by a list of indexers.
    fn dim_slice(&self, indexers: &[Indexer]) -> Result<Layout<IxD>>;
}

impl<D> IndexerDynamic for Layout<D>
where
    D: DimLayoutAPI,
{
    fn dim_select(&self, dim: usize, index: isize) -> Result<Layout<IxD>> {
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
                // dimension to be selected
                if index < 0 {
                    let idx = d as isize + index;
                    if idx < 0 {
                        return Err(Error::IndexOutOfBound { index, bound: d as isize });
                    }
                    offset += s * idx;
                } else {
                    if index >= d as isize {
                        return Err(Error::IndexOutOfBound { index, bound: d as isize });
                    }
                    offset += s * index;
                }
            } else {
                // other dimensions
                shape_new.push(d);
                stride_new.push(s);
            }
        }

        let offset = offset as usize;
        return Ok(Layout::<IxD>::new(Shape(shape_new), Stride(stride_new), offset));
    }

    fn dim_insert(&self, dim: usize) -> Result<Layout<IxD>> {
        // dimension check
        if dim > self.ndim() {
            panic!("Index out of bound: index {}, shape {}", dim, self.ndim());
        }

        // get essential information
        let is_f_prefer = self.is_f_prefer();
        let mut shape = self.shape_ref().as_ref().to_vec();
        let mut stride = self.stride_ref().as_ref().to_vec();
        let offset = self.offset();

        if is_f_prefer {
            if dim == 0 {
                shape.insert(0, 1);
                stride.insert(0, 1);
            } else {
                shape.insert(dim, 1);
                stride.insert(dim, stride[dim - 1]);
            }
        } else {
            if dim == self.ndim() {
                shape.push(1);
                stride.push(1);
            } else {
                shape.insert(dim, 1);
                stride.insert(dim, stride[dim]);
            }
        }

        return Ok(Layout::<IxD>::new(Shape(shape), Stride(stride), offset));
    }

    fn dim_slice(&self, indexers: &[Indexer]) -> Result<Layout<IxD>> {
        // transform any layout to dynamic layout
        let shape = self.shape_ref().as_ref().to_vec();
        let stride = self.stride_ref().as_ref().to_vec();
        let mut layout =
            Layout { shape: Shape(shape), stride: Stride(stride), offset: self.offset() };

        // clone indexers to vec to make it changeable
        let mut indexers = indexers.to_vec();

        // counter for indexer
        let mut counter_slice = 0;
        let mut counter_select = 0;
        let mut idx_eclipse = None;
        for (n, indexer) in indexers.iter().enumerate() {
            match indexer {
                Indexer::Slice(_) => counter_slice += 1,
                Indexer::Select(_) => counter_select += 1,
                Indexer::Eclipse => match idx_eclipse {
                    Some(_) => {
                        return Err(Error::InvalidInteger {
                            value: n as isize,
                            msg: "Eclipse-type indexer could not exceed 1.".to_string(),
                        });
                    },
                    None => {
                        idx_eclipse = Some(n);
                    },
                },
                _ => {},
            }
        }

        // check if slice-type and select-type indexer exceed the number of dimensions
        if counter_slice + counter_select > self.ndim() {
            return Err(Error::InvalidInteger {
                value: (counter_slice + counter_select) as isize,
                msg:
                    "Slice-type and Select-type indexer could not exceed the number of dimensions."
                        .to_string(),
            });
        }

        // insert eclipse by slice(:) anyway, default append at last
        let n_eclipse = self.ndim() - counter_slice - counter_select;
        if idx_eclipse.is_some() && n_eclipse == 0 {
            indexers.remove(idx_eclipse.unwrap());
        } else {
            let idx_eclipse = idx_eclipse.unwrap_or(indexers.len());
            if n_eclipse > 0 {
                indexers[idx_eclipse] = SliceI { start: None, stop: None, step: None }.into();
                if n_eclipse > 1 {
                    for _ in 1..n_eclipse {
                        indexers.insert(
                            idx_eclipse,
                            SliceI { start: None, stop: None, step: None }.into(),
                        );
                    }
                }
            }
        }

        // handle indexers from last
        // it is possible to be zero-dim, minus after -= 1
        let mut cur_dim = self.ndim() as isize;
        for indexer in indexers.iter().rev() {
            match indexer {
                Indexer::Slice(slice) => {
                    cur_dim -= 1;
                    layout = layout.dim_narrow(cur_dim as usize, slice.clone())?;
                },
                Indexer::Select(index) => {
                    cur_dim -= 1;
                    layout = layout.dim_select(cur_dim as usize, *index)?;
                },
                Indexer::Insert => {
                    layout = layout.dim_insert(cur_dim as usize)?;
                },
                _ => {
                    return Err(Error::Msg("Invalid indexer found.".to_string()));
                },
            }
        }

        // this program should be designed that cur_dim is zero at the end
        if cur_dim != 0 {
            return Err(Error::Msg("Internal program error in indexer.".to_string()));
        }

        return Ok(layout);
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

#[macro_export]
macro_rules! s {
    // basic rule
    [$($slc:expr),*] => {
        &[$(($slc).into()),*]
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
        let l1 = l.dim_narrow(1, s).unwrap();
        println!("{:?}", l1);
        let l2 = l.dim_select(1, -2).unwrap();
        println!("{:?}", l2);
        let l3 = l.dim_insert(1).unwrap();
        println!("{:?}", l3);

        let l = Layout::<Ix3>::new(Shape([2, 3, 4]), Stride([100, 10, 1]), 0);
        let l3 = l.dim_insert(1).unwrap();
        println!("{:?}", l3);

        let l4 = l.dim_slice(s![Indexer::Eclipse, 1..3, None, 2]).unwrap();
        let l4 = l4.into_dim::<Ix3>().unwrap();
        println!("{:?}", l4);
    }
}
