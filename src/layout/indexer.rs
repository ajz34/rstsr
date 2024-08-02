use super::layout::*;
use super::*;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub enum Indexer {
    /// Slice the tensor by a range, denoted by slice instead of
    /// std::ops::Range.
    Slice(SliceI),
    /// Indexing via a 1d tensor. Currently not applied.
    IndexSelect(Vec<usize>),
    /// Marginalize one dimension out by index.
    Select(usize),
    /// Insert dimension at index, something like unsqueeze. Currently not
    /// applied.
    Insert,
    /// Leave other dimensions to be the same. Currently not applied.
    Eclipse,
}

pub trait IndexerPreserve: Sized {
    /// Narrowing tensor by slicing at a specific dimension. Number of dimension
    /// will not change after slicing.
    fn slice_at_dim(&self, dim: usize, slice: SliceI) -> Result<Self>;

    /// Narrowing tensor by slicing at multiple dimensions. Number of dimension
    /// will not change after slicing. Number of slices should be the same to
    /// the number of dimensions.
    fn slice_by_slices(&self, slices: &[SliceI]) -> Result<Self>;
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
        let Shape(mut shape) = self.shape();
        let Stride(mut stride) = self.stride();
        let shape_mut = shape.as_mut();
        let stride_mut = stride.as_mut();

        let mut start = slice.start().unwrap_or(0);
        let mut stop = slice.stop().unwrap_or(shape_mut[dim] as isize);
        let step = slice.step().unwrap_or(1);

        if step == 0 {
            return Err(Error::InvalidInteger { value: step, msg: "step cannot be 0".to_string() });
        }

        // handle negative slice
        if start < 0 {
            start = (shape_mut[dim] as isize + start).max(0);
        }
        if stop < 0 {
            stop = (shape_mut[dim] as isize + stop).max(0);
        }
        // change shape and stride
        let offset = (self.offset() as isize + shape_mut[dim] as isize * start) as usize;
        shape_mut[dim] = ((stop - start + step - 1) / step).max(0) as usize;
        stride_mut[dim] = stride_mut[dim] * step;
        return Ok(Self::new(Shape(shape), Stride(stride), offset));
    }

    fn slice_by_slices(&self, slices: &[SliceI]) -> Result<Self> {
        // dimension check
        if slices.len() != self.ndim() {
            return Err(Error::ValueOutOfRange {
                value: slices.len() as isize,
                min: 0,
                max: self.ndim() as isize,
            });
        }

        // get essential information
        let mut offset = self.offset() as isize;
        let Shape(mut shape) = self.shape();
        let Stride(mut stride) = self.stride();
        let shape_mut = shape.as_mut();
        let stride_mut = stride.as_mut();

        for (dim, slice) in slices.iter().enumerate() {
            let mut start = slice.start().unwrap_or(0);
            let mut stop = slice.stop().unwrap_or(shape_mut[dim] as isize);
            let step = slice.step().unwrap_or(1);

            if step == 0 {
                return Err(Error::InvalidInteger {
                    value: step,
                    msg: "step cannot be 0".to_string(),
                });
            }

            // handle negative slice
            if start < 0 {
                start = (shape_mut[dim] as isize + start).max(0);
            }
            if stop < 0 {
                stop = (shape_mut[dim] as isize + stop).max(0);
            }
            // change shape and stride
            offset += shape_mut[dim] as isize * start;
            shape_mut[dim] = ((stop - start + step - 1) / step).max(0) as usize;
            stride_mut[dim] = stride_mut[dim] * step;
        }

        let offset = offset as usize;
        return Ok(Self::new(Shape(shape), Stride(stride), offset));
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
        let Shape(shape) = self.as_shape();
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
