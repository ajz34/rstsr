use super::*;
use itertools::izip;

/// Basic layout iteration trait. Any layout iteration struct should implement
/// this trait.
pub trait IterLayoutBaseAPI {
    /// Dimension type that actually be indexed
    type D: DimAPI;
    /// Dimension type for iterator constructor
    type Din: DimAPI;
    /// Iterator constructor
    fn new(layout: &Layout<Self::Din>) -> Self;
    /// Combined getter for layout, index, offset
    fn combined_getter(&mut self) -> (&Layout<Self::D>, &mut Option<Self::D>, &mut usize);
}

/* #region row-major */

/// Basic layout iteration struct.
///
/// This iteration will naively iterate over all elements by row-major.
#[derive(Clone, Debug)]
pub struct IterLayoutRowMajor<D>
where
    D: DimAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<D>,
    offset: usize,
}

impl<D> IterLayoutBaseAPI for IterLayoutRowMajor<D>
where
    D: DimAPI,
{
    type D = D;
    type Din = D;

    fn new(layout: &Layout<D>) -> Self {
        let layout = layout.clone();
        if layout.ndim() == 0 {
            return Self { layout, index: None, offset: 0 };
        }
        let mut last_index = layout.shape().0.clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        Self { layout, index: Some(last_index), offset: 0 }
    }

    fn combined_getter(&mut self) -> (&Layout<D>, &mut Option<D>, &mut usize) {
        (&self.layout, &mut self.index, &mut self.offset)
    }
}

/// Trait for layout iteration, generates next index from previous for row-major
/// case.
pub trait IterLayoutRowMajorAPI: IterLayoutBaseAPI {
    /// Get the next index, but note that this operation shall handle index
    /// iterator in-place.
    fn next_index(&mut self) -> Option<&Self::D> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let mut done = false;
        for (shape, idx, stride) in
            izip!(layout.shape_ref().as_ref(), index_in.as_mut(), layout.stride_ref().as_ref())
                .rev()
        {
            *idx += 1;
            *offset = (*offset as isize + stride) as usize;
            if idx == shape {
                *idx = 0;
                *offset = (*offset as isize - *shape as isize * stride) as usize;
            } else {
                done = true;
                break;
            }
        }
        if done {
            return Some(index.as_mut().unwrap());
        } else {
            *index = None;
            return None;
        }
    }
}

impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix0> {
    fn next_index(&mut self) -> Option<&Self::D> {
        self.index = None;
        return None;
    }
}

impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix1> {
    fn next_index(&mut self) -> Option<&Self::D> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        index_in[0] += 1;
        *offset = (*offset as isize + stride[0]) as usize;
        if index_in[0] == shape[0] {
            *index = None;
            return None;
        } else {
            return Some(index.as_mut().unwrap());
        }
    }
}

impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix2> {
    fn next_index(&mut self) -> Option<&Self::D> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        index_in[1] += 1;
        *offset = (*offset as isize + stride[1]) as usize;
        if index_in[1] == shape[1] {
            index_in[1] = 0;
            *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
            index_in[0] += 1;
            *offset = (*offset as isize + stride[0]) as usize;
            if index_in[0] == shape[0] {
                *index = None;
                return None;
            }
        }
        return Some(index.as_mut().unwrap());
    }
}

impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix3> {
    fn next_index(&mut self) -> Option<&Self::D> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        index_in[2] += 1;
        *offset = (*offset as isize + stride[2]) as usize;
        if index_in[2] == shape[2] {
            index_in[2] = 0;
            *offset = (*offset as isize - shape[2] as isize * stride[2]) as usize;
            index_in[1] += 1;
            *offset = (*offset as isize + stride[1]) as usize;
            if index_in[1] == shape[1] {
                index_in[1] = 0;
                *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                    return None;
                }
            }
        }
        return Some(index.as_mut().unwrap());
    }
}

impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix4> {}
impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix5> {}
impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix6> {}
impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix7> {}
impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix8> {}
impl IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix9> {}
impl IterLayoutRowMajorAPI for IterLayoutRowMajor<IxD> {}

impl<D> Iterator for IterLayoutRowMajor<D>
where
    D: DimAPI,
    Self: IterLayoutRowMajorAPI,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index.is_none() {
            return None;
        }
        let index = self.index.as_ref().unwrap();
        let offset = unsafe { self.layout.index_uncheck_by_ref(&index) };
        self.next_index();
        return Some(offset);
    }
}

#[test]
fn test_iter_layout_row_major() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutRowMajor::new(&layout);
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = [2, 3, 4].f();
    let iter = IterLayoutRowMajor::new(&layout);
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
}

/* #endregion */
