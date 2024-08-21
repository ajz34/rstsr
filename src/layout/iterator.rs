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
    fn next_index(&mut self) -> Option<&Self::D>;
}

impl<const N: usize> IterLayoutRowMajorAPI for IterLayoutRowMajor<Ix<N>> {
    #[inline]
    fn next_index(&mut self) -> Option<&Ix<N>> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        match N {
            0 => {
                *index = None;
                return None;
            },
            1 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                    return None;
                }
                return Some(index.as_mut().unwrap());
            },
            2 => {
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
            },
            3 => {
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
            },
            4 => {
                index_in[3] += 1;
                *offset = (*offset as isize + stride[3]) as usize;
                if index_in[3] == shape[3] {
                    index_in[3] = 0;
                    *offset = (*offset as isize - shape[3] as isize * stride[3]) as usize;
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
                }
                return Some(index.as_mut().unwrap());
            },
            _ => {
                let mut done = false;
                for (d, t, idx) in izip!(shape, stride, index_in).rev() {
                    *idx += 1;
                    *offset = (*offset as isize + t) as usize;
                    if idx == d {
                        *idx = 0;
                        *offset = (*offset as isize - *d as isize * t) as usize;
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
            },
        }
    }
}

impl IterLayoutRowMajorAPI for IterLayoutRowMajor<IxD> {
    #[inline]
    fn next_index(&mut self) -> Option<&IxD> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape: &[usize] = layout.shape_ref().as_ref();
        let stride: &[isize] = layout.stride_ref().as_ref();
        let mut done = false;
        for (d, t, idx) in izip!(shape, stride, index_in).rev() {
            *idx += 1;
            *offset = (*offset as isize + t) as usize;
            if idx == d {
                *idx = 0;
                *offset = (*offset as isize - *d as isize * t) as usize;
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

impl<D> Iterator for IterLayoutRowMajor<D>
where
    D: DimAPI,
    Self: IterLayoutRowMajorAPI,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        self.next_index();
        return Some(offset);
    }
}

#[test]
fn test_iter_layout_row_major() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutRowMajor::new(&layout);
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutRowMajor::new(&layout);
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(4, 3, 2).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].f();
    let iter = IterLayoutRowMajor::new(&layout);
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutRowMajor::new(&layout);
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
}

/* #endregion */

/* #region col-major */

/// Basic layout iteration struct.
///
/// This iteration will naively iterate over all elements by row-major.
#[derive(Clone, Debug)]
pub struct IterLayoutColMajor<D>
where
    D: DimAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<D>,
    offset: usize,
}

impl<D> IterLayoutBaseAPI for IterLayoutColMajor<D>
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

/// Trait for layout iteration, generates next index from previous for col-major
/// case.
pub trait IterLayoutColMajorAPI: IterLayoutBaseAPI {
    /// Get the next index, but note that this operation shall handle index
    /// iterator in-place.
    fn next_index(&mut self) -> Option<&Self::D>;
}

impl<const N: usize> IterLayoutColMajorAPI for IterLayoutColMajor<Ix<N>> {
    #[inline]
    fn next_index(&mut self) -> Option<&Ix<N>> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        match N {
            0 => {
                *index = None;
                return None;
            },
            1 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                    return None;
                }
                return Some(index.as_mut().unwrap());
            },
            2 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    index_in[0] = 0;
                    *offset = (*offset as isize - shape[0] as isize * stride[0]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        *index = None;
                        return None;
                    }
                }
                return Some(index.as_mut().unwrap());
            },
            3 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    index_in[0] = 0;
                    *offset = (*offset as isize - shape[0] as isize * stride[0]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        index_in[1] = 0;
                        *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                        index_in[2] += 1;
                        *offset = (*offset as isize + stride[2]) as usize;
                        if index_in[2] == shape[2] {
                            *index = None;
                            return None;
                        }
                    }
                }
                return Some(index.as_mut().unwrap());
            },
            4 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    index_in[0] = 0;
                    *offset = (*offset as isize - shape[0] as isize * stride[0]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        index_in[1] = 0;
                        *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                        index_in[2] += 1;
                        *offset = (*offset as isize + stride[2]) as usize;
                        if index_in[2] == shape[2] {
                            index_in[2] = 0;
                            *offset = (*offset as isize - shape[2] as isize * stride[2]) as usize;
                            index_in[3] += 1;
                            *offset = (*offset as isize + stride[3]) as usize;
                            if index_in[3] == shape[3] {
                                *index = None;
                                return None;
                            }
                        }
                    }
                }
                return Some(index.as_mut().unwrap());
            },
            _ => {
                let mut done = false;
                for (d, t, idx) in izip!(shape, stride, index_in.as_mut(),) {
                    *idx += 1;
                    *offset = (*offset as isize + t) as usize;
                    if idx == d {
                        *idx = 0;
                        *offset = (*offset as isize - *d as isize * t) as usize;
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
            },
        }
    }
}

impl IterLayoutColMajorAPI for IterLayoutColMajor<IxD> {
    #[inline]
    fn next_index(&mut self) -> Option<&Self::D> {
        let (layout, index, offset) = self.combined_getter();
        if index.is_none() {
            return None;
        }
        let index_in = index.as_mut().unwrap();
        let shape: &[usize] = layout.shape_ref().as_ref();
        let stride: &[isize] = layout.stride_ref().as_ref();
        let mut done = false;
        for (d, t, idx) in izip!(shape, stride, index_in) {
            *idx += 1;
            *offset = (*offset as isize + t) as usize;
            if idx == d {
                *idx = 0;
                *offset = (*offset as isize - *d as isize * t) as usize;
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

impl<D> Iterator for IterLayoutColMajor<D>
where
    D: DimAPI,
    Self: IterLayoutColMajorAPI,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        self.next_index();
        return Some(offset);
    }
}

#[test]
fn test_iter_layout_col_major() {
    let layout = [2, 3, 4].f();
    let iter = IterLayoutColMajor::new(&layout);
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutColMajor::new(&layout);
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(2, 3, 4).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].c();
    let iter = IterLayoutColMajor::new(&layout);
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutColMajor::new(&layout);
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
}

/* #endregion */
