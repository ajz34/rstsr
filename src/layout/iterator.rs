use crate::prelude_dev::*;

/// Basic layout iteration trait. Any layout iteration struct should implement
/// this trait.
pub trait LayoutIterBaseAPI: Sized {
    /// Dimension type that actually be indexed
    type Dim: DimAPI;
    /// Iterator constructor
    fn new(layout: &Layout<Self::Dim>) -> Result<Self>;
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

impl<D> LayoutIterBaseAPI for IterLayoutRowMajor<D>
where
    D: DimAPI,
{
    type Dim = D;

    fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(Self { layout, index: None, offset });
        }
        let mut last_index = layout.shape().0.clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        return Ok(Self { layout, index: Some(last_index), offset });
    }
}

/// Trait for layout iteration, generates next index from previous for row-major
/// case.
pub trait LayoutIteratorAPI: LayoutIterBaseAPI {
    /// Get the next index, but note that this operation shall handle index
    /// iterator in-place.
    fn next_index(&mut self);
}

impl<const N: usize> LayoutIteratorAPI for IterLayoutRowMajor<Ix<N>> {
    #[inline]
    fn next_index(&mut self) {
        let (layout, index, offset) = (&self.layout, &mut self.index, &mut self.offset);
        if index.is_none() {
            return;
        };
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        match N {
            0 => {
                *index = None;
            },
            1 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                }
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
                    }
                }
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
                        }
                    }
                }
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
                            }
                        }
                    }
                }
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
                if !done {
                    *index = None;
                }
            },
        }
    }
}

impl LayoutIteratorAPI for IterLayoutRowMajor<IxD> {
    #[inline]
    fn next_index(&mut self) {
        let (layout, index, offset) = (&self.layout, &mut self.index, &mut self.offset);
        if index.is_none() {
            return;
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
        if !done {
            *index = None;
        }
    }
}

impl<D> Iterator for IterLayoutRowMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        self.next_index();
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutRowMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

#[test]
fn test_iter_layout_row_major() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutRowMajor::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutRowMajor::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(4, 3, 2).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].f();
    let iter = IterLayoutRowMajor::new(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutRowMajor::new(&layout).unwrap();
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

impl<D> LayoutIterBaseAPI for IterLayoutColMajor<D>
where
    D: DimAPI,
{
    type Dim = D;

    fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(Self { layout, index: None, offset });
        }
        let mut last_index = layout.shape().0.clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        return Ok(Self { layout, index: Some(last_index), offset });
    }
}

impl<const N: usize> LayoutIteratorAPI for IterLayoutColMajor<Ix<N>> {
    #[inline]
    fn next_index(&mut self) {
        let (layout, index, offset) = (&self.layout, &mut self.index, &mut self.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape_ref().as_ref();
        let stride = layout.stride_ref().as_ref();
        match N {
            0 => {
                *index = None;
            },
            1 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                }
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
                    }
                }
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
                        }
                    }
                }
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
                            }
                        }
                    }
                }
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
                if !done {
                    *index = None;
                }
            },
        }
    }
}

impl LayoutIteratorAPI for IterLayoutColMajor<IxD> {
    #[inline]
    fn next_index(&mut self) {
        let (layout, index, offset) = (&self.layout, &mut self.index, &mut self.offset);
        if index.is_none() {
            return;
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
        if !done {
            *index = None;
        }
    }
}

impl<D> Iterator for IterLayoutColMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        self.next_index();
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutColMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

#[test]
fn test_iter_layout_col_major() {
    let layout = [2, 3, 4].f();
    let iter = IterLayoutColMajor::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutColMajor::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(2, 3, 4).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].c();
    let iter = IterLayoutColMajor::new(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutColMajor::new(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
}

/* #endregion */

/* #region mem-non-strided */

/// Iterator that only applies to layout that has contiguous memory (not exactly
/// same to c-contig or f-contig).
#[derive(Clone, Debug)]
pub struct IterLayoutMemNonStrided<D>
where
    D: DimAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<usize>,
    idx_max: usize,
    offset: usize,
}

impl<D> LayoutIterBaseAPI for IterLayoutMemNonStrided<D>
where
    D: DimAPI,
{
    type Dim = D;

    fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(Self { layout, index: None, idx_max: 0, offset });
        }
        let (idx_min, idx_max) = layout.bounds_index()?;
        rstsr_assert_eq!(idx_max - idx_min, layout.size(), InvalidLayout)?;
        return Ok(Self { layout, index: Some(idx_min), idx_max, offset });
    }
}

impl<D> LayoutIteratorAPI for IterLayoutMemNonStrided<D>
where
    D: DimAPI,
{
    #[inline]
    fn next_index(&mut self) {
        if let Some(index) = self.index.as_mut() {
            *index += 1;
            self.offset += 1;
            if *index == self.idx_max {
                self.index = None;
            }
        }
    }
}

impl<D> Iterator for IterLayoutMemNonStrided<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        self.next_index();
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutMemNonStrided<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

#[test]
fn test_iter_layout_mem_non_strided() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutMemNonStrided::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutMemNonStrided::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = [2, 3, 4].f();
    let iter = IterLayoutMemNonStrided::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f().swapaxes(1, 2).unwrap();
    let iter = IterLayoutMemNonStrided::new(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
}

/* #endregion */

/* #region greedy-major */

pub struct IterLayoutGreedyMajor<D>
where
    D: DimAPI,
{
    pub(crate) inner: IterLayoutColMajor<D>,
}

impl<D> LayoutIterBaseAPI for IterLayoutGreedyMajor<D>
where
    D: DimAPI,
{
    type Dim = D;

    fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.greedy_layout();
        let inner = IterLayoutColMajor::new(&layout)?;
        return Ok(Self { inner });
    }
}

impl<const N: usize> LayoutIteratorAPI for IterLayoutGreedyMajor<Ix<N>> {
    fn next_index(&mut self) {
        IterLayoutColMajor::<Ix<N>>::next_index(&mut self.inner);
    }
}

impl<D> Iterator for IterLayoutGreedyMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.index.as_ref()?;
        let offset = self.inner.offset;
        self.inner.next_index();
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutGreedyMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
{
    fn len(&self) -> usize {
        self.inner.layout.size()
    }
}

/* #endregion */

/* #region enum of iterator */

pub enum IterLayoutEnum<D>
where
    D: DimAPI,
{
    RowMajor(IterLayoutRowMajor<D>),
    ColMajor(IterLayoutColMajor<D>),
    MemNonStrided(IterLayoutMemNonStrided<D>),
    GreedyMajor(IterLayoutGreedyMajor<D>),
}

impl<D> LayoutIterBaseAPI for IterLayoutEnum<D>
where
    D: DimAPI,
{
    type Dim = D;

    fn new(layout: &Layout<D>) -> Result<Self> {
        // this implementation generates the most efficient iterator, but not the
        // standard layout.
        let layout = layout.clone();
        let iter_mem_non_strided = IterLayoutMemNonStrided::new(&layout);
        if let Ok(it) = iter_mem_non_strided {
            return Ok(Self::MemNonStrided(it));
        } else if layout.is_c_prefer() {
            return Ok(Self::RowMajor(IterLayoutRowMajor::new(&layout)?));
        } else if layout.is_f_prefer() {
            return Ok(Self::ColMajor(IterLayoutColMajor::new(&layout)?));
        } else {
            return Ok(Self::GreedyMajor(IterLayoutGreedyMajor::new(&layout)?));
        }
    }
}

impl<D> LayoutIteratorAPI for IterLayoutEnum<D>
where
    D: DimAPI,
    IterLayoutRowMajor<D>: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
    IterLayoutMemNonStrided<D>: LayoutIteratorAPI,
    IterLayoutGreedyMajor<D>: LayoutIteratorAPI,
{
    fn next_index(&mut self) {
        match self {
            Self::RowMajor(it) => it.next_index(),
            Self::ColMajor(it) => it.next_index(),
            Self::MemNonStrided(it) => it.next_index(),
            Self::GreedyMajor(it) => it.next_index(),
        }
    }
}

impl<D> Iterator for IterLayoutEnum<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
    IterLayoutRowMajor<D>: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
    IterLayoutMemNonStrided<D>: LayoutIteratorAPI,
    IterLayoutGreedyMajor<D>: LayoutIteratorAPI,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::RowMajor(it) => it.next(),
            Self::ColMajor(it) => it.next(),
            Self::MemNonStrided(it) => it.next(),
            Self::GreedyMajor(it) => it.next(),
        }
    }
}

impl<D> ExactSizeIterator for IterLayoutEnum<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
    IterLayoutRowMajor<D>: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
    IterLayoutMemNonStrided<D>: LayoutIteratorAPI,
    IterLayoutGreedyMajor<D>: LayoutIteratorAPI,
{
    fn len(&self) -> usize {
        match self {
            Self::RowMajor(it) => it.len(),
            Self::ColMajor(it) => it.len(),
            Self::MemNonStrided(it) => it.len(),
            Self::GreedyMajor(it) => it.len(),
        }
    }
}

/* #endregion */

pub trait LayoutIterAPI:
    LayoutIterBaseAPI + LayoutIteratorAPI + Iterator<Item = usize> + ExactSizeIterator
{
}

impl<D> LayoutIterAPI for IterLayoutRowMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
}

impl<D> LayoutIterAPI for IterLayoutColMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
{
}

impl<D> LayoutIterAPI for IterLayoutMemNonStrided<D> where D: DimAPI {}

impl<D> LayoutIterAPI for IterLayoutGreedyMajor<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
{
}

impl<D> LayoutIterAPI for IterLayoutEnum<D>
where
    D: DimAPI,
    Self: LayoutIteratorAPI,
    IterLayoutRowMajor<D>: LayoutIteratorAPI,
    IterLayoutColMajor<D>: LayoutIteratorAPI,
    IterLayoutMemNonStrided<D>: LayoutIteratorAPI,
    IterLayoutGreedyMajor<D>: LayoutIteratorAPI,
{
}
