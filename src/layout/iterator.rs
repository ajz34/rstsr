use crate::prelude_dev::*;

/// Basic layout iteration trait. Any layout iteration struct should implement
/// this trait.
pub trait DimIterLayoutBaseAPI<It>: DimDevAPI {
    /// Iterator constructor
    fn new_it(layout: &Layout<Self>) -> Result<It>;
}

/// Trait for layout iteration, generates next index from previous for row-major
/// case.
pub trait DimIterLayoutAPI<It>: DimIterLayoutBaseAPI<It> {
    /// Get the next index, but note that this operation shall handle index
    /// iterator in-place.
    fn next_iter_index(it_obj: &mut It);
}

pub trait IterLayoutBaseAPI<D>: Sized
where
    D: DimIterLayoutBaseAPI<Self> + DimIterLayoutAPI<Self>,
{
    fn new_it(layout: &Layout<D>) -> Result<Self> {
        D::new_it(layout)
    }
    fn next_index(&mut self) {
        D::next_iter_index(self);
    }
}

/* #region row-major */

/// Basic layout iteration struct.
///
/// This iteration will naively iterate over all elements by row-major.
#[derive(Clone, Debug)]
pub struct IterLayoutC<D>
where
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<D>,
    offset: usize,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutC<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutC<D>> {
        type It<D> = IterLayoutC<D>;
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(It::<D> { layout, index: None, offset });
        }
        let mut last_index = layout.shape().clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        return Ok(It::<D> { layout, index: Some(last_index), offset });
    }
}

impl<const N: usize> DimIterLayoutAPI<IterLayoutC<Ix<N>>> for Ix<N> {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutC<Ix<N>>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        };
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
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

impl DimIterLayoutAPI<IterLayoutC<IxD>> for IxD {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutC<IxD>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape: &[usize] = layout.shape().as_ref();
        let stride: &[isize] = layout.stride().as_ref();
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

impl<D> Iterator for IterLayoutC<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        D::next_iter_index(self);
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutC<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutC<D> where D: DimIterLayoutAPI<Self> {}

#[test]
fn test_iter_layout_row_major() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(4, 3, 2).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].f();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutC::new_it(&layout).unwrap();
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
pub struct IterLayoutF<D>
where
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<D>,
    offset: usize,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutF<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutF<D>> {
        type It<D> = IterLayoutF<D>;
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(It::<D> { layout, index: None, offset });
        }
        let mut last_index = layout.shape().clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        return Ok(It::<D> { layout, index: Some(last_index), offset });
    }
}

impl<const N: usize> DimIterLayoutAPI<IterLayoutF<Ix<N>>> for Ix<N> {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutF<Ix<N>>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
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

impl DimIterLayoutAPI<IterLayoutF<IxD>> for IxD {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutF<IxD>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape: &[usize] = layout.shape().as_ref();
        let stride: &[isize] = layout.stride().as_ref();
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

impl<D> Iterator for IterLayoutF<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        D::next_iter_index(self);
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutF<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutF<D> where D: DimIterLayoutAPI<Self> {}

#[test]
fn test_iter_layout_col_major() {
    let layout = [2, 3, 4].f();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(2, 3, 4).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].c();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutF::new_it(&layout).unwrap();
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
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<usize>,
    idx_max: usize,
    offset: usize,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutMemNonStrided<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutMemNonStrided<D>> {
        type It<D> = IterLayoutMemNonStrided<D>;
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(It::<D> { layout, index: None, idx_max: 0, offset });
        }
        let (idx_min, idx_max) = layout.bounds_index()?;
        rstsr_assert_eq!(idx_max - idx_min, layout.size(), InvalidLayout)?;
        return Ok(It::<D> { layout, index: Some(idx_min), idx_max, offset });
    }
}

impl<D> DimIterLayoutAPI<IterLayoutMemNonStrided<D>> for D
where
    D: DimDevAPI,
{
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutMemNonStrided<D>) {
        if let Some(index) = it_obj.index.as_mut() {
            *index += 1;
            it_obj.offset += 1;
            if *index == it_obj.idx_max {
                it_obj.index = None;
            }
        }
    }
}

impl<D> Iterator for IterLayoutMemNonStrided<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        D::next_iter_index(self);
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutMemNonStrided<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutMemNonStrided<D> where D: DimIterLayoutAPI<Self> {}

#[test]
fn test_iter_layout_mem_non_strided() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = [2, 3, 4].f();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f().swapaxes(1, 2).unwrap();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
}

/* #endregion */

/* #region greedy-major */

pub struct IterLayoutGreedy<D>
where
    D: DimDevAPI,
{
    pub(crate) inner: IterLayoutF<D>,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutGreedy<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutGreedy<D>> {
        let layout = layout.greedy_layout();
        let inner = D::new_it(&layout)?;
        return Ok(IterLayoutGreedy::<D> { inner });
    }
}

impl<D> DimIterLayoutAPI<IterLayoutGreedy<D>> for D
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn next_iter_index(it_obj: &mut IterLayoutGreedy<D>) {
        D::next_iter_index(&mut it_obj.inner);
    }
}

impl<D> Iterator for IterLayoutGreedy<D>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>,
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

impl<D> ExactSizeIterator for IterLayoutGreedy<D>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn len(&self) -> usize {
        self.inner.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutGreedy<D> where D: DimIterLayoutAPI<Self> {}

/* #endregion */

/* #region enum of iterator */

pub enum IterLayoutEnum<D, const CHG: bool>
where
    D: DimDevAPI,
{
    C(IterLayoutC<D>),
    F(IterLayoutF<D>),
    MemNonStrided(IterLayoutMemNonStrided<D>),
    GreedyMajor(IterLayoutGreedy<D>),
}

impl<D, const CHG: bool> DimIterLayoutBaseAPI<IterLayoutEnum<D, CHG>> for D
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutEnum<D, CHG>> {
        type It<D, const CHG: bool> = IterLayoutEnum<D, CHG>;
        // this implementation generates the most efficient iterator, but not the
        // standard layout.
        let layout = layout.clone();
        match CHG {
            false => match (layout.is_c_prefer(), layout.is_f_prefer()) {
                (true, false) => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                (false, true) => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                (_, _) => match Order::default() {
                    Order::C => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                    Order::F => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                },
            },
            true => {
                let iter_mem_non_strided = IterLayoutMemNonStrided::new_it(&layout);
                if let Ok(it) = iter_mem_non_strided {
                    Ok(It::<D, CHG>::MemNonStrided(it))
                } else {
                    match (layout.is_c_prefer(), layout.is_f_prefer()) {
                        (true, false) => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                        (false, true) => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                        (true, true) => match Order::default() {
                            Order::C => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                            Order::F => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                        },
                        (false, false) => {
                            Ok(It::<D, CHG>::GreedyMajor(IterLayoutGreedy::new_it(&layout)?))
                        },
                    }
                }
            },
        }
    }
}

impl<D, const CHG: bool> DimIterLayoutAPI<IterLayoutEnum<D, CHG>> for D
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn next_iter_index(it_obj: &mut IterLayoutEnum<D, CHG>) {
        type It<D, const CHG: bool> = IterLayoutEnum<D, CHG>;
        match it_obj {
            It::<D, CHG>::C(it) => it.next_index(),
            It::<D, CHG>::F(it) => it.next_index(),
            It::<D, CHG>::MemNonStrided(it) => it.next_index(),
            It::<D, CHG>::GreedyMajor(it) => it.next_index(),
        }
    }
}

impl<D, const CHG: bool> Iterator for IterLayoutEnum<D, CHG>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::C(it) => it.next(),
            Self::F(it) => it.next(),
            Self::MemNonStrided(it) => it.next(),
            Self::GreedyMajor(it) => it.next(),
        }
    }
}

impl<D, const CHG: bool> ExactSizeIterator for IterLayoutEnum<D, CHG>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn len(&self) -> usize {
        match self {
            Self::C(it) => it.len(),
            Self::F(it) => it.len(),
            Self::MemNonStrided(it) => it.len(),
            Self::GreedyMajor(it) => it.len(),
        }
    }
}

impl<D, const CHG: bool> IterLayoutBaseAPI<D> for IterLayoutEnum<D, CHG> where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>
{
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterLayoutType {
    C,
    F,
    MemNonStrided,
    GreedyMajor,
}

pub fn iter_layout_by_type<D>(
    ty: IterLayoutType,
    layout: &Layout<D>,
) -> Result<IterLayoutEnum<D, true>>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    match ty {
        IterLayoutType::C => Ok(IterLayoutEnum::C(IterLayoutC::new_it(layout)?)),
        IterLayoutType::F => Ok(IterLayoutEnum::F(IterLayoutF::new_it(layout)?)),
        IterLayoutType::MemNonStrided => {
            Ok(IterLayoutEnum::MemNonStrided(IterLayoutMemNonStrided::new_it(layout)?))
        },
        IterLayoutType::GreedyMajor => {
            Ok(IterLayoutEnum::GreedyMajor(IterLayoutGreedy::new_it(layout)?))
        },
    }
}

/* #endregion */

pub trait IterLayoutAPI<D>:
    IterLayoutBaseAPI<D> + Iterator<Item = usize> + ExactSizeIterator
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
}

impl<D> IterLayoutAPI<D> for IterLayoutC<D> where D: DimDevAPI + DimIterLayoutAPI<Self> {}
impl<D> IterLayoutAPI<D> for IterLayoutF<D> where D: DimDevAPI + DimIterLayoutAPI<Self> {}
impl<D> IterLayoutAPI<D> for IterLayoutMemNonStrided<D> where D: DimDevAPI + DimIterLayoutAPI<Self> {}
impl<D> IterLayoutAPI<D> for IterLayoutGreedy<D> where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>
{
}
impl<D, const CHG: bool> IterLayoutAPI<D> for IterLayoutEnum<D, CHG> where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>
{
}
