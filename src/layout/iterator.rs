use crate::prelude_dev::*;

// type alias for this file
type Order = TensorIterOrder;

/* #region translate tensor order to col-major with TensorIterType */

/// This function will return a f-prefer layout that make minimal memory
/// accessing efforts (pointers will not frequently back-and-forth).
///
/// Note that this function should only be used for iteration.
///
/// # Parameter `keep_shape`
///
/// Keep size of output layout when input layout is boardcasted.
/// This option should be false if [`TensorIterOrder::K`] and true if
/// [`TensorIterOrder::G`].
///
/// For example of layout shape `[5, 1, 2, 1, 3, 6]` and stride `[1000, 10, 10,
/// 40, 0, 100]`,
/// - false: shape `[2, 6, 5, 1, 1, 1]` and stride `[10, 100, 1000, 0, 0, 0]`;
///   meaning that broadcasted shapes are eliminated and moved to last axes.
/// - true: shape `[3, 1, 1, 2, 6, 5]` and stride `[0, 10, 40, 10, 100, 1000]`;
///   meaning that broadcasted shapes are iterated with most priority.
///
/// # Returns
///
/// - `layout`: The output layout of greedy iteration.
/// - `index`: Transpose index from input layout to output layout.
pub fn greedy_layout<D>(layout: &Layout<D>, keep_shape: bool) -> (Layout<D>, Vec<usize>)
where
    D: DimDevAPI,
{
    // if no elements in layout, return itself
    if layout.size() == 0 {
        return (layout.clone(), (0..layout.ndim()).collect::<Vec<usize>>());
    }

    // revert negative strides
    let mut layout = layout.clone();
    for n in 0..layout.ndim() {
        if layout.stride()[n] < 0 {
            // should not panic here
            layout = layout.dim_narrow(n as isize, slice!(None, None, -1)).unwrap();
        }
    }

    let shape_old = layout.shape.as_ref();
    let stride_old = layout.stride.as_ref();

    let mut index = (0..layout.ndim()).collect::<Vec<usize>>();
    if keep_shape {
        // sort shape and strides if keep shape
        // - (shape = 1 / stride = 0) the smallest (pointer not moving for these cases)
        // - if (shape = 1 / stride = 0, broadcastable axes) preserve order
        // - (larger shape first) if not broadcastable axes, then compare stride size
        //   (smaller stride first)
        index.sort_by(|&i1, &i2| {
            let d1 = shape_old[i1];
            let d2 = shape_old[i2];
            let t1 = stride_old[i1];
            let t2 = stride_old[i2];
            match (d1 == 1 || t1 == 0, d2 == 1 || t2 == 0) {
                (true, true) => i1.cmp(&i2),
                (true, false) => core::cmp::Ordering::Less,
                (false, true) => core::cmp::Ordering::Greater,
                (false, false) => t1.cmp(&t2),
            }
        });
    } else {
        // sort shape and strides if not keep shape
        // everything is similar, though broadcastable axes should be moved to last
        index.sort_by(|&i1, &i2| {
            let d1 = shape_old[i1];
            let d2 = shape_old[i2];
            let t1 = stride_old[i1];
            let t2 = stride_old[i2];
            match (d1 == 1 || t1 == 0, d2 == 1 || t2 == 0) {
                (true, true) => i1.cmp(&i2),
                (true, false) => core::cmp::Ordering::Greater,
                (false, true) => core::cmp::Ordering::Less,
                (false, false) => t1.cmp(&t2),
            }
        });
    }

    let index_isize = index.iter().map(|&i| i as isize).collect::<Vec<isize>>();
    let mut layout = layout.transpose(&index_isize).unwrap();

    // for case of not keep shape, dimension of broadcastable axes will be set to 1,
    // strides will be set to 0.
    if !keep_shape {
        let mut shape = layout.shape().clone();
        let mut stride = layout.stride().clone();
        shape.as_mut().iter_mut().zip(stride.as_mut().iter_mut()).for_each(|(d, t)| {
            if *d == 1 || *t == 0 {
                *d = 1;
                *t = 0;
            }
        });
        layout = unsafe { Layout::new_unchecked(shape, stride, layout.offset()) };
    }

    return (layout, index);
}

/// Translate one layout to column-major iteration.
///
/// For how parameter `it_type` works, we refer to definition of
/// [`TensorIterOrder`].
///
/// - C: reverse axes
/// - F: preserve axes
/// - A: B if contiguous, C if c-prefer, F if f-prefer; otherwise default
/// - K: greedy layout, keep shape
/// - G: greedy layout, eliminate broadcastable dimensions
/// - B: sequential memory; valid option if `size = bound_max - bound_min`,
///   otherwise raise err
pub fn translate_to_col_major_unary<D>(
    layout: &Layout<D>,
    it_ord: TensorIterOrder,
) -> Result<Layout<D>>
where
    D: DimAPI,
{
    let fn_c = |l: &Layout<D>| Ok(l.reverse_axes());
    let fn_f = |l: &Layout<D>| Ok(l.clone());
    let fn_b = |l: &Layout<D>| {
        let (bounds_min, bounds_max) = l.bounds_index()?;
        rstsr_assert_eq!(
            bounds_max - bounds_min,
            l.size(),
            InvalidLayout,
            "Data in this layout could not be represented as sequential memory."
        )?;
        let mut shape = l.new_shape();
        let mut stride = l.new_stride();
        shape[0] = l.size();
        stride[0] = 1;
        for i in 1..l.ndim() {
            shape[i] = 1;
            stride[i] = l.size() as isize;
        }
        Ok(unsafe { Layout::new_unchecked(shape, stride, l.offset()) })
    };
    match it_ord {
        Order::C => fn_c(layout),
        Order::F => fn_f(layout),
        Order::A => {
            let c_contig = layout.c_contig();
            let f_contig = layout.f_contig();
            if c_contig || f_contig {
                fn_b(layout)
            } else {
                let c_prefer = layout.c_prefer();
                let f_prefer = layout.f_prefer();
                match (c_prefer, f_prefer) {
                    (true, false) => fn_c(layout),
                    (false, true) => fn_f(layout),
                    (_, _) => match TensorOrder::default() {
                        TensorOrder::C => fn_c(layout),
                        TensorOrder::F => fn_f(layout),
                    },
                }
            }
        },
        Order::K => Ok(greedy_layout(layout, true).0),
        Order::G => Ok(greedy_layout(layout, false).0),
        Order::B => fn_b(layout),
    }
}

/// Translate multiple layouts to column-major iteration.
///
/// This function requires all layouts have the same shape.
///
/// For how parameter `it_type` works, we refer to definition of
/// [`TensorIterOrder`].
///
/// - C: reverse axes
/// - F: preserve axes
/// - A: B if contiguous, C if c-prefer, F if f-prefer; otherwise default
/// - K: greedy layout for the one which have the largest non-broadcast-size,
///   otherwise left-most layout (usually for mutable-assign/inplace-op)
/// - G: invalid option here
/// - B:sequential memory; valid option if `size = bound_max - bound_min`,
///   otherwise raise err
pub fn translate_to_col_major<D>(
    layouts: &[&Layout<D>],
    it_ord: TensorIterOrder,
) -> Result<Vec<Layout<D>>>
where
    D: DimAPI,
{
    if layouts.is_empty() {
        return Ok(vec![]);
    }

    // this function will map all layouts to column-major iteration by a single
    // iter-order.
    let fn_single = |ls: &[&Layout<D>], it_type| {
        ls.iter().map(|l| translate_to_col_major_unary(l, it_type)).collect()
    };

    // make sure all layouts have the same shape
    let is_same_shape = layouts.windows(2).all(|w| w[0].shape() == w[1].shape());
    rstsr_assert!(
        is_same_shape,
        InvalidLayout,
        "All shape of layout in this function must be the same."
    )?;

    match it_ord {
        Order::C | Order::F | Order::B => fn_single(layouts, it_ord),
        Order::A => {
            let c_contig = layouts.iter().all(|&l| l.c_contig());
            let f_contig = layouts.iter().all(|&l| l.f_contig());
            if c_contig || f_contig {
                fn_single(layouts, TensorIterOrder::B)
            } else {
                let c_prefer = layouts.iter().all(|&l| l.c_contig());
                let f_prefer = layouts.iter().all(|&l| l.f_contig());
                match (c_prefer, f_prefer) {
                    (true, false) => fn_single(layouts, TensorIterOrder::C),
                    (false, true) => fn_single(layouts, TensorIterOrder::F),
                    (_, _) => match TensorOrder::default() {
                        TensorOrder::C => fn_single(layouts, TensorIterOrder::C),
                        TensorOrder::F => fn_single(layouts, TensorIterOrder::F),
                    },
                }
            }
        },
        Order::K => {
            // find the layout with the largest non-broadcast-size
            let size_iter = layouts.iter().map(|l| l.size_non_broadcast()).collect::<Vec<_>>();
            let idx_layout = if size_iter.iter().max() == size_iter.iter().min() {
                0
            } else {
                size_iter.into_iter().enumerate().max_by_key(|(_, v)| *v).unwrap_or((0, 0)).0
            };
            // make same permutation for all layouts
            let (_, permute_index) = greedy_layout(layouts[idx_layout], true);
            let permute_index = permute_index.iter().map(|&i| i as isize).collect::<Vec<isize>>();
            layouts.iter().map(|l| l.transpose(&permute_index)).collect()
        },
        Order::G => rstsr_invalid!(it_ord, "This option is not valid for multiple layouts")?,
    }
}

/// This function will return minimal dimension layout, that the first axis is
/// f-contiguous.
///
/// For example, if shape [2, 4, 6, 8, 10] is contiguous in f-order for the
/// first three axes, then it will return shape [48, 8, 10], and the contiguous
/// size 48.
///
/// # Notes
///
/// - Should be used after [`translate_to_col_major`].
/// - Accepts multiple layouts to be compared.
/// - Due to that final dimension is not known to compiler, this function will
///   return dynamic layout.
pub fn translate_to_col_major_with_contig<D>(layouts: &[&Layout<D>]) -> (Vec<Layout<IxD>>, usize)
where
    D: DimAPI,
{
    if layouts.is_empty() {
        return (vec![], 0);
    }

    let dims_f_contig = layouts.iter().map(|l| l.ndim_of_f_contig()).collect::<Vec<usize>>();
    let ndim_f_contig = *dims_f_contig.iter().min().unwrap();
    // following is the worst case: no axes are contiguous in f-order
    if ndim_f_contig == 0 {
        return (layouts.iter().map(|&l| l.clone().into_dim::<IxD>().unwrap()).collect(), 0);
    } else {
        let size_contig = layouts[0].shape().as_ref()[0..ndim_f_contig].iter().product::<usize>();
        let result = layouts
            .iter()
            .map(|l| {
                let shape = l.shape().as_ref()[ndim_f_contig..].iter().cloned().collect_vec();
                let stride = l.stride().as_ref()[ndim_f_contig..].iter().cloned().collect_vec();
                unsafe { Layout::new_unchecked(shape, stride, l.offset()) }
            })
            .collect::<Vec<_>>();
        return (result, size_contig);
    }
}

/* #endregion */

/* #region new code */

/// Layout iterator.
///
/// This iterator only handles column-major iterator.
/// For other iteration orders, use function [`translate_to_col_major`] to
/// generate the corresponding col-major (f-prefer) layout, then iterate as
/// col-major.
///
/// # Note
///
/// This crate implements col-major iterator only; the layout iterator that
/// actaully works is internal realization; though it's public struct, it is not
/// intended to be exposed to user.
/// Choosing col-major iterator is because it is possibly the most efficient
/// way. It is not related to default order, which could be defined by crate
/// feature `c_prefer`.
#[derive(Clone, Debug)]
pub struct IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    layout: Layout<D>,

    index_start: D, // this is not used for buffer-order
    iter_start: usize,
    offset_start: usize,

    index_end: D, // this is not used for buffer-order
    iter_end: usize,
    offset_end: usize,
}

impl<D> IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    /// This function generates col-major (f-prefer) layout, then give its
    /// iterator object.
    pub fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.clone();
        let iter_start = 0;
        let iter_end = layout.size();
        let index_start = layout.new_shape();
        let index_end = unsafe { layout.unravel_index_f(iter_end) };
        let offset_start = layout.offset();
        let offset_end = unsafe { layout.index_uncheck(index_end.as_ref()) };

        return Ok(Self {
            layout,
            index_start,
            iter_start,
            offset_start,
            index_end,
            iter_end,
            offset_end,
        });
    }
}

impl<D> IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    #[inline]
    fn next_iter_index(&mut self) {
        let layout = &self.layout;
        let index = self.index_start.as_mut();
        let mut offset = self.offset_start as isize;
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
        match layout.ndim() {
            0 => (),
            1 => {
                index[0] += 1;
                offset += stride[0];
            },
            2 => {
                index[0] += 1;
                offset += stride[0];
                if index[0] == shape[0] {
                    index[0] = 0;
                    offset -= shape[0] as isize * stride[0];
                    index[1] += 1;
                    offset += stride[1];
                }
            },
            3 => {
                index[0] += 1;
                offset += stride[0];
                if index[0] == shape[0] {
                    index[0] = 0;
                    offset -= shape[0] as isize * stride[0];
                    index[1] += 1;
                    offset += stride[1];
                    if index[1] == shape[1] {
                        index[1] = 0;
                        offset -= shape[1] as isize * stride[1];
                        index[2] += 1;
                        offset += stride[2];
                    }
                }
            },
            4 => {
                index[0] += 1;
                offset += stride[0];
                if index[0] == shape[0] {
                    index[0] = 0;
                    offset -= shape[0] as isize * stride[0];
                    index[1] += 1;
                    offset += stride[1];
                    if index[1] == shape[1] {
                        index[1] = 0;
                        offset -= shape[1] as isize * stride[1];
                        index[2] += 1;
                        offset += stride[2];
                        if index[2] == shape[2] {
                            index[2] = 0;
                            offset -= shape[2] as isize * stride[2];
                            index[3] += 1;
                            offset += stride[3];
                        }
                    }
                }
            },
            _ => {
                for (d, t, idx) in izip!(shape, stride, index.as_mut(),) {
                    *idx += 1;
                    offset += t;
                    if idx == d {
                        *idx = 0;
                        offset -= *d as isize * t;
                    } else {
                        break;
                    }
                }
            },
        }
        self.offset_start = offset as usize;
        self.iter_start += 1;
    }

    #[inline]
    fn back_iter_index(&mut self) {
        let layout = &self.layout;
        let index = self.index_end.as_mut();
        let mut offset = self.offset_end as isize;
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
        match layout.ndim() {
            0 => (),
            1 => {
                index[0] -= 1;
                offset -= stride[0];
            },
            2 => {
                if index[0] == 0 {
                    index[0] = shape[0] - 1;
                    offset += (shape[0] - 1) as isize * stride[0];
                    index[1] -= 1;
                    offset -= stride[1];
                } else {
                    index[0] -= 1;
                    offset -= stride[0];
                }
            },
            3 => {
                if index[0] == 0 {
                    index[0] = shape[0] - 1;
                    offset += (shape[0] - 1) as isize * stride[0];
                    if index[1] == 0 {
                        index[1] = shape[1] - 1;
                        offset += (shape[1] - 1) as isize * stride[1];
                        index[2] -= 1;
                        offset -= stride[2];
                    } else {
                        index[1] -= 1;
                        offset -= stride[1];
                    }
                } else {
                    index[0] -= 1;
                    offset -= stride[0];
                }
            },
            4 => {
                if index[0] == 0 {
                    index[0] = shape[0] - 1;
                    offset += (shape[0] - 1) as isize * stride[0];
                    if index[1] == 0 {
                        index[1] = shape[1] - 1;
                        offset += (shape[1] - 1) as isize * stride[1];
                        if index[2] == 0 {
                            index[2] = shape[2] - 1;
                            offset += (shape[2] - 1) as isize * stride[2];
                            index[3] -= 1;
                            offset -= stride[3];
                        } else {
                            index[2] -= 1;
                            offset -= stride[2];
                        }
                    } else {
                        index[1] -= 1;
                        offset -= stride[1];
                    }
                } else {
                    index[0] -= 1;
                    offset -= stride[0];
                }
            },
            _ => {
                for (d, t, idx) in izip!(shape, stride, index.as_mut()) {
                    if *idx == 0 {
                        *idx = *d - 1;
                        offset += (*d - 1) as isize * t;
                    } else {
                        *idx -= 1;
                        offset -= t;
                        break;
                    }
                }
            },
        }
        self.offset_end = offset as usize;
        self.iter_end -= 1;
    }
}

impl<D> Iterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_start >= self.iter_end {
            return None;
        }
        let offset = self.offset_start;
        self.next_iter_index();
        return Some(offset);
    }
}

impl<D> DoubleEndedIterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.iter_start >= self.iter_end {
            return None;
        }
        self.back_iter_index();
        let offset = self.offset_end;
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.iter_end - self.iter_start
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_greedy_layout() {
        unsafe {
            // c-contiguous layout
            let layout = [2, 3, 4].c();
            let (greedy, _) = greedy_layout(&layout, false);
            assert_eq!(greedy, [4, 3, 2].f());
            let (greedy, _) = greedy_layout(&layout, true);
            assert_eq!(greedy, [4, 3, 2].f());
            // f-contiguous layout
            let layout = [2, 3, 4].f();
            let (greedy, _) = greedy_layout(&layout, false);
            assert_eq!(greedy, [2, 3, 4].f());
            let (greedy, _) = greedy_layout(&layout, true);
            assert_eq!(greedy, [2, 3, 4].f());
            // dimension-size 1 or stride-size 0
            let layout = Layout::new_unchecked([5, 1, 2, 1, 3, 6], [1000, 10, 10, 40, 0, 100], 0);
            let (greedy, _) = greedy_layout(&layout, false);
            let expect = Layout::new_unchecked([2, 6, 5, 1, 1, 1], [10, 100, 1000, 0, 0, 0], 0);
            assert_eq!(greedy, expect);
            let (greedy, _) = greedy_layout(&layout, true);
            let expect = Layout::new_unchecked([1, 1, 3, 2, 6, 5], [10, 40, 0, 10, 100, 1000], 0);
            assert_eq!(greedy, expect);
        }
    }

    #[test]
    fn test_iter_next() {
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            782, 797, 812, 827, 842, 857, 602, 617, 632, 647, 662, 677, 785, 800, 815, 830, 845,
            860, 605, 620, 635, 650, 665, 680, 788, 803, 818, 833, 848, 863, 608, 623, 638, 653,
            668, 683
        ]);
        // np.array(np.nditer(a, order="F"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::F).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            782, 785, 788, 602, 605, 608, 797, 800, 803, 617, 620, 623, 812, 815, 818, 632, 635,
            638, 827, 830, 833, 647, 650, 653, 842, 845, 848, 662, 665, 668, 857, 860, 863, 677,
            680, 683
        ]);
        // np.array(np.nditer(a, order="K"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::K).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            602, 605, 608, 617, 620, 623, 632, 635, 638, 647, 650, 653, 662, 665, 668, 677, 680,
            683, 782, 785, 788, 797, 800, 803, 812, 815, 818, 827, 830, 833, 842, 845, 848, 857,
            860, 863
        ]);
        // np.array(np.nditer(a, order="G"))
        // for no broadcast case, greedy-order is same as k-order
        let layout_trans = translate_to_col_major_unary(&layout, Order::K).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            602, 605, 608, 617, 620, 623, 632, 635, 638, 647, 650, 653, 662, 665, 668, 677, 680,
            683, 782, 785, 788, 797, 800, 803, 812, 815, 818, 827, 830, 833, 842, 845, 848, 857,
            860, 863
        ]);
        // buffer should fail
        assert!(translate_to_col_major_unary(&layout, Order::B).is_err());
    }

    #[test]
    fn test_iter_back() {
        let layout = Layout::new([10, 10, 10], [10, 1, 100], 0);
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        println!("{:?}", unsafe { layout.unravel_index_f(100) });
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_next = iter.collect::<Vec<_>>();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_back = iter.rev().collect::<Vec<_>>();
        assert_eq!(vec_next, vec_back.iter().rev().copied().collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_back_empty() {
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_next = iter.clone().collect::<Vec<_>>();
        let vec_back = iter.clone().rev().collect::<Vec<_>>();
        assert_eq!(vec_next, vec_back.iter().rev().copied().collect::<Vec<_>>());

        let layout = Layout::new([10], [10], 10);
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_next = iter.clone().collect::<Vec<_>>();
        let vec_back = iter.clone().rev().collect::<Vec<_>>();
        assert_eq!(vec_next, vec_back.iter().rev().copied().collect::<Vec<_>>());
    }
}
