//! Searchsorted kernels (serial): binary search of sorted 1-D `x1` for the
//! values of `x2`.

use core::cmp::Ordering;

use crate::prelude_dev::*;
use rstsr_dtype_traits::ExtSortCmp;

/// Binary search one value `v` in the sorted sequence `x1` (as permuted by
/// `sorter`, when given).
///
/// Returns the insertion position: `side_left` picks the first `i` with
/// `x1[i] >= v` (i.e. all earlier elements `< v`); `side_right` picks the
/// first `i` with `x1[i] > v`. NaN keys land after all finite values
/// (consistent with the sort order of [`ExtSortCmp`]).
pub fn searchsorted_value_cpu_serial<'a, T>(
    _x1: &'a [T],
    n: usize,
    v: &T,
    side_left: bool,
    at: &dyn Fn(usize) -> &'a T,
    is_nan: &dyn Fn(&T) -> bool,
) -> usize
where
    T: ExtSortCmp,
{
    let key_cmp = |a: &T, b: &T| a.ext_total_cmp(b);
    let nan_v = is_nan(v);
    let mut lo = 0_usize;
    let mut hi = n;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let x = at(mid);
        let x_nan = is_nan(x);
        let go_right = match (x_nan, nan_v) {
            // NaN is the tail of the order: any NaN key goes after finite x;
            // NaN keys compare Equal among themselves
            (false, true) => true,
            (true, false) => false,
            _ => {
                let ord = key_cmp(x, v);
                if side_left {
                    ord == Ordering::Less
                } else {
                    ord != Ordering::Greater
                }
            },
        };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Searchsorted over all values of `x2` (any shape) into a contiguous output
/// of the same shape (`layout_c`).
pub fn searchsorted_cpu_serial<T, D2>(
    c: &mut [MaybeUninit<usize>],
    layout_c: &Layout<IxD>,
    x1: &[T],
    l1: &Layout<IxD>,
    x2: &[T],
    l2: &Layout<D2>,
    side_left: bool,
    sorter: Option<&[usize]>,
    is_nan: &dyn Fn(&T) -> bool,
) -> Result<()>
where
    T: ExtSortCmp,
    D2: DimAPI,
{
    rstsr_assert_eq!(l1.ndim(), 1, InvalidLayout, "searchsorted requires a one-dimensional x1.")?;
    let n = l1.shape()[0];
    let stride1: isize = l1.stride()[0];
    let base1 = l1.offset() as isize;
    // element accessor through x1's layout (offset + stride), then the sorter
    let at = |i: usize, sorter: Option<&[usize]>| -> &T {
        let pos = match sorter {
            Some(perm) => perm[i],
            None => i,
        };
        let j = base1 + stride1 * pos as isize;
        &x1[j as usize]
    };
    // iterate x2 in its own K order; write into the contiguous output
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&l2.to_dim()?, RowMajor)?;
    let stride_ref: &[isize] = layout_c.stride().as_ref();
    let out_strides: Vec<usize> = stride_ref.iter().map(|&s| s.unsigned_abs()).collect();
    let ndim_out = layout_c.ndim();
    for (index, off) in iter {
        let index_ref: &[usize] = index.as_ref();
        let mut out_pos = 0_usize;
        for (i, &v) in index_ref.iter().enumerate() {
            out_pos += v * out_strides[i];
        }
        let _ = ndim_out;
        let pos = searchsorted_value_cpu_serial(x1, n, &x2[off], side_left, &|i| at(i, sorter), is_nan);
        c[out_pos].write(pos);
    }
    Ok(())
}
