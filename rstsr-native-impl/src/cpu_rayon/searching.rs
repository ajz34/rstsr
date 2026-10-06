//! Searchsorted kernels (rayon): parallel over the values of `x2`.

use crate::cpu_serial::searching::searchsorted_value_cpu_serial;
use crate::prelude_dev::*;
use rstsr_dtype_traits::ExtSortCmp;

// per-value work is a single binary search; parallelize at 64 KiB of f64
const PARALLEL_SWITCH: usize = 8192;

/// Rayon twin of [`crate::cpu_serial::searching::searchsorted_cpu_serial`]:
/// identical output; values are searched in parallel when the workload is
/// large enough.
#[allow(clippy::too_many_arguments)]
pub fn searchsorted_cpu_rayon<T, D2>(
    c: &mut [MaybeUninit<usize>],
    layout_c: &Layout<IxD>,
    x1: &[T],
    l1: &Layout<IxD>,
    x2: &[T],
    l2: &Layout<D2>,
    side_left: bool,
    sorter: Option<&[usize]>,
    is_nan: &(dyn Fn(&T) -> bool + Send + Sync),
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: ExtSortCmp + Send + Sync,
    D2: DimAPI,
{
    if pool.is_none() || l2.size() < PARALLEL_SWITCH {
        return crate::cpu_serial::searching::searchsorted_cpu_serial(
            c, layout_c, x1, l1, x2, l2, side_left, sorter, is_nan,
        );
    }

    rstsr_assert_eq!(l1.ndim(), 1, InvalidLayout, "searchsorted requires a one-dimensional x1.")?;
    let n = l1.shape()[0];
    let stride1: isize = l1.stride()[0];
    let base1 = l1.offset() as isize;
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&l2.to_dim()?, RowMajor)?;
    let stride_ref: &[isize] = layout_c.stride().as_ref();
    let out_strides: Vec<usize> = stride_ref.iter().map(|&s| s.unsigned_abs()).collect();

    // base-pointer hoist through AtomicPtr (house pattern; disjoint writes:
    // each value writes exactly one output position)
    let c_ptr = AtomicPtr::new(c.as_mut_ptr());

    let task = || {
        iter.into_par_iter().try_for_each(|(index, off)| -> Result<()> {
            let index_ref: &[usize] = index.as_ref();
            let mut out_pos = 0_usize;
            for (i, &v) in index_ref.iter().enumerate() {
                out_pos += v * out_strides[i];
            }
            let at = |i: usize| -> &T {
                let j = match sorter {
                    Some(perm) => base1 + stride1 * perm[i] as isize,
                    None => base1 + stride1 * i as isize,
                };
                &x1[j as usize]
            };
            let pos = searchsorted_value_cpu_serial(x1, n, &x2[off], side_left, &at, is_nan);
            // SAFETY: base pointer hoisted through AtomicPtr (relaxed load;
            // never reassigned); `out_pos` values are a bijection over the
            // output (each value's multi-index maps to one position).
            unsafe {
                c_ptr.load(Ordering::Relaxed).add(out_pos).write(MaybeUninit::new(pos));
            }
            Ok(())
        })
    };
    pool.expect("pool checked Some above").install(task)
}
