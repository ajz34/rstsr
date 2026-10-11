use crate::prelude_dev::*;
use core::ops::{Mul, Sub};
use rstsr_dtype_traits::DTypePromoteAPI;

const PARALLEL_SWITCH: usize = 512;

/// Rayon twin of [`cross_ext_naive_cpu_serial`]; below [`PARALLEL_SWITCH`]
/// output elements or without a pool it delegates to the serial kernel.
pub fn cross_ext_naive_cpu_rayon<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    axis: usize,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    D: DimAPI,
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Mul<TC, Output = TC> + Sub<TC, Output = TC>,
    TA: DTypePromoteAPI<TB, Res = TC>,
{
    if lc.size() < PARALLEL_SWITCH || pool.is_none() {
        return cross_ext_naive_cpu_serial(c, lc, a, la, b, lb, axis);
    }

    let (sa, sb, sc) = (la.stride()[axis], lb.stride()[axis], lc.stride()[axis]);
    let singleton = |layout: &Layout<D>| -> Result<Layout<D>> {
        let mut shape = layout.shape().clone();
        let mut stride = layout.stride().clone();
        shape[axis] = 1;
        stride[axis] = 0;
        Layout::new(shape, stride, layout.offset())
    };
    let lac = singleton(la)?;
    let lbc = singleton(lb)?;
    let lcc = singleton(lc)?;
    let layouts = translate_to_col_major(&[&lcc, &lac, &lbc], TensorIterOrder::K)?;
    let (lcc, lac, lbc) = (&layouts[0], &layouts[1], &layouts[2]);

    let at = |base: usize, s: isize, k: isize| (base as isize + k * s) as usize;
    let pair = |ia: usize, ib: usize| a[ia].clone().promote_pair(b[ib].clone());
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        layout_col_major_dim_dispatch_par_3(lcc, lac, lbc, |(ic, ia, ib)| {
            let (a0, a1, a2) = (at(ia, sa, 0), at(ia, sa, 1), at(ia, sa, 2));
            let (b0, b1, b2) = (at(ib, sb, 0), at(ib, sb, 1), at(ib, sb, 2));
            let (c0, c1, c2) = (at(ic, sc, 0), at(ic, sc, 1), at(ic, sc, 2));
            let (a1p, b2p) = pair(a1, b2);
            let (a2p, b1p) = pair(a2, b1);
            let (a2q, b0q) = pair(a2, b0);
            let (a0q, b2q) = pair(a0, b2);
            let (a0r, b1r) = pair(a0, b1);
            let (a1r, b0r) = pair(a1, b0);
            // SAFETY: `thr_c` hoists `c`'s base pointer (relaxed load; `c` is never
            // reassigned through it) and each task writes its own disjoint output
            // positions `c0`/`c1`/`c2`.
            unsafe {
                let base = thr_c.load(Ordering::Relaxed);
                (*base.add(c0)).write(a1p * b2p - a2p * b1p);
                (*base.add(c1)).write(a2q * b0q - a0q * b2q);
                (*base.add(c2)).write(a0r * b1r - a1r * b0r);
            }
        })
    };
    pool.map_or_else(task, |pool| pool.install(task))
}
