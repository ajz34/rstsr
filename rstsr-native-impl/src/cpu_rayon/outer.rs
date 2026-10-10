use crate::prelude_dev::*;
use core::ops::Mul;

const PARALLEL_SWITCH: usize = 512;

/// Rayon twin of [`outer_naive_cpu_serial`]; below [`PARALLEL_SWITCH`] elements
/// or without a pool it delegates to the serial kernel.
pub fn outer_naive_cpu_rayon<TA, TB, TC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix1>,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Send + Sync,
    TA: Mul<TB, Output = TC>,
{
    if la.size().saturating_mul(lb.size()) < PARALLEL_SWITCH || pool.is_none() {
        return outer_naive_cpu_serial(c, lc, a, la, b, lb);
    }

    let (n, m) = (la.shape()[0], lb.shape()[0]);
    rstsr_assert_eq!(lc.shape(), &[n, m], InvalidLayout, "the outer-product output should have shape (a.len, b.len)")?;
    let lam = Layout::new([n, m], [la.stride()[0], 0], la.offset())?;
    let lbm = Layout::new([n, m], [0, lb.stride()[0]], lb.offset())?;
    let layouts = translate_to_col_major(&[lc, &lam, &lbm], TensorIterOrder::K)?;
    let (lc, lam, lbm) = (&layouts[0], &layouts[1], &layouts[2]);

    // pass the mutable output pointer into the parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        layout_col_major_dim_dispatch_par_3(lc, lam, lbm, |(idx_c, idx_a, idx_b)| {
            // SAFETY: `thr_c` hoists `c`'s base pointer (relaxed load; `c` is never
            // reassigned through it) and each task writes the disjoint output
            // position `idx_c`.
            unsafe {
                let c_ptr = thr_c.load(Ordering::Relaxed).add(idx_c);
                (*c_ptr).write(a[idx_a].clone() * b[idx_b].clone());
            }
        })
    };
    pool.map_or_else(task, |pool| pool.install(task))
}
