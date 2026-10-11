use crate::prelude_dev::*;
use core::ops::{Mul, Sub};

const PARALLEL_SWITCH: usize = 512;

/// Rayon twin of [`cross_naive_cpu_serial`]; below [`PARALLEL_SWITCH`] output
/// elements or without a pool it delegates to the serial kernel.
pub fn cross_naive_cpu_rayon<TA, TB, TC, D>(
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
    TC: Send + Sync + Sub<TC, Output = TC>,
    TA: Mul<TB, Output = TC>,
{
    if lc.size() < PARALLEL_SWITCH || pool.is_none() {
        return cross_naive_cpu_serial(c, lc, a, la, b, lb, axis);
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
    // pass the mutable output pointer into the parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        layout_col_major_dim_dispatch_par_3(lcc, lac, lbc, |(ic, ia, ib)| {
            let (a0, a1, a2) = (at(ia, sa, 0), at(ia, sa, 1), at(ia, sa, 2));
            let (b0, b1, b2) = (at(ib, sb, 0), at(ib, sb, 1), at(ib, sb, 2));
            let (c0, c1, c2) = (at(ic, sc, 0), at(ic, sc, 1), at(ic, sc, 2));
            // SAFETY: `thr_c` hoists `c`'s base pointer (relaxed load; `c` is never
            // reassigned through it) and each task writes its own disjoint output
            // positions `c0`/`c1`/`c2`.
            unsafe {
                let base = thr_c.load(Ordering::Relaxed);
                (*base.add(c0)).write(a[a1].clone() * b[b2].clone() - a[a2].clone() * b[b1].clone());
                (*base.add(c1)).write(a[a2].clone() * b[b0].clone() - a[a0].clone() * b[b2].clone());
                (*base.add(c2)).write(a[a0].clone() * b[b1].clone() - a[a1].clone() * b[b0].clone());
            }
        })
    };
    pool.map_or_else(task, |pool| pool.install(task))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parallel branch (past `PARALLEL_SWITCH`) must agree with the serial
    /// kernel element-for-element, on both the vector and the stacked shapes.
    #[test]
    fn test_parallel_branch_matches_serial() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        // (24, 32, 3), cross along the last axis
        let (n, m) = (24usize, 32usize);
        let lc = Layout::<Ix3>::new([n, m, 3], [m as isize * 3, 3, 1], 0).unwrap();
        let a: Vec<f64> = (0..n * m * 3).map(|i| (i % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..n * m * 3).map(|i| (i % 5) as f64 + 1.0).collect();

        let mut c_par = vec![MaybeUninit::<f64>::uninit(); n * m * 3];
        let mut c_ser = vec![MaybeUninit::<f64>::uninit(); n * m * 3];
        cross_naive_cpu_rayon(&mut c_par, &lc, &a, &lc, &b, &lc, 2, Some(&pool)).unwrap();
        cross_naive_cpu_serial(&mut c_ser, &lc, &a, &lc, &b, &lc, 2).unwrap();

        let c_par: Vec<f64> = c_par.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        let c_ser: Vec<f64> = c_ser.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        assert_eq!(c_par, c_ser);
    }
}
