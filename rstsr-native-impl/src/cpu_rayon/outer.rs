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

#[cfg(test)]
mod tests {
    use super::*;

    /// The parallel branch (`n * m` past `PARALLEL_SWITCH`) must agree with the
    /// serial kernel element-for-element.
    #[test]
    fn test_parallel_branch_matches_serial() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let (n, m) = (32usize, 32usize);
        let la = Layout::<Ix1>::new([n], [1isize], 0).unwrap();
        let lb = Layout::<Ix1>::new([m], [1isize], 0).unwrap();
        let lc = Layout::<Ix2>::new([n, m], [m as isize, 1], 0).unwrap();
        let a: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..m).map(|i| (i % 5) as f64 + 1.0).collect();

        let mut c_par = vec![MaybeUninit::<f64>::uninit(); n * m];
        let mut c_ser = vec![MaybeUninit::<f64>::uninit(); n * m];
        outer_naive_cpu_rayon(&mut c_par, &lc, &a, &la, &b, &lb, Some(&pool)).unwrap();
        outer_naive_cpu_serial(&mut c_ser, &lc, &a, &la, &b, &lb).unwrap();

        let c_par: Vec<f64> = c_par.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        let c_ser: Vec<f64> = c_ser.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        assert_eq!(c_par, c_ser);
    }
}
