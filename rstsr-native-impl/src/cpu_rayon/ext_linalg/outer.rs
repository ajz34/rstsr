use crate::prelude_dev::*;
use core::ops::Mul;
use rstsr_dtype_traits::DTypePromoteAPI;

const PARALLEL_SWITCH: usize = 512;

/// Rayon twin of [`outer_ext_naive_cpu_serial`]; below [`PARALLEL_SWITCH`]
/// elements or without a pool it delegates to the serial kernel.
pub fn outer_ext_naive_cpu_rayon<TA, TB, TC>(
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
    TC: Clone + Send + Sync + Mul<TC, Output = TC>,
    TA: DTypePromoteAPI<TB, Res = TC>,
{
    if la.size().saturating_mul(lb.size()) < PARALLEL_SWITCH || pool.is_none() {
        return outer_ext_naive_cpu_serial(c, lc, a, la, b, lb);
    }

    let (n, m) = (la.shape()[0], lb.shape()[0]);
    rstsr_assert_eq!(lc.shape(), &[n, m], InvalidLayout, "the outer-product output should have shape (a.len, b.len)")?;
    let lam = Layout::new([n, m], [la.stride()[0], 0], la.offset())?;
    let lbm = Layout::new([n, m], [0, lb.stride()[0]], lb.offset())?;
    let layouts = translate_to_col_major(&[lc, &lam, &lbm], TensorIterOrder::K)?;
    let (lc, lam, lbm) = (&layouts[0], &layouts[1], &layouts[2]);

    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        layout_col_major_dim_dispatch_par_3(lc, lam, lbm, |(idx_c, idx_a, idx_b)| {
            // SAFETY: `thr_c` hoists `c`'s base pointer (relaxed load; `c` is never
            // reassigned through it) and each task writes the disjoint output
            // position `idx_c`.
            unsafe {
                let c_ptr = thr_c.load(Ordering::Relaxed).add(idx_c);
                let (x, y) = a[idx_a].clone().promote_pair(b[idx_b].clone());
                (*c_ptr).write(x * y);
            }
        })
    };
    pool.map_or_else(task, |pool| pool.install(task))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parallel branch of the promoting twin (past `PARALLEL_SWITCH`) must
    /// agree with its serial kernel, and carry the promoted values.
    #[test]
    fn test_ext_parallel_branch_matches_serial() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let (n, m) = (32usize, 32usize);
        let la = Layout::<Ix1>::new([n], [1isize], 0).unwrap();
        let lb = Layout::<Ix1>::new([m], [1isize], 0).unwrap();
        let lc = Layout::<Ix2>::new([n, m], [m as isize, 1], 0).unwrap();
        let a: Vec<u8> = (0..n).map(|i| (i % 7) as u8).collect();
        let b: Vec<u16> = (0..m).map(|i| (i % 5) as u16 + 1).collect();

        let mut c_par = vec![MaybeUninit::<u16>::uninit(); n * m];
        let mut c_ser = vec![MaybeUninit::<u16>::uninit(); n * m];
        outer_ext_naive_cpu_rayon(&mut c_par, &lc, &a, &la, &b, &lb, Some(&pool)).unwrap();
        outer_ext_naive_cpu_serial(&mut c_ser, &lc, &a, &la, &b, &lb).unwrap();

        let c_par: Vec<u16> = c_par.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        let c_ser: Vec<u16> = c_ser.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        assert_eq!(c_par, c_ser);
        // promoted u8 x u16 -> u16
        assert_eq!(c_par[0], (a[0] as u16) * b[0]);
        assert_eq!(c_par[n * m - 1], (a[n - 1] as u16) * b[m - 1]);
    }
}
