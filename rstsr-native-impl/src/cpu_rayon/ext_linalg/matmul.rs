//! Matrix multiplication with dtype promotion for the CPU-rayon backend.
//!
//! **This implementation is not optimized!**
//!
//! Rayon twin of `cpu_serial::matmul_ext`: the operands are promoted to their
//! common type element-wise ([`DTypePromoteAPI`]) inside the kernel. It is the
//! non-faer-dtype fallback of `DeviceExtMatMulAPI::ext_matmul_uninit`.

use crate::prelude_dev::*;
use core::ops::{Add, Mul};
use num::Zero;
use rstsr_dtype_traits::DTypePromoteAPI;

#[allow(clippy::too_many_arguments)]
pub fn matmul_ext_naive_uninit_cpu_rayon<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    alpha: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    // NOTE: this only works for row-major layout; the column-major caller
    // reverses every axis and swaps the operands (C = A * B  =>  C^T = B^T *
    // A^T).
    match (la.ndim(), lb.ndim(), lc.ndim()) {
        (1, 1, 0) => {
            // rule 1: vector inner dot
            let la = &la.clone().into_dim::<Ix1>().unwrap();
            let lb = &lb.clone().into_dim::<Ix1>().unwrap();
            let lc = &lc.clone().into_dim::<Ix0>().unwrap();
            return inner_dot_ext_naive_uninit_cpu_rayon(c, lc, a, la, b, lb, alpha, pool);
        },
        (2, 2, 2) => {
            // rule 2: matrix multiplication
            let la = &la.clone().into_dim::<Ix2>().unwrap();
            let lb = &lb.clone().into_dim::<Ix2>().unwrap();
            let lc = &lc.clone().into_dim::<Ix2>().unwrap();
            return gemm_ix2_ext_naive_uninit_cpu_rayon(c, lc, a, la, b, lb, alpha, pool);
        },
        _ => (),
    }

    // broadcasted rules 3..7: the config resolves the rule, broadcasts the
    // batch (`rest`) dims against `lc`, and hands us 2-D matmul layouts (via
    // `dim_insert` for the vector rules). Sequential outer, parallel gemm.
    let cfg = layout_matmul_dyn_row_major_with_lc(&la.to_dim()?, &lb.to_dim()?, &lc.to_dim()?)?;
    let la_matmul = cfg.la_matmul.into_dim::<Ix2>()?;
    let lb_matmul = cfg.lb_matmul.into_dim::<Ix2>()?;
    let lc_matmul = cfg.lc_matmul.into_dim::<Ix2>()?;
    let la_rest = cfg.la_rest.unwrap();
    let lb_rest = cfg.lb_rest.unwrap();
    let lc_rest = cfg.lc_rest.unwrap();
    let ita_rest = IterLayoutColMajor::new(&la_rest)?;
    let itb_rest = IterLayoutColMajor::new(&lb_rest)?;
    let itc_rest = IterLayoutColMajor::new(&lc_rest)?;
    for (ia_rest, ib_rest, ic_rest) in izip!(ita_rest, itb_rest, itc_rest) {
        let mut la_m = la_matmul.clone();
        let mut lb_m = lb_matmul.clone();
        let mut lc_m = lc_matmul.clone();
        unsafe {
            // SAFETY: offsets come from the rest-layout iterators of the validated matmul
            // config; sub-layout + offset addresses only in-bounds elements.
            la_m.set_offset(ia_rest);
            lb_m.set_offset(ib_rest);
            lc_m.set_offset(ic_rest);
        }
        gemm_ix2_ext_naive_uninit_cpu_rayon(c, &lc_m, a, &la_m, b, &lb_m, alpha.clone(), pool)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_ix2_ext_naive_uninit_cpu_rayon<TA, TB, TC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB, Res = TC>,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
{
    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;
    let (m, n, k) = (sc[0], sc[1], sa[1]);

    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        (0..n).into_par_iter().for_each(|j| {
            (0..m).into_par_iter().for_each(|i| unsafe {
                // SAFETY: `ptr_c` is `c`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `c` is never reassigned through it). Each (i, j) is
                // written by exactly one parallel task, via `write` — the slot is
                // never read.
                let ptr_c = thr_c.load(Ordering::Relaxed).offset(lc.index_uncheck(&[i, j]));
                let dot = (0..k).fold(TC::zero(), |acc, p| {
                    let val_a = a[la.index_uncheck(&[i, p]) as usize].clone();
                    let val_b = b[lb.index_uncheck(&[p, j]) as usize].clone();
                    let (val_a, val_b) = TA::promote_pair(val_a, val_b);
                    acc + alpha.clone() * (val_a * val_b)
                });
                ptr_c.write(MaybeUninit::new(dot));
            });
        });
    };

    pool.map_or_else(task, |pool| pool.install(task));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn inner_dot_ext_naive_uninit_cpu_rayon<TA, TB, TC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<Ix0>,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix1>,
    alpha: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB, Res = TC>,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
{
    // shape check
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sa[0], sb[0], InvalidLayout)?;
    let n = sa[0];

    let task = || {
        (0..n)
            .into_par_iter()
            .fold(
                || TC::zero(),
                |acc, i| {
                    let val_a = a[la.index_uncheck(&[i]) as usize].clone();
                    let val_b = b[lb.index_uncheck(&[i]) as usize].clone();
                    let (val_a, val_b) = TA::promote_pair(val_a, val_b);
                    acc + val_a * val_b
                },
            )
            .reduce_with(|a, b| a + b)
            .unwrap_or(TC::zero())
    };
    let c_innerdot = pool.map_or_else(task, |pool| pool.install(task));
    c[lc.offset()].write(c_innerdot * alpha);
    Ok(())
}
