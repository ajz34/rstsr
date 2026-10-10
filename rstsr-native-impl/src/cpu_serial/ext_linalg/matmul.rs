//! Matrix multiplication with dtype promotion for the CPU-serial backend.
//!
//! **This implementation is not optimized!**
//!
//! These kernels mirror `matmul_naive::*_uninit_*`, but the operands may have
//! different dtypes: the pair is promoted to its common type element-wise
//! ([`DTypePromoteAPI`]) and the product is accumulated in that type. They are
//! the write-only path behind `DeviceExtMatMulAPI::ext_matmul_uninit`. For
//! same-dtype operands promotion is the identity, so the results agree with the
//! plain matmul kernels.

use crate::prelude_dev::*;
use core::ops::{Add, Mul};
use num::Zero;
use rstsr_dtype_traits::DTypePromoteAPI;

#[allow(clippy::too_many_arguments)]
pub fn matmul_ext_naive_uninit_cpu_serial<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    alpha: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    // NOTE: this only works for row-major layout. For column-major layout the
    // caller reverses every axis and swaps the operands (C = A * B  =>
    // C^T = B^T * A^T), which is why the device impl requires both promotion
    // directions.
    match (la.ndim(), lb.ndim(), lc.ndim()) {
        (1, 1, 0) => {
            // rule 1: vector inner dot
            let la = &la.clone().into_dim::<Ix1>().unwrap();
            let lb = &lb.clone().into_dim::<Ix1>().unwrap();
            let lc = &lc.clone().into_dim::<Ix0>().unwrap();
            inner_dot_ext_naive_uninit_cpu_serial(c, lc, a, la, b, lb, alpha)?;
        },
        (2, 2, 2) => {
            // rule 2: matrix multiplication
            let la = &la.clone().into_dim::<Ix2>().unwrap();
            let lb = &lb.clone().into_dim::<Ix2>().unwrap();
            let lc = &lc.clone().into_dim::<Ix2>().unwrap();
            gemm_ext_naive_uninit_cpu_serial(c, lc, a, la, b, lb, alpha)?;
        },
        _ => {
            // broadcasted rules 3..7: the config resolves the rule, broadcasts
            // the batch (`rest`) dims against `lc`, and hands us 2-D matmul
            // layouts (via `dim_insert` for the vector rules).
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
                gemm_ext_naive_uninit_cpu_serial(c, &lc_m, a, &la_m, b, &lb_m, alpha.clone())?;
            }
        },
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_ext_naive_uninit_cpu_serial<TA, TB, TC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;
    let (m, n, k) = (sc[0], sc[1], sa[1]);

    // naive iteration: assuming c-prefer; each slot is written exactly once
    for i_m in 0..m {
        for i_n in 0..n {
            let idx_c = lc.index_uncheck(&[i_m, i_n]) as usize;
            let mut sum = TC::zero();
            for i_k in 0..k {
                let idx_a = la.index_uncheck(&[i_m, i_k]) as usize;
                let idx_b = lb.index_uncheck(&[i_k, i_n]) as usize;
                let (val_a, val_b) = TA::promote_pair(a[idx_a].clone(), b[idx_b].clone());
                sum = sum + alpha.clone() * (val_a * val_b);
            }
            c[idx_c].write(sum);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn inner_dot_ext_naive_uninit_cpu_serial<TA, TB, TC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<Ix0>,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix1>,
    alpha: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    // shape check
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sa[0], sb[0], InvalidLayout)?;
    let n = sa[0];

    // naive iteration; the single slot is written exactly once
    let mut sum = TC::zero();
    for i in 0..n {
        let idx_a = la.index_uncheck(&[i]) as usize;
        let idx_b = lb.index_uncheck(&[i]) as usize;
        let (val_a, val_b) = TA::promote_pair(a[idx_a].clone(), b[idx_b].clone());
        sum = sum + alpha.clone() * (val_a * val_b);
    }
    c[lc.offset()].write(sum);
    Ok(())
}
