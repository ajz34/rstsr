use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;
use rstsr_dtype_traits::DTypePromoteAPI;

/// Promoting twin of [`tensordot_naive_cpu_serial`]: `a` and `b` may have
/// different dtypes, each pair promoted to `TC` (= `TA::Res`) before the
/// product.
pub fn tensordot_ext_naive_cpu_serial<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    axes_a: &[isize],
    axes_b: &[isize],
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero + Mul<TC, Output = TC>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
{
    let (las, lam, lbs, lbm) = split_tensordot_axes(la, axes_a, lb, axes_b)?;

    let lc = lc.to_dim::<IxD>()?;
    let offset_a = la.offset();
    let offset_b = lb.offset();

    // Free-axis layouts expanded onto the output shape: the block belonging to
    // the *other* operand gets zero stride (a broadcast view). Only read, so
    // aliasing is sound.
    let shape_c = lc.shape().clone();
    let stride_a = lam.stride().iter().copied().chain(core::iter::repeat_n(0isize, lbm.ndim())).collect_vec();
    let stride_b = core::iter::repeat_n(0isize, lam.ndim()).chain(lbm.stride().iter().copied()).collect_vec();
    // SAFETY: both are valid (read-only) broadcast views of real layouts.
    let lam_e = unsafe { Layout::<IxD>::new_unchecked(shape_c.clone(), stride_a, offset_a) };
    let lbm_e = unsafe { Layout::<IxD>::new_unchecked(shape_c, stride_b, offset_b) };

    layout_col_major_dim_dispatch_3(&lc, &lam_e, &lbm_e, |(idx_c, idx_ma, idx_mb)| {
        let mut val_c: TC = Zero::zero();
        let stat = layout_col_major_dim_dispatch_2(&las, &lbs, |(idx_sa, idx_sb)| {
            // `idx_m*` and `idx_s*` each carry the operand offset already; remove
            // the double count once.
            let ia = idx_ma + idx_sa - offset_a;
            let ib = idx_mb + idx_sb - offset_b;
            let (x, y) = a[ia].clone().promote_pair(b[ib].clone());
            val_c = val_c.clone() + x * y;
        });
        stat.rstsr_unwrap();
        c[idx_c].write(val_c);
    })
}
