//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;
use num::Zero;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

/// Fold over the manually unrolled `xs` with `f`
///
/// # See also
///
/// This code is from <https://github.com/rust-ndarray/ndarray/blob/master/src/numeric_util.rs>
pub fn unrolled_fold<A, I, F>(mut xs: &[A], init: I, f: F) -> A
where
    A: Clone,
    I: Fn() -> A,
    F: Fn(A, A) -> A,
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(), init(), init(), init(), init());
    while xs.len() >= 8 {
        p0 = f(p0, xs[0].clone());
        p1 = f(p1, xs[1].clone());
        p2 = f(p2, xs[2].clone());
        p3 = f(p3, xs[3].clone());
        p4 = f(p4, xs[4].clone());
        p5 = f(p5, xs[5].clone());
        p6 = f(p6, xs[6].clone());
        p7 = f(p7, xs[7].clone());

        xs = &xs[8..];
    }
    acc = f(acc.clone(), f(p0, p4));
    acc = f(acc.clone(), f(p1, p5));
    acc = f(acc.clone(), f(p2, p6));
    acc = f(acc.clone(), f(p3, p7));

    // make it clear to the optimizer that this loop is short
    // and can not be autovectorized.
    for (i, x) in xs.iter().enumerate() {
        if i >= 7 {
            break;
        }
        acc = f(acc.clone(), x.clone())
    }
    acc
}

/* #region op_func */

impl<TA, TB, TC, D, F> DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D, F> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB),
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut Storage<TC, CpuDevice>,
        lc: &Layout<D>,
        a: &Storage<TA, CpuDevice>,
        la: &Layout<D>,
        b: &Storage<TB, CpuDevice>,
        lb: &Layout<D>,
        mut f: F,
    ) -> Result<()> {
        // re-align layouts
        let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
        let layouts_full_ref = layouts_full.iter().collect_vec();
        let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

        // contiguous iteration if possible, otherwise use iterator of layout
        if size_contig >= CONTIG_SWITCH {
            let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
            let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
            let iter_b = IterLayoutColMajor::new(&layouts_contig[2])?;
            for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
                for i in 0..size_contig {
                    f(&mut c.rawvec[idx_c + i], &a.rawvec[idx_a + i], &b.rawvec[idx_b + i]);
                }
            }
        } else {
            let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
            let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
            let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
            for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
                f(&mut c.rawvec[idx_c], &a.rawvec[idx_a], &b.rawvec[idx_b]);
            }
        }
        return Ok(());
    }
}

impl<TA, TB, D, F> DeviceOp_MutA_RefB_API<TA, TB, D, F> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    D: DimAPI,
    F: FnMut(&mut TA, &TB),
{
    fn op_muta_refb_func(
        &self,
        a: &mut Storage<TA, CpuDevice>,
        la: &Layout<D>,
        b: &Storage<TB, CpuDevice>,
        lb: &Layout<D>,
        mut f: F,
    ) -> Result<()> {
        // re-align layouts
        let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
        let layouts_full_ref = layouts_full.iter().collect_vec();
        let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

        // contiguous iteration if possible, otherwise use iterator of layout
        if size_contig >= CONTIG_SWITCH {
            let iter_a = IterLayoutColMajor::new(&layouts_contig[0])?;
            let iter_b = IterLayoutColMajor::new(&layouts_contig[1])?;
            for (idx_a, idx_b) in izip!(iter_a, iter_b) {
                for i in 0..size_contig {
                    f(&mut a.rawvec[idx_a + i], &b.rawvec[idx_b + i]);
                }
            }
        } else {
            let iter_a = IterLayoutColMajor::new(&layouts_full[0])?;
            let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
            for (idx_a, idx_b) in izip!(iter_a, iter_b) {
                f(&mut a.rawvec[idx_a], &b.rawvec[idx_b]);
            }
        }
        return Ok(());
    }
}

impl<T, D, F> DeviceOp_MutA_API<T, D, F> for CpuDevice
where
    T: Clone,
    D: DimAPI,
    F: FnMut(&mut T),
{
    fn op_muta_func(&self, a: &mut Storage<T, CpuDevice>, la: &Layout<D>, mut f: F) -> Result<()> {
        let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
        let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

        if size_contig >= CONTIG_SWITCH {
            let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
            for idx_a in iter_a {
                for i in 0..size_contig {
                    f(&mut a.rawvec[idx_a + i]);
                }
            }
        } else {
            let iter_a = IterLayoutColMajor::new(&layout)?;
            for idx_a in iter_a {
                f(&mut a.rawvec[idx_a]);
            }
        }
        return Ok(());
    }
}

/* #region op_mutc_refa_refb_operator */

macro_rules! impl_op_mutc_refa_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident, $op_mutc_refa_refb_func:ident, $func:expr) => {
        impl<TA, TB, TC, D> $DeviceOpAPI<TA, TB, TC, D> for CpuDevice
        where
            TA: Clone + core::ops::$Op<TB, Output = TC>,
            TB: Clone,
            TC: Clone,
            D: DimAPI,
        {
            fn $op_mutc_refa_refb_func(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_mutc_refa_refb_operator {
    use super::*;
    impl_op_mutc_refa_refb_operator!(DeviceAddAPI   , Add   , op_mutc_refa_refb_add   , |c, a, b| *c = a.clone() +  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceSubAPI   , Sub   , op_mutc_refa_refb_sub   , |c, a, b| *c = a.clone() -  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceMulAPI   , Mul   , op_mutc_refa_refb_mul   , |c, a, b| *c = a.clone() *  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceDivAPI   , Div   , op_mutc_refa_refb_div   , |c, a, b| *c = a.clone() /  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceRemAPI   , Rem   , op_mutc_refa_refb_rem   , |c, a, b| *c = a.clone() %  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceBitOrAPI , BitOr , op_mutc_refa_refb_bitor , |c, a, b| *c = a.clone() |  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceBitAndAPI, BitAnd, op_mutc_refa_refb_bitand, |c, a, b| *c = a.clone() &  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceBitXorAPI, BitXor, op_mutc_refa_refb_bitxor, |c, a, b| *c = a.clone() ^  b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceShlAPI   , Shl   , op_mutc_refa_refb_shl   , |c, a, b| *c = a.clone() << b.clone());
    impl_op_mutc_refa_refb_operator!(DeviceShrAPI   , Shr   , op_mutc_refa_refb_shr   , |c, a, b| *c = a.clone() >> b.clone());
}

/* #endregion */

/* #region op_muta_refb_operator */

macro_rules! impl_op_muta_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident, $op_muta_refb_func:ident, $func:expr) => {
        impl<TA, TB, D> $DeviceOpAPI<TA, TB, D> for CpuDevice
        where
            TA: Clone + core::ops::$Op<TB>,
            TB: Clone,
            D: DimAPI,
        {
            fn $op_muta_refb_func(
                &self,
                a: &mut Storage<TA, CpuDevice>,
                la: &Layout<D>,
                b: &Storage<TB, CpuDevice>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_operator {
    use super::*;
    impl_op_muta_refb_operator!(DeviceAddAssignAPI   , AddAssign   , op_muta_refb_add_assign   , |a, b| *a +=  b.clone());
    impl_op_muta_refb_operator!(DeviceSubAssignAPI   , SubAssign   , op_muta_refb_sub_assign   , |a, b| *a -=  b.clone());
    impl_op_muta_refb_operator!(DeviceMulAssignAPI   , MulAssign   , op_muta_refb_mul_assign   , |a, b| *a *=  b.clone());
    impl_op_muta_refb_operator!(DeviceDivAssignAPI   , DivAssign   , op_muta_refb_div_assign   , |a, b| *a /=  b.clone());
    impl_op_muta_refb_operator!(DeviceRemAssignAPI   , RemAssign   , op_muta_refb_rem_assign   , |a, b| *a %=  b.clone());
    impl_op_muta_refb_operator!(DeviceBitOrAssignAPI , BitOrAssign , op_muta_refb_bitor_assign , |a, b| *a |=  b.clone());
    impl_op_muta_refb_operator!(DeviceBitAndAssignAPI, BitAndAssign, op_muta_refb_bitand_assign, |a, b| *a &=  b.clone());
    impl_op_muta_refb_operator!(DeviceBitXorAssignAPI, BitXorAssign, op_muta_refb_bitxor_assign, |a, b| *a ^=  b.clone());
    impl_op_muta_refb_operator!(DeviceShlAssignAPI   , ShlAssign   , op_muta_refb_shl_assign   , |a, b| *a <<= b.clone());
    impl_op_muta_refb_operator!(DeviceShrAssignAPI   , ShrAssign   , op_muta_refb_shr_assign   , |a, b| *a >>= b.clone());
}

macro_rules! impl_op_muta_refb_unary {
    ($DeviceOpAPI:ident, $Op:ident, $op_muta_refb_func:ident, $func:expr) => {
        impl<TA, TB, D> $DeviceOpAPI<TA, TB, D> for CpuDevice
        where
            TA: Clone,
            TB: Clone + core::ops::$Op<Output = TA>,
            D: DimAPI,
        {
            fn $op_muta_refb_func(
                &self,
                a: &mut Storage<TA, CpuDevice>,
                la: &Layout<D>,
                b: &Storage<TB, CpuDevice>,
                lb: &Layout<D>,
            ) -> Result<()> {
                self.op_muta_refb_func(a, la, b, lb, $func)
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_unary {
    use super::*;
    impl_op_muta_refb_unary!(DeviceNegAPI, Neg, op_muta_refb_neg, |a, b| *a = -b.clone());
    impl_op_muta_refb_unary!(DeviceNotAPI, Not, op_muta_refb_not, |a, b| *a = !b.clone());
}

/* #endregion */

impl<T, D> OpSumAPI<T, D> for CpuDevice
where
    T: Zero + core::ops::Add<Output = T> + Clone,
    D: DimAPI,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T> {
        let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
        let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

        if size_contig >= CONTIG_SWITCH {
            let mut sum = T::zero();
            let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
            for idx_a in iter_a {
                let slc = &a.rawvec[idx_a..idx_a + size_contig];
                sum = sum + unrolled_fold(slc, || T::zero(), |acc, x| acc + x.clone());
            }
            return Ok(sum);
        } else {
            let iter_a = IterLayoutColMajor::new(&layout)?;
            let sum = iter_a.fold(T::zero(), |acc, idx| acc + a.rawvec[idx].clone());
            return Ok(sum);
        }
    }
}