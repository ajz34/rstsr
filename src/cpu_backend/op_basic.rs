//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;
use core::ops::IndexMut;
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

impl<T, DC, DA> OpAssignAPI<T, DC, DA> for CpuDevice
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary_layout(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()> {
        if lc.c_contig() && la.c_contig() || lc.f_contig() && la.f_contig() {
            let offset_c = lc.offset();
            let offset_a = la.offset();
            let size = lc.size();
            for i in 0..size {
                c.rawvec[offset_c + i] = a.rawvec[offset_a + i].clone();
            }
        } else {
            let order = {
                if lc.c_prefer() && la.c_prefer() {
                    TensorIterOrder::C
                } else if lc.f_prefer() && la.f_prefer() {
                    TensorIterOrder::F
                } else {
                    match TensorOrder::default() {
                        TensorOrder::C => TensorIterOrder::C,
                        TensorOrder::F => TensorIterOrder::F,
                    }
                }
            };
            let lc = translate_to_col_major_unary(lc, order)?;
            let la = translate_to_col_major_unary(la, order)?;
            let iter_c = IterLayoutColMajor::new(&lc)?;
            let iter_a = IterLayoutColMajor::new(&la)?;
            for (idx_c, idx_a) in izip!(iter_c, iter_a) {
                c.rawvec[idx_c] = a.rawvec[idx_a].clone();
            }
        }
        return Ok(());
    }
}

/* #region ternary-op */

macro_rules! impl_op_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_ternary:ident, $op:ident) => {
        impl<T, TB, D> $DeviceOpAPI<T, TB, D> for CpuDevice
        where
            TB: Clone,
            T: core::ops::$Op<TB, Output = T> + Clone,
            D: DimAPI,
        {
            fn $op_ternary(
                &self,
                c: &mut Storage<T, CpuDevice>,
                lc: &Layout<D>,
                a: &Storage<T, CpuDevice>,
                la: &Layout<D>,
                b: &Storage<TB, CpuDevice>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
                let layouts_full_ref = layouts_full.iter().collect_vec();
                let (layouts_contig, size_contig) =
                    translate_to_col_major_with_contig(&layouts_full_ref);

                if size_contig >= CONTIG_SWITCH {
                    let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
                    let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
                    let iter_b = IterLayoutColMajor::new(&layouts_contig[2])?;
                    for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
                        for i in 0..size_contig {
                            c.rawvec[idx_c + i] = core::ops::$Op::$op(
                                a.rawvec[idx_a + i].clone(),
                                b.rawvec[idx_b + i].clone(),
                            );
                        }
                    }
                } else {
                    let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
                    let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
                    let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
                    for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
                        c.rawvec[idx_c] =
                            core::ops::$Op::$op(a.rawvec[idx_a].clone(), b.rawvec[idx_b].clone());
                    }
                }
                return Ok(());
            }
        }
    };
}

impl_op_api!(DeviceAddAPI, Add, add_ternary, add);
impl_op_api!(DeviceSubAPI, Sub, sub_ternary, sub);
impl_op_api!(DeviceMulAPI, Mul, mul_ternary, mul);
impl_op_api!(DeviceDivAPI, Div, div_ternary, div);
impl_op_api!(DeviceRemAPI, Rem, rem_ternary, rem);
impl_op_api!(DeviceBitOrAPI, BitOr, bitor_ternary, bitor);
impl_op_api!(DeviceBitAndAPI, BitAnd, bitand_ternary, bitand);
impl_op_api!(DeviceBitXorAPI, BitXor, bitxor_ternary, bitxor);
impl_op_api!(DeviceShlAPI, Shl, shl_ternary, shl);
impl_op_api!(DeviceShrAPI, Shr, shr_ternary, shr);

/* #endregion */

/* #region binary-op */

macro_rules! impl_op_assign_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_binary:ident, $op:ident) => {
        impl<T, TB, D> $DeviceOpAPI<T, TB, D> for CpuDevice
        where
            T: core::ops::$Op<TB> + Clone,
            TB: Clone,
            D: DimAPI,
        {
            fn $op_binary(
                &self,
                c: &mut Storage<T, CpuDevice>,
                lc: &Layout<D>,
                b: &Storage<TB, CpuDevice>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
                let layouts_full_ref = layouts_full.iter().collect_vec();
                let (layouts_contig, size_contig) =
                    translate_to_col_major_with_contig(&layouts_full_ref);

                if size_contig >= CONTIG_SWITCH {
                    let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
                    let iter_b = IterLayoutColMajor::new(&layouts_contig[1])?;
                    for (idx_c, idx_b) in izip!(iter_c, iter_b) {
                        for i in 0..size_contig {
                            core::ops::$Op::$op(
                                c.rawvec.index_mut(idx_c + i),
                                b.rawvec[idx_b + i].clone(),
                            );
                        }
                    }
                } else {
                    let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
                    let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
                    for (idx_c, idx_b) in izip!(iter_c, iter_b) {
                        core::ops::$Op::$op(c.rawvec.index_mut(idx_c), b.rawvec[idx_b].clone());
                    }
                }
                return Ok(());
            }
        }
    };
}

impl_op_assign_api!(DeviceAddAssignAPI, AddAssign, add_assign_binary, add_assign);
impl_op_assign_api!(DeviceSubAssignAPI, SubAssign, sub_assign_binary, sub_assign);
impl_op_assign_api!(DeviceMulAssignAPI, MulAssign, mul_assign_binary, mul_assign);
impl_op_assign_api!(DeviceDivAssignAPI, DivAssign, div_assign_binary, div_assign);
impl_op_assign_api!(DeviceRemAssignAPI, RemAssign, rem_assign_binary, rem_assign);
impl_op_assign_api!(DeviceBitOrAssignAPI, BitOrAssign, bitor_assign_binary, bitor_assign);
impl_op_assign_api!(DeviceBitAndAssignAPI, BitAndAssign, bitand_assign_binary, bitand_assign);
impl_op_assign_api!(DeviceBitXorAssignAPI, BitXorAssign, bitxor_assign_binary, bitxor_assign);
impl_op_assign_api!(DeviceShlAssignAPI, ShlAssign, shl_assign_binary, shl_assign);
impl_op_assign_api!(DeviceShrAssignAPI, ShrAssign, shr_assign_binary, shr_assign);

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
