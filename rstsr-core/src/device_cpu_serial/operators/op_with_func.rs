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

/* #region op_func definition */

pub fn op_mutc_refa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut TC, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
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
                f(&mut c[idx_c + i], &a[idx_a + i], &b[idx_b + i]);
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
        for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
            f(&mut c[idx_c], &a[idx_a], &b[idx_b]);
        }
    }
    return Ok(());
}

pub fn op_mutc_refa_numb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut TC, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b);
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            f(&mut c[idx_c], &a[idx_a], &b);
        }
    }
    return Ok(());
}

pub fn op_mutc_numa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut TC, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_contig[1])?;
        for (idx_c, idx_b) in izip!(iter_c, iter_b) {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a, &b[idx_b + i]);
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_c, idx_b) in izip!(iter_c, iter_b) {
            f(&mut c[idx_c], &a, &b[idx_b]);
        }
    }
    return Ok(());
}

pub fn op_muta_refb_func_cpu_serial<TA, TB, D>(
    a: &mut [TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
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
                f(&mut a[idx_a + i], &b[idx_b + i]);
            }
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_a, idx_b) in izip!(iter_a, iter_b) {
            f(&mut a[idx_a], &b[idx_b]);
        }
    }
    return Ok(());
}

pub fn op_muta_numb_func_cpu_serial<TA, TB, D>(
    a: &mut [TA],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        for idx_a in iter_a {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b);
            }
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        for idx_a in iter_a {
            f(&mut a[idx_a], &b);
        }
    }
    return Ok(());
}

pub fn op_muta_func_cpu_serial<T, D>(
    a: &mut [T],
    la: &Layout<D>,
    mut f: impl FnMut(&mut T),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        for idx_a in iter_a {
            for i in 0..size_contig {
                f(&mut a[idx_a + i]);
            }
        }
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        for idx_a in iter_a {
            f(&mut a[idx_a]);
        }
    }
    return Ok(());
}

impl<TA, TB, TC, D> DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    D: DimAPI,
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut Storage<TC, DeviceCpuSerial>,
        lc: &Layout<D>,
        a: &Storage<TA, DeviceCpuSerial>,
        la: &Layout<D>,
        b: &Storage<TB, DeviceCpuSerial>,
        lb: &Layout<D>,
        f: impl FnMut(&mut TC, &TA, &TB),
    ) -> Result<()> {
        op_mutc_refa_refb_func_cpu_serial(c.rawvec_mut(), lc, a.rawvec(), la, b.rawvec(), lb, f)
    }
}

impl<TA, TB, TC, D> DeviceOp_MutC_RefA_NumB_API<TA, TB, TC, D> for DeviceCpuSerial
where
    TA: Clone,
    TC: Clone,
    D: DimAPI,
{
    fn op_mutc_refa_numb_func(
        &self,
        c: &mut Storage<TC, DeviceCpuSerial>,
        lc: &Layout<D>,
        a: &Storage<TA, DeviceCpuSerial>,
        la: &Layout<D>,
        b: TB,
        f: impl FnMut(&mut TC, &TA, &TB),
    ) -> Result<()> {
        op_mutc_refa_numb_func_cpu_serial(c.rawvec_mut(), lc, a.rawvec(), la, b, f)
    }
}

impl<TA, TB, TC, D> DeviceOp_MutC_NumA_RefB_API<TA, TB, TC, D> for DeviceCpuSerial
where
    TB: Clone,
    TC: Clone,
    D: DimAPI,
{
    fn op_mutc_numa_refb_func(
        &self,
        c: &mut Storage<TC, DeviceCpuSerial>,
        lc: &Layout<D>,
        a: TA,
        b: &Storage<TB, DeviceCpuSerial>,
        lb: &Layout<D>,
        f: impl FnMut(&mut TC, &TA, &TB),
    ) -> Result<()> {
        op_mutc_numa_refb_func_cpu_serial(c.rawvec_mut(), lc, a, b.rawvec(), lb, f)
    }
}

impl<TA, TB, D> DeviceOp_MutA_RefB_API<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    D: DimAPI,
{
    fn op_muta_refb_func(
        &self,
        a: &mut Storage<TA, DeviceCpuSerial>,
        la: &Layout<D>,
        b: &Storage<TB, DeviceCpuSerial>,
        lb: &Layout<D>,
        f: impl FnMut(&mut TA, &TB),
    ) -> Result<()> {
        op_muta_refb_func_cpu_serial(a.rawvec_mut(), la, b.rawvec(), lb, f)
    }
}

impl<TA, TB, D> DeviceOp_MutA_NumB_API<TA, TB, D> for DeviceCpuSerial
where
    TA: Clone,
    D: DimAPI,
{
    fn op_muta_numb_func(
        &self,
        a: &mut Storage<TA, DeviceCpuSerial>,
        la: &Layout<D>,
        b: TB,
        f: impl FnMut(&mut TA, &TB),
    ) -> Result<()> {
        op_muta_numb_func_cpu_serial(a.rawvec_mut(), la, b, f)
    }
}

impl<T, D> DeviceOp_MutA_API<T, D> for DeviceCpuSerial
where
    T: Clone,
    D: DimAPI,
{
    fn op_muta_func(
        &self,
        a: &mut Storage<T, DeviceCpuSerial>,
        la: &Layout<D>,
        f: impl FnMut(&mut T),
    ) -> Result<()> {
        op_muta_func_cpu_serial(a.rawvec_mut(), la, f)
    }
}

impl<T, D> OpSumAPI<T, D> for DeviceCpuSerial
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
