//! Quaternary elementwise operations (see
//! `crate::operators::ops::op_quaternary_common`).

use crate::prelude_dev::*;
use rstsr_dtype_traits::ExtReal;

// Special case for where (select): promote only the selected branch, so the
// discarded operand is never cloned or promoted.

#[inline]
fn select_promote<TX, TY>(cond: bool, x: TX, y: TY) -> <TX as DTypePromoteAPI<TY>>::Res
where
    TX: DTypePromoteAPI<TY>,
{
    if cond {
        TX::promote_self(x)
    } else {
        <TX as DTypePromoteAPI<TY>>::promote_other(y)
    }
}

impl<TX, TY, D> OpWhereAPI<TX, TY, D> for DeviceRayonAutoImpl
where
    TX: Clone + Send + Sync + DTypePromoteAPI<TY, Res: Clone + Send + Sync>,
    TY: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = TX::Res;

    fn op_mutd_refa_refb_refc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<bool>,
        la: &Layout<D>,
        b: &Vec<TX>,
        lb: &Layout<D>,
        c: &Vec<TY>,
        lc: &Layout<D>,
    ) -> Result<()> {
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, b: &TX, c: &TY| {
            d.write(select_promote(*a, b.clone(), c.clone()));
        };
        self.op_mutd_refa_refb_refc_func(d, ld, a, la, b, lb, c, lc, &mut func)
    }

    fn op_mutd_refa_refb_numc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<bool>,
        la: &Layout<D>,
        b: &Vec<TX>,
        lb: &Layout<D>,
        c: TY,
    ) -> Result<()> {
        let c = <TX as DTypePromoteAPI<TY>>::promote_other(c);
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, b: &TX| {
            d.write(if *a { TX::promote_self(b.clone()) } else { c.clone() });
        };
        self.op_mutc_refa_refb_func(d, ld, a, la, b, lb, &mut func)
    }

    fn op_mutd_refa_numb_refc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<bool>,
        la: &Layout<D>,
        b: TX,
        c: &Vec<TY>,
        lc: &Layout<D>,
    ) -> Result<()> {
        let b = TX::promote_self(b);
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &bool, c: &TY| {
            d.write(if *a { b.clone() } else { <TX as DTypePromoteAPI<TY>>::promote_other(c.clone()) });
        };
        self.op_mutc_refa_refb_func(d, ld, a, la, c, lc, &mut func)
    }
}

// clip: d = min(max(a, b), c) with either bound optional; promote all three
// operands to their common type before clamping.

type ClipRes<TA, TB, TC> = <<TA as DTypePromoteAPI<TB>>::Res as DTypePromoteAPI<TC>>::Res;
type ClipResT<TA, TB> = <TA as DTypePromoteAPI<TB>>::Res;

#[inline]
fn clip_promote_a<TA, TB, TC>(a: TA) -> ClipRes<TA, TB, TC>
where
    TA: DTypePromoteAPI<TB>,
    <TA as DTypePromoteAPI<TB>>::Res: DTypePromoteAPI<TC>,
{
    let ab = <TA as DTypePromoteAPI<TB>>::promote_self(a);
    <ClipResT<TA, TB> as DTypePromoteAPI<TC>>::promote_self(ab)
}

#[inline]
fn clip_promote_b<TA, TB, TC>(b: TB) -> ClipRes<TA, TB, TC>
where
    TA: DTypePromoteAPI<TB>,
    <TA as DTypePromoteAPI<TB>>::Res: DTypePromoteAPI<TC>,
{
    let ab = <TA as DTypePromoteAPI<TB>>::promote_other(b);
    <ClipResT<TA, TB> as DTypePromoteAPI<TC>>::promote_self(ab)
}

#[inline]
fn clip_promote_c<TA, TB, TC>(c: TC) -> ClipRes<TA, TB, TC>
where
    TA: DTypePromoteAPI<TB>,
    <TA as DTypePromoteAPI<TB>>::Res: DTypePromoteAPI<TC>,
{
    <ClipResT<TA, TB> as DTypePromoteAPI<TC>>::promote_other(c)
}

/// Shared promote-then-clamp step, reused by every operand-kind combination.
#[inline]
fn clip_apply<T>(x: T, lo: Option<&T>, hi: Option<&T>) -> T
where
    T: ExtReal,
{
    let mut r = x;
    if let Some(lo) = lo {
        r = r.ext_max(lo.clone());
    }
    if let Some(hi) = hi {
        r = r.ext_min(hi.clone());
    }
    r
}

impl<TA, TB, TC, D> OpClipAPI<TA, TB, TC, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB>,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync,
    <TA as DTypePromoteAPI<TB>>::Res: DTypePromoteAPI<TC>,
    ClipRes<TA, TB, TC>: ExtReal + Send + Sync,
    D: DimAPI,
{
    type TOut = ClipRes<TA, TB, TC>;

    fn op_mutd_refa_optrefb_optrefc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: Option<(&Vec<TB>, &Layout<D>)>,
        c: Option<(&Vec<TC>, &Layout<D>)>,
    ) -> Result<()> {
        match (b, c) {
            (Some((b, lb)), Some((c, lc))) => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB, c: &TC| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    let lo = clip_promote_b::<TA, TB, TC>(b.clone());
                    let hi = clip_promote_c::<TA, TB, TC>(c.clone());
                    d.write(clip_apply(x, Some(&lo), Some(&hi)));
                };
                self.op_mutd_refa_refb_refc_func(d, ld, a, la, b, lb, c, lc, &mut func)
            },
            (Some((b, lb)), None) => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    let lo = clip_promote_b::<TA, TB, TC>(b.clone());
                    d.write(clip_apply(x, Some(&lo), None));
                };
                self.op_mutc_refa_refb_func(d, ld, a, la, b, lb, &mut func)
            },
            (None, Some((c, lc))) => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA, c: &TC| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    let hi = clip_promote_c::<TA, TB, TC>(c.clone());
                    d.write(clip_apply(x, None, Some(&hi)));
                };
                self.op_mutc_refa_refb_func(d, ld, a, la, c, lc, &mut func)
            },
            (None, None) => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA| {
                    d.write(clip_promote_a::<TA, TB, TC>(a.clone()));
                };
                self.op_muta_refb_func(d, ld, a, la, &mut func)
            },
        }
    }

    fn op_mutd_refa_optrefb_optnumc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: Option<(&Vec<TB>, &Layout<D>)>,
        c: Option<TC>,
    ) -> Result<()> {
        let hi = c.map(clip_promote_c::<TA, TB, TC>);
        match b {
            Some((b, lb)) => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    let lo = clip_promote_b::<TA, TB, TC>(b.clone());
                    d.write(clip_apply(x, Some(&lo), hi.as_ref()));
                };
                self.op_mutc_refa_refb_func(d, ld, a, la, b, lb, &mut func)
            },
            None => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    d.write(clip_apply(x, None, hi.as_ref()));
                };
                self.op_muta_refb_func(d, ld, a, la, &mut func)
            },
        }
    }

    fn op_mutd_refa_optnumb_optrefc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: Option<TB>,
        c: Option<(&Vec<TC>, &Layout<D>)>,
    ) -> Result<()> {
        let lo = b.map(clip_promote_b::<TA, TB, TC>);
        match c {
            Some((c, lc)) => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA, c: &TC| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    let hi = clip_promote_c::<TA, TB, TC>(c.clone());
                    d.write(clip_apply(x, lo.as_ref(), Some(&hi)));
                };
                self.op_mutc_refa_refb_func(d, ld, a, la, c, lc, &mut func)
            },
            None => {
                let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA| {
                    let x = clip_promote_a::<TA, TB, TC>(a.clone());
                    d.write(clip_apply(x, lo.as_ref(), None));
                };
                self.op_muta_refb_func(d, ld, a, la, &mut func)
            },
        }
    }

    fn op_mutd_refa_optnumb_optnumc(
        &self,
        d: &mut Vec<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: Option<TB>,
        c: Option<TC>,
    ) -> Result<()> {
        let lo = b.map(clip_promote_b::<TA, TB, TC>);
        let hi = c.map(clip_promote_c::<TA, TB, TC>);
        let mut func = |d: &mut MaybeUninit<Self::TOut>, a: &TA| {
            let x = clip_promote_a::<TA, TB, TC>(a.clone());
            d.write(clip_apply(x, lo.as_ref(), hi.as_ref()));
        };
        self.op_muta_refb_func(d, ld, a, la, &mut func)
    }
}
