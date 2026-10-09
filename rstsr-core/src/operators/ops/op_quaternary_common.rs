//! Quaternary elementwise operations: one output plus up to three inputs.
//!
//! - [`OpWhereAPI`]: the select operation (NumPy `where` / array API `where(condition, x1, x2)`);
//!   operand letters map as (a, b, c) = (cond, x, y), output letter `d`.
//! - [`OpClipAPI`]: element-wise clip, `d = min(max(a, b), c)`, where the two bounds `b`/`c` are
//!   each optional.

use crate::prelude_dev::*;

// select operation: where(cond, x, y); operands map to (a, b, c) = (cond, x, y)

pub trait OpWhereAPI<TX, TY, D>
where
    D: DimAPI,
    Self: DeviceAPI<bool> + DeviceAPI<TX> + DeviceAPI<TY> + DeviceAPI<MaybeUninit<Self::TOut>>,
{
    type TOut;

    fn op_mutd_refa_refb_refc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TX>>::Raw,
        lb: &Layout<D>,
        c: &<Self as DeviceRawAPI<TY>>::Raw,
        lc: &Layout<D>,
    ) -> Result<()>;

    fn op_mutd_refa_refb_numc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TX>>::Raw,
        lb: &Layout<D>,
        c: TY,
    ) -> Result<()>;

    fn op_mutd_refa_numb_refc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        b: TX,
        c: &<Self as DeviceRawAPI<TY>>::Raw,
        lc: &Layout<D>,
    ) -> Result<()>;
}

// clip operation: clip(a, (b, c)) = min(max(a, b), c); either bound may be absent

/// Element-wise clip: `d = min(max(a, b), c)`, where `a` is the tensor and `b`,
/// `c` are the lower / upper bounds, each of which may be a tensor (the
/// `*refb*` / `*refc*` methods) or a scalar (the `*numb*` / `*numc*` methods),
/// or absent (`None`).
///
/// This trait deliberately performs no dtype promotion: the element types
/// `TA`/`TB`/`TC` are the operand dtypes, and each device implementation is
/// free to pick its own `TOut` (the reference devices promote `TOut = promote(TA,
/// TB, TC)`). An absent bound is passed as `None`; its dtype parameter still
/// has to name some element type of the device, so callers use `TA` as the
/// neutral placeholder.
pub trait OpClipAPI<TA, TB, TC, D>
where
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC> + DeviceAPI<MaybeUninit<Self::TOut>>,
{
    type TOut;

    fn op_mutd_refa_optrefb_optrefc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: Option<(&<Self as DeviceRawAPI<TB>>::Raw, &Layout<D>)>,
        c: Option<(&<Self as DeviceRawAPI<TC>>::Raw, &Layout<D>)>,
    ) -> Result<()>;

    fn op_mutd_refa_optrefb_optnumc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: Option<(&<Self as DeviceRawAPI<TB>>::Raw, &Layout<D>)>,
        c: Option<TC>,
    ) -> Result<()>;

    fn op_mutd_refa_optnumb_optrefc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: Option<TB>,
        c: Option<(&<Self as DeviceRawAPI<TC>>::Raw, &Layout<D>)>,
    ) -> Result<()>;

    fn op_mutd_refa_optnumb_optnumc(
        &self,
        d: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        ld: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: Option<TB>,
        c: Option<TC>,
    ) -> Result<()>;
}
