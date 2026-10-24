//! Where/clip device-op impls (B7).
//!
//! Where arms: tensor-tensor through the planned ternary select; one
//! scalar arm rides the scalar parameter (the tensor tier promotes the
//! scalar to the x-dtype first, matching CPU's `promote_other`).
//! Clip bound combinations are single-pass fused expressions; both
//! bounds scalar use the dedicated two-scalar kernel.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaDType, CudaReal};
use crate::kernels::elementwise;
use crate::prelude_dev::*;

// /* #region where */
impl<TX, TY, D> OpWhereAPI<TX, TY, D> for DeviceCuda
where
    Self: DeviceRawAPI<bool, Raw = CudaBuffer<bool>>
        + DeviceRawAPI<TX, Raw = CudaBuffer<TX>>
        + DeviceRawAPI<TY, Raw = CudaBuffer<TY>>
        + DeviceRawAPI<MaybeUninit<TX>, Raw = CudaBuffer<MaybeUninit<TX>>>,
    TX: CudaDType,
    TY: CudaDType,
    D: DimAPI,
{
    type TOut = TX;

    fn op_mutd_refa_refb_refc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        b: &CudaBuffer<TX>,
        lb: &Layout<D>,
        c: &CudaBuffer<TY>,
        lc: &Layout<D>,
    ) -> Result<()> {
        elementwise::where_launch::<TX, TX, TY>(self, d, ld, a, la, b, lb, c, lc)
    }

    fn op_mutd_refa_refb_numc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        b: &CudaBuffer<TX>,
        lb: &Layout<D>,
        c: TY,
    ) -> Result<()> {
        // scalar promotes to TX at the tensor tier (CPU: promote_other);
        // SAFETY: same-type transmute — the caller passed TY = TX::Res
        // after promotion; the value bytes are the promoted representation
        // when TY == TX (same-type where is the supported promotion).
        let b_typed: &TX = unsafe { &*(core::ptr::addr_of!(c) as *const TX) };
        elementwise::where_launch_sc::<TX, TX, TX>(self, d, ld, a, la, b, lb, b_typed)
    }

    fn op_mutd_refa_numb_refc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        b: TX,
        c: &CudaBuffer<TY>,
        lc: &Layout<D>,
    ) -> Result<()> {
        // scalar-b: swap into the scalar-c form (c ? b : v flips to
        // !c ? c : b); the condition is inverted in-kernel — reuse the
        // same launcher by feeding the swapped arms through a bool-not
        // view is not free, so launch with (c, v=b) via the sc kernel on
        // swapped operands
        let a_not = crate::kernels::elementwise::bool_not_scratch(self, a, la)?;
        elementwise::where_launch_sc::<TX, TY, TX>(self, d, ld, &a_not, la, c, lc, &b)
    }
}

// /* #endregion */
// /* #region clip */
use crate::kernels::elementwise::ClipBound;

/// Clip: single pass over any bound combination (`clip_launch` carries
/// the bound kinds; expressions are the NaN-propagating max/min).
impl<TA, TB, TC, D> OpClipAPI<TA, TB, TC, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<TC, Raw = CudaBuffer<TC>>
        + DeviceRawAPI<MaybeUninit<TA>, Raw = CudaBuffer<MaybeUninit<TA>>>,
    TA: CudaDType + CudaReal,
    TB: CudaDType + CudaReal,
    TC: CudaDType + CudaReal,
    D: DimAPI,
{
    type TOut = TA;

    fn op_mutd_refa_optrefb_optrefc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: Option<(&CudaBuffer<TB>, &Layout<D>)>,
        c: Option<(&CudaBuffer<TC>, &Layout<D>)>,
    ) -> Result<()> {
        let t = TA::CNAME;
        let max_expr = crate::kernels::exprs::maximum(t, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(t, "val", "hi");
        let lo = b.map_or(ClipBound::None, |(buf, lb)| ClipBound::Tensor(buf, lb));
        let hi = c.map_or(ClipBound::None, |(buf, lc)| ClipBound::Tensor(buf, lc));
        elementwise::clip_launch::<TA, TB, TC, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }

    fn op_mutd_refa_optrefb_optnumc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: Option<(&CudaBuffer<TB>, &Layout<D>)>,
        c: Option<TC>,
    ) -> Result<()> {
        let t = TA::CNAME;
        let max_expr = crate::kernels::exprs::maximum(t, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(t, "val", "hi");
        let lo = b.map_or(ClipBound::None, |(buf, lb)| ClipBound::Tensor(buf, lb));
        let hi = c.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        elementwise::clip_launch::<TA, TB, TC, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }

    fn op_mutd_refa_optnumb_optrefc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: Option<TB>,
        c: Option<(&CudaBuffer<TC>, &Layout<D>)>,
    ) -> Result<()> {
        let t = TA::CNAME;
        let max_expr = crate::kernels::exprs::maximum(t, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(t, "val", "hi");
        let lo = b.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        let hi = c.map_or(ClipBound::None, |(buf, lc)| ClipBound::Tensor(buf, lc));
        elementwise::clip_launch::<TA, TB, TC, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }

    fn op_mutd_refa_optnumb_optnumc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: Option<TB>,
        c: Option<TC>,
    ) -> Result<()> {
        let t = TA::CNAME;
        let max_expr = crate::kernels::exprs::maximum(t, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(t, "val", "hi");
        let lo = b.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        let hi = c.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        elementwise::clip_launch::<TA, TB, TC, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }
}

// /* #endregion */
