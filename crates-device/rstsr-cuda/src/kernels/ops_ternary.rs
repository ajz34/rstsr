//! Where/clip device-op impls (B7).
//!
//! `where` and `clip` are same-dtype only on this backend: the CPU devices
//! promote scalar and mixed tensor operands through `DTypePromoteAPI`, but
//! promotion is not mirrored here yet (a documented divergence). Same-dtype
//! calls, including monomorphic scalar literals, behave like the CPU.
//!
//! Where arms: tensor-tensor through the planned ternary select; a scalar
//! arm rides the scalar parameter. Clip bound combinations are single-pass
//! fused expressions; both bounds scalar use the dedicated two-scalar
//! kernel.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaDType, CudaReal};
use crate::kernels::elementwise;
use crate::prelude_dev::*;

// /* #region where */
impl<T, D> OpWhereAPI<T, T, D> for DeviceCuda
where
    Self: DeviceRawAPI<bool, Raw = CudaBuffer<bool>>
        + DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType,
    D: DimAPI,
{
    type TOut = T;

    fn op_mutd_refa_refb_refc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
        c: &CudaBuffer<T>,
        lc: &Layout<D>,
    ) -> Result<()> {
        elementwise::where_launch::<T, T, T>(self, d, ld, a, la, b, lb, c, lc)
    }

    fn op_mutd_refa_refb_numc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
        c: T,
    ) -> Result<()> {
        elementwise::where_launch_sc::<T, T, T>(self, d, ld, a, la, b, lb, &c, false)
    }

    fn op_mutd_refa_numb_refc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        b: T,
        c: &CudaBuffer<T>,
        lc: &Layout<D>,
    ) -> Result<()> {
        // d = cond ? b : c is the scalar-c form with an inverted condition
        elementwise::where_launch_sc::<T, T, T>(self, d, ld, a, la, c, lc, &b, true)
    }
}

// /* #endregion */
// /* #region clip */
use crate::kernels::elementwise::ClipBound;

/// Clip: single pass over any bound combination (`clip_launch` carries
/// the bound kinds; expressions are the NaN-propagating max/min). Same-dtype
/// bounds only (see the module note).
impl<T, D> OpClipAPI<T, T, T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaReal,
    D: DimAPI,
{
    type TOut = T;

    fn op_mutd_refa_optrefb_optrefc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        b: Option<(&CudaBuffer<T>, &Layout<D>)>,
        c: Option<(&CudaBuffer<T>, &Layout<D>)>,
    ) -> Result<()> {
        let max_expr = crate::kernels::exprs::maximum(T::CNAME, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(T::CNAME, "val", "hi");
        let lo = b.map_or(ClipBound::None, |(buf, lb)| ClipBound::Tensor(buf, lb));
        let hi = c.map_or(ClipBound::None, |(buf, lc)| ClipBound::Tensor(buf, lc));
        elementwise::clip_launch::<T, T, T, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }

    fn op_mutd_refa_optrefb_optnumc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        b: Option<(&CudaBuffer<T>, &Layout<D>)>,
        c: Option<T>,
    ) -> Result<()> {
        let max_expr = crate::kernels::exprs::maximum(T::CNAME, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(T::CNAME, "val", "hi");
        let lo = b.map_or(ClipBound::None, |(buf, lb)| ClipBound::Tensor(buf, lb));
        let hi = c.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        elementwise::clip_launch::<T, T, T, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }

    fn op_mutd_refa_optnumb_optrefc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        b: Option<T>,
        c: Option<(&CudaBuffer<T>, &Layout<D>)>,
    ) -> Result<()> {
        let max_expr = crate::kernels::exprs::maximum(T::CNAME, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(T::CNAME, "val", "hi");
        let lo = b.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        let hi = c.map_or(ClipBound::None, |(buf, lc)| ClipBound::Tensor(buf, lc));
        elementwise::clip_launch::<T, T, T, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }

    fn op_mutd_refa_optnumb_optnumc(
        &self,
        d: &mut CudaBuffer<MaybeUninit<Self::TOut>>,
        ld: &Layout<D>,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        b: Option<T>,
        c: Option<T>,
    ) -> Result<()> {
        let max_expr = crate::kernels::exprs::maximum(T::CNAME, "val", "lo");
        let min_expr = crate::kernels::exprs::minimum(T::CNAME, "val", "hi");
        let lo = b.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        let hi = c.as_ref().map_or(ClipBound::None, |v| ClipBound::Scalar(v));
        elementwise::clip_launch::<T, T, T, D, D, D, D>(self, d, ld, a, la, lo, hi, &max_expr, &min_expr)
    }
}

// /* #endregion */
