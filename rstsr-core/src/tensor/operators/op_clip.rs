//! Element-wise clip function `clip`.
//!
//! The anchor documentation is on [`clip`](clip()).

use crate::prelude_dev::*;

/* #region bound argument */

/// One bound of a [`clip`](clip()) call: a scalar, a tensor view, or the
/// `None` wildcard.
pub enum ClipSide<'a, T, B, D>
where
    D: DimAPI,
    B: DeviceRawAPI<T>,
{
    /// No bound on this side.
    Absent,
    /// Scalar bound.
    Scalar(T),
    /// Tensor bound, broadcast against the clipped tensor.
    View(TensorView<'a, T, B, D>),
}

/// Normalization of a `clip` bound position into a [`ClipSide`].
///
/// Implemented for real scalars, tensor operands (a [`TensorView`] or any
/// `&TensorAny`), and `Option<TA>`: `None` is the wildcard (no bound on that
/// side) and `Some(v)` a scalar bound of the tensor's own element type. `TA`
/// (the element type of the tensor being clipped) fixes the option's
/// otherwise-unconstrained type parameter.
pub trait ClipSideArg<TA, B, D>
where
    D: DimAPI,
    B: DeviceRawAPI<Self::TD>,
{
    /// Element type of the bound (`TA` when the side is absent).
    type TD;
    /// Dimensionality of the bound (the tensor's own dimensionality when absent).
    type TDim: DimAPI;

    /// Normalize `self` into a [`ClipSide`].
    fn clip_side<'a>(self) -> ClipSide<'a, Self::TD, B, Self::TDim>
    where
        Self: 'a;
}

impl<TA, B, D> ClipSideArg<TA, B, D> for Option<TA>
where
    D: DimAPI,
    B: DeviceRawAPI<TA>,
{
    type TD = TA;
    type TDim = D;

    fn clip_side<'a>(self) -> ClipSide<'a, TA, B, D>
    where
        Self: 'a,
    {
        match self {
            None => ClipSide::Absent,
            Some(v) => ClipSide::Scalar(v),
        }
    }
}

/// Scalar element types accepted as a `clip` bound.
///
/// This marker exists so the blanket [`ClipSideArg`] impl below can name a
/// *local* bound: a blanket over a foreign bound (`num::Num`, `ExtReal`) is
/// rejected by coherence against the `Option` wildcard impl, because an
/// upstream crate could still add `Num for Option<_>`. A local trait with a
/// closed set of impls is provably disjoint from `Option`, and — unlike a set
/// of per-type impls — it keeps integer/float literals defaulting to their
/// usual types in `rt::clip(&x, (2, 5))`.
///
/// The impls are the `DTypePromoteAPI` scalar dtypes (so a bound can promote
/// against the tensor); the pointer-width and 128-bit integers are not in that
/// lattice and therefore cannot be used as a bound.
pub trait ClipScalar: Clone {}

macro_rules! impl_clip_scalar {
    ($($t:ty),* $(,)?) => {
        $( impl ClipScalar for $t {} )*
    };
}

impl_clip_scalar!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl<TA, TB, B, D> ClipSideArg<TA, B, D> for TB
where
    TB: ClipScalar,
    D: DimAPI,
    B: DeviceRawAPI<TB>,
{
    type TD = TB;
    type TDim = D;

    fn clip_side<'a>(self) -> ClipSide<'a, TB, B, D>
    where
        Self: 'a,
    {
        ClipSide::Scalar(self)
    }
}

impl<TA, RB, TB, B, DB, D> ClipSideArg<TA, B, D> for &TensorAny<RB, TB, B, DB>
where
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    DB: DimAPI,
    D: DimAPI,
    B: DeviceRawAPI<TB> + DeviceAPI<TB>,
{
    type TD = TB;
    type TDim = DB;

    fn clip_side<'a>(self) -> ClipSide<'a, TB, B, DB>
    where
        Self: 'a,
    {
        ClipSide::View(self.view())
    }
}

impl<'r, TA, TB, B, DB, D> ClipSideArg<TA, B, D> for TensorView<'r, TB, B, DB>
where
    DB: DimAPI,
    D: DimAPI,
    B: DeviceRawAPI<TB>,
{
    type TD = TB;
    type TDim = DB;

    fn clip_side<'a>(self) -> ClipSide<'a, TB, B, DB>
    where
        Self: 'a,
    {
        ClipSide::View(self)
    }
}

/// Arguments for [`clip`](clip()): the lower and upper bounds as a `(lo, hi)`
/// pair.
///
/// Each bound may be a scalar, a tensor (broadcast against the clipped tensor),
/// or `None` for "no bound on this side". With both bounds absent the clip is
/// the identity (a copy of the input), matching the array-API.
pub struct ClipArgs<LO, HI> {
    /// Lower bound.
    pub lo: LO,
    /// Upper bound.
    pub hi: HI,
}

impl<LO, HI> From<(LO, HI)> for ClipArgs<LO, HI> {
    fn from((lo, hi): (LO, HI)) -> Self {
        Self { lo, hi }
    }
}

/* #endregion */

/* #region tensor trait */

/// API trait for element-wise clip; the anchor is the free function
/// [`clip`](clip()).
pub trait TensorClipAPI<LO, HI> {
    type Output;

    /// Fallible method form of element-wise clip.
    fn clip_f(self, args: impl Into<ClipArgs<LO, HI>>) -> Result<Self::Output>;

    /// Panicking method form of element-wise clip.
    fn clip(self, args: impl Into<ClipArgs<LO, HI>>) -> Self::Output
    where
        Self: Sized,
    {
        self.clip_f(args).rstsr_unwrap()
    }
}

type ClipDim<DA, LD, HD> = <<DA as DimMaxAPI<LD>>::Max as DimMaxAPI<HD>>::Max;

impl<RA, TA, B, DA, LO, HI> TensorClipAPI<LO, HI> for &TensorAny<RA, TA, B, DA>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    LO: ClipSideArg<TA, B, DA>,
    HI: ClipSideArg<TA, B, DA>,
    DA: DimAPI + DimMaxAPI<LO::TDim>,
    LO::TDim: DimAPI,
    DA::Max: DimAPI + DimMaxAPI<HI::TDim>,
    HI::TDim: DimAPI,
    B: OpClipAPI<TA, LO::TD, HI::TD, ClipDim<DA, LO::TDim, HI::TDim>>,
    B: DeviceRawAPI<TA> + DeviceRawAPI<LO::TD> + DeviceRawAPI<HI::TD>,
    B: DeviceAPI<TA>
        + DeviceAPI<LO::TD>
        + DeviceAPI<HI::TD>
        + DeviceAPI<B::TOut>
        + DeviceAPI<MaybeUninit<B::TOut>>
        + DeviceCreationAnyAPI<B::TOut>,
{
    type Output = Tensor<B::TOut, B, ClipDim<DA, LO::TDim, HI::TDim>>;

    fn clip_f(self, args: impl Into<ClipArgs<LO, HI>>) -> Result<Self::Output> {
        let ClipArgs { lo, hi } = args.into();
        let lo = lo.clip_side();
        let hi = hi.clip_side();

        // move each side apart before borrowing, so no `Clone` is needed
        let (lo_scalar, lo_view) = match lo {
            ClipSide::Absent => (None, None),
            ClipSide::Scalar(s) => (Some(s), None),
            ClipSide::View(v) => (None, Some(v)),
        };
        let (hi_scalar, hi_view) = match hi {
            ClipSide::Absent => (None, None),
            ClipSide::Scalar(s) => (Some(s), None),
            ClipSide::View(v) => (None, Some(v)),
        };

        // check device
        if let Some(v) = &lo_view {
            rstsr_assert!(self.device().same_device(v.device()), DeviceMismatch)?;
        }
        if let Some(v) = &hi_view {
            rstsr_assert!(self.device().same_device(v.device()), DeviceMismatch)?;
        }

        // broadcast layouts; a scalar/absent bound contributes the tensor's own
        // layout (its `TDim` is `DA`), so the pairwise chain covers every shape
        let device = self.device();
        let order = device.default_order();
        let la = self.layout().to_dim::<IxD>()?;
        let llo = match &lo_view {
            Some(v) => v.layout().to_dim::<IxD>()?,
            None => la.clone(),
        };
        let lhi = match &hi_view {
            Some(v) => v.layout().to_dim::<IxD>()?,
            None => la.clone(),
        };
        let (la_b, llo_b) = broadcast_layout(&la, &llo, order)?;
        let (la_f, lhi_f) = broadcast_layout(&la_b, &lhi, order)?;
        let (llo_f, _) = broadcast_layout(&llo_b, &lhi_f, order)?;
        let lc = layout_for_clip_output(&[&la_f, &llo_f, &lhi_f], order)?;

        let la_f = la_f.to_dim::<ClipDim<DA, LO::TDim, HI::TDim>>()?;
        let llo_f = llo_f.to_dim::<ClipDim<DA, LO::TDim, HI::TDim>>()?;
        let lhi_f = lhi_f.to_dim::<ClipDim<DA, LO::TDim, HI::TDim>>()?;
        let lc = lc.to_dim::<ClipDim<DA, LO::TDim, HI::TDim>>()?;

        // perform operation and return
        let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
        match (lo_view.as_ref(), hi_view.as_ref()) {
            (Some(lv), Some(hv)) => {
                device.op_mutd_refa_optrefb_optrefc(
                    storage_c.raw_mut(),
                    &lc,
                    self.raw(),
                    &la_f,
                    Some((lv.raw(), &llo_f)),
                    Some((hv.raw(), &lhi_f)),
                )?;
            },
            (Some(lv), None) => {
                device.op_mutd_refa_optrefb_optnumc(
                    storage_c.raw_mut(),
                    &lc,
                    self.raw(),
                    &la_f,
                    Some((lv.raw(), &llo_f)),
                    hi_scalar,
                )?;
            },
            (None, Some(hv)) => {
                device.op_mutd_refa_optnumb_optrefc(
                    storage_c.raw_mut(),
                    &lc,
                    self.raw(),
                    &la_f,
                    lo_scalar,
                    Some((hv.raw(), &lhi_f)),
                )?;
            },
            (None, None) => {
                device.op_mutd_refa_optnumb_optnumc(
                    storage_c.raw_mut(),
                    &lc,
                    self.raw(),
                    &la_f,
                    lo_scalar,
                    hi_scalar,
                )?;
            },
        }
        // SAFETY: the op above wrote every element of the fresh `storage_c`.
        let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
        Tensor::new_f(storage_c, lc)
    }
}

/// Output layout for element-wise clip: the common elementwise-op policy of
/// `layout_for_array_copy` candidates over all operand layouts, falling back to
/// the device default order when the operands disagree.
fn layout_for_clip_output<D>(layouts: &[&Layout<D>], order: FlagOrder) -> Result<Layout<D>>
where
    D: DimAPI,
{
    let candidates =
        layouts.iter().map(|l| layout_for_array_copy(l, TensorIterOrder::default())).collect::<Result<Vec<_>>>()?;
    if candidates.iter().all(|c| *c == candidates[0]) {
        Ok(candidates[0].clone())
    } else {
        match order {
            RowMajor => Ok(layouts[0].shape().c()),
            ColMajor => Ok(layouts[0].shape().f()),
        }
    }
}

/* #endregion */

/* #region function impl */

/// Clip (limit) the values of a tensor.
///
/// Element-wise `min(max(x, lo), hi)`: values below `lo` become `lo`, values
/// above `hi` become `hi`. Either bound may be omitted with `None`, in which
/// case only the other bound applies; with both bounds absent the result is a
/// copy of `a` (array-API `clip`). Each bound may be a scalar or a tensor
/// (broadcast against `x`), and scalar bounds follow rstsr's usual (strong)
/// promotion, like
/// [`maximum`](crate::tensor::operators::op_binary_common::maximum()). The
/// result is a newly allocated tensor of the promoted dtype and the broadcast
/// shape of the tensor operands.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// See [`order_semantics`](crate::order_semantics) for the two device default
/// orders: the tensor shapes are broadcast together from the last axis under
/// [`RowMajor`] (NumPy-like) and from the first axis under [`ColMajor`], so a
/// call accepted by one order may be rejected by the other.
///
/// # Parameters
///
/// - `a`: [`&TensorAny<R, T, B, D>`](TensorAny) - the tensor to clip.
/// - `args`: the `(lo, hi)` bounds (any `Into<`[`ClipArgs`]`>`):
///   - `lo`, `hi`: the lower and upper bounds; each is a scalar, a tensor
///     ([`&TensorAny`](TensorAny) or [`TensorView`]), or `None` (no bound on that side).
///
/// # Returns
///
/// A newly allocated [`Tensor`]`<B::TOut, B, D>` with the broadcast shape and
/// the promoted dtype of the operands.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([-3, -1, 0, 2, 5], &device);
/// println!("{}", rt::clip(&x, (0, 3)));
/// // [ 0 0 0 2 3]
/// # assert_eq!(format!("{}", rt::clip(&x, (0, 3))), "[ 0 0 0 2 3]");
/// ```
///
/// A tensor bound broadcasts, and `None` disables one side:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([[-3, 2], [4, -5]], &device);
/// let lo = rt::tensor_from_nested!([[-1, 0], [1, -1]], &device);
/// println!("{}", rt::clip(&x, (&lo, 3)));
/// // [[ -1 2]
/// //  [ 3 -1]]
/// println!("{}", rt::clip(&x, (0, None)));
/// // [[ 0 2]
/// //  [ 4 0]]
/// # assert_eq!(format!("{}", rt::clip(&x, (&lo, 3))), "[[ -1 2]\n [ 3 -1]]");
/// # assert_eq!(format!("{}", rt::clip(&x, (0, None))), "[[ 0 2]\n [ 4 0]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `clip(x, /, min=None, max=None)` ([`clip`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.clip.html))
/// - NumPy: `numpy.clip(a, a_min, a_max)` ([`numpy.clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html))
/// - RSTSR: `rt::clip(&x, (lo, hi))` or `x.clip((lo, hi))`
///
/// Please note the differences (recorded in the
/// [tracking registry](https://github.com/RESTGroup/rstsr/blob/master/rstsr-core/tests/tracking/numpy_differences.md)):
///
/// - Scalar bounds are strongly promoted (rstsr house rule); NumPy's NEP 50 weak-scalar dtype
///   minimization is not implemented.
/// - The two existing NumPy entry points (`a_min`/`a_max` as positional or keyword arguments) are
///   unified into a single `(lo, hi)` argument group carrying `None` for an absent bound.
/// - With both bounds absent (`(None, None)`), the result is a copy of `a` (array-API); NumPy's
///   bound-less `clip(a)` instead raises `ValueError`.
///
/// # Panics
///
/// - Panics if the bounds are on a different device than `a`.
/// - Panics if the shapes cannot be broadcast together under the current default order (see the
///   notice above).
///
/// For a fallible version, use [`clip_f`](clip_f()).
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy `numpy.clip` and array-api `clip` (see `# Notes of API accordance` above).
///
/// ## Related functions in RSTSR
///
/// - [`maximum`](crate::tensor::operators::op_binary_common::maximum()) and
///   [`minimum`](crate::tensor::operators::op_binary_common::minimum()): the one-sided bounds.
///
/// ## Variants of this function
///
/// - [`clip_f`](clip_f()): fallible version, actual implementation.
/// - Method form on trait `TensorClipAPI`: `x.clip((lo, hi))`.
pub fn clip<TRA, LO, HI>(a: TRA, args: impl Into<ClipArgs<LO, HI>>) -> TRA::Output
where
    TRA: TensorClipAPI<LO, HI>,
{
    a.clip(args)
}

/// Element-wise clip (fallible): limits the values of a tensor to `[lo, hi]`.
///
/// See also [`clip`](clip()).
pub fn clip_f<TRA, LO, HI>(a: TRA, args: impl Into<ClipArgs<LO, HI>>) -> Result<TRA::Output>
where
    TRA: TensorClipAPI<LO, HI>,
{
    a.clip_f(args)
}

/* #endregion */
