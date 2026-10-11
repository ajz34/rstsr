//! Cross product of 3-element vectors along one axis.

use crate::prelude_dev::*;

/* #region cross by function */

/// Cross product of 3-element vectors along one axis.
///
/// Let $a$ and $b$ be arrays whose `axis` has length 3. The result has the broadcast shape of `a`
/// and `b`, and its component $k$ along `axis` is
///
/// $$c_k = a_{k+1} b_{k+2} - a_{k+2} b_{k+1}$$
///
/// with the indices taken cyclically (modulo 3) and **no conjugation**.
///
/// Both operands must have length 3 along `axis` (before broadcasting); the other axes broadcast
/// against each other. `axis` counts from the back when negative, and [`None`][AxisIndex::None]
/// (or `()`) selects the last axis (the Array-API default `axis = -1`). The result follows the
/// input's device default order and behaves identically under [`RowMajor`] and [`ColMajor`].
///
/// # Parameters
///
/// - `a`: impl [`TensorViewAPI`]
///
///   - The first input array; must have length 3 along `axis`.
///
/// - `b`: impl [`TensorViewAPI`]
///
///   - The second input array; must have length 3 along `axis`.
///
/// - `axis`: impl `TryInto<`[`AxisIndex<isize>`]`>`
///
///   - The axis along which to compute the cross product; defaults to the last axis.
///
/// # Returns
///
/// [`Tensor<TA::Output, B, DC>`]
///
/// - The cross product, with the broadcast shape of the inputs.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 2, 3], &device);
/// let b = rt::tensor_from_nested!([4, 5, 6], &device);
/// let c = rt::cross(&a, &b, -1);
/// println!("{c}");
/// // [-3  6 -3]
/// # assert_eq!(c.to_vec(), vec![-3, 6, -3]);
/// ```
///
/// The axis may be selected explicitly, and the operands broadcast on the other axes:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
/// let b = rt::tensor_from_nested!([4, 5, 6], &device);
/// let c = rt::cross(&a, &b, 1);
/// assert_eq!(c.shape(), &[2, 3]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `linalg.cross(x1, x2, /, *, axis=-1)` ([`linalg.cross`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.cross.html))
///   — this is the `xp.linalg` extension member.
/// - NumPy: `numpy.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)` ([`numpy.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html))
///   — the same component formula and the same no-conjugation rule, but NumPy allows an independent
///   axis per operand and for the result; this function uses one shared `axis`.
/// - RSTSR: `rt::cross(&a, &b, axis)`, method `a.cross(&b, axis)`.
///
/// # Panics
///
/// - Panics if the two operands are not on the same device, if `axis` is out of bounds for either
///   operand, or if `axis` does not have length 3 in both operands.
///
/// For a fallible version, use [`cross_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Array-API standard: [`linalg.cross`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.cross.html)
/// - NumPy: [`numpy.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)
///
/// ## Related functions in RSTSR
///
/// - [`ext_cross`]: the array-API-fulfilment form, which promotes mixed-dtype operands.
/// - [`vecdot`]: the contraction of two vectors into a scalar; conjugation and no conjugation
///   respectively.
///
/// ## Variants of this function
///
/// - [`cross`] / [`cross_f`]: Returning a new tensor.
pub fn cross<TA, TB, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Tensor<TA::Output, B, DC>
where
    // dimension
    DA: DimAPI + DimMaxAPI<DB, Max = DC>,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB>,
    B: DeviceCrossAPI<TA, TB, TA::Output, DC>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    cross_f(a, b, axis).rstsr_unwrap()
}

/// Cross product of 3-element vectors along one axis.
///
/// See also [`cross`].
pub fn cross_f<TA, TB, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TA::Output, B, DC>>
where
    // dimension
    DA: DimAPI + DimMaxAPI<DB, Max = DC>,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB>,
    B: DeviceCrossAPI<TA, TB, TA::Output, DC>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    op_refa_refb_cross(a, b, axis)
}

/// Device-level driver of the cross product, allocating the output; see also [`cross`].
pub fn op_refa_refb_cross<TA, TB, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TA::Output, B, DC>>
where
    // dimension
    DA: DimAPI + DimMaxAPI<DB, Max = DC>,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB>,
    B: DeviceCrossAPI<TA, TB, TA::Output, DC>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    let (a, b) = (a.view(), b.view());

    // check devices
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // the array-API contract: the axis is size 3 in both operands, before broadcasting
    let ndim = a.ndim().max(b.ndim());
    let axis = cross_axis_in_output(axis.try_into().map_err(Into::into)?, ndim)?;
    cross_check_axis(a.shape(), a.ndim(), axis, ndim)?;
    cross_check_axis(b.shape(), b.ndim(), axis, ndim)?;

    let order = device.default_order();
    let (la, lb) = broadcast_layout(a.layout(), b.layout(), order)?;
    // the freshly allocated result follows the input's device default order
    let lc = match order {
        RowMajor => la.shape().c(),
        ColMajor => la.shape().f(),
    };
    let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
    device.cross(storage_c.raw_mut(), &lc, a.raw(), &la, b.raw(), &lb, axis)?;
    // SAFETY: `device.cross` above wrote every element of `lc`, covering the
    // fresh storage exactly.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, lc) }
}

/// Normalize the requested axis to a non-negative index into the (broadcast) output rank.
pub(crate) fn cross_axis_in_output(axis: AxisIndex<isize>, ndim: usize) -> Result<usize> {
    rstsr_assert!(ndim > 0, InvalidValue, "cross expects at least one axis")?;
    match axis {
        AxisIndex::None => Ok(ndim - 1),
        AxisIndex::Val(axis) => rstsr_check_axis!(axis, ndim),
    }
}

/// Check that an operand carries the compute axis at length 3, mapping the output-rank axis back
/// to the operand's own rank (broadcasting aligns trailing axes).
pub(crate) fn cross_check_axis<D>(shape: &D, ndim: usize, axis: usize, out_ndim: usize) -> Result<()>
where
    D: DimAPI,
{
    let axis = axis as isize - (out_ndim as isize - ndim as isize);
    rstsr_assert!(axis >= 0, InvalidValue, "cross: the axis must be shared by both operands")?;
    rstsr_assert_eq!(shape[axis as usize], 3, InvalidValue, "cross: the axis must have length 3")?;
    Ok(())
}

/* #endregion */

/* #region cross tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Cross product of 3-element vectors along one axis.
    ///
    /// See also [`cross`].
    pub fn cross_f<TB, DC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    ) -> Result<Tensor<T::Output, B, DC>>
    where
        // dimension
        D: DimAPI + DimMaxAPI<DB, Max = DC>,
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB>,
        B: DeviceCrossAPI<T, TB, T::Output, DC>
            + DeviceAPI<TB>
            + DeviceAPI<T::Output>
            + DeviceCreationAnyAPI<T::Output>,
    {
        op_refa_refb_cross(self.view(), b, axis)
    }

    /// Cross product of 3-element vectors along one axis.
    ///
    /// See also [`cross`].
    pub fn cross<TB, DC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    ) -> Tensor<T::Output, B, DC>
    where
        // dimension
        D: DimAPI + DimMaxAPI<DB, Max = DC>,
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB>,
        B: DeviceCrossAPI<T, TB, T::Output, DC>
            + DeviceAPI<TB>
            + DeviceAPI<T::Output>
            + DeviceCreationAnyAPI<T::Output>,
    {
        op_refa_refb_cross(self.view(), b, axis).rstsr_unwrap()
    }
}

/* #endregion */
