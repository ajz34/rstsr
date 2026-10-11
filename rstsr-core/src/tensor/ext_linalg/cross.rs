//! Cross product with Array-API dtype promotion.

use crate::prelude_dev::*;
use crate::tensor::linalg::cross::{cross_axis_in_output, cross_check_axis};

/* #region ext_cross by function */

/// Cross product of 3-element vectors along one axis, promoting mixed-dtype operands to their
/// common dtype.
///
/// Let $a$ and $b$ be arrays whose `axis` has length 3. The result has the broadcast shape of `a`
/// and `b`, and its component $k$ along `axis` is
///
/// $$c_k = a_{k+1} b_{k+2} - a_{k+2} b_{k+1}$$
///
/// with the indices taken cyclically (modulo 3) and **no conjugation**. The operands may have
/// different dtypes: each pair is promoted to its common dtype ([`DTypePromoteAPI`], the same rule
/// as NumPy) before the product.
///
/// [`cross`] instead requires the operands to share one dtype; this function is the
/// array-API-fulfilment form, like the `ext_*` element-wise arithmetic functions. The shape
/// contract is that of [`cross`]: both operands must have length 3 along `axis`, and the
/// Row/Column Major Notice of [`cross`] applies to operands of different ranks.
///
/// <div class="warning">
///
/// **Array-API Compliance Form**
///
/// This function exists only for array-API compliance, not as the idiomatic rstsr surface. Prefer
/// [`cross`] whenever the operands already share a dtype.
///
/// </div>
///
/// # Parameters
///
/// - `a`: the first operand; must have length 3 along `axis` (views and owned tensors both
///   accepted).
/// - `b`: the second operand; must have length 3 along `axis` (views and owned tensors both
///   accepted).
/// - `axis`: the axis along which to compute the cross product; defaults to the last axis.
///
/// # Returns
///
/// - [`Tensor<TC, B, DC>`][`Tensor`]: the cross product in the promoted dtype, with the broadcast
///   shape of the inputs.
///
/// # Examples
///
/// Mixed `i8` × `i16` operands give the promoted `i16` result:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1i8, 2, 3], &device);
/// let b = rt::tensor_from_nested!([4i16, 5, 6], &device);
/// let c = rt::ext_cross(&a, &b, -1);
/// assert_eq!(c.to_vec(), vec![-3i16, 6, -3]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `linalg.cross(x1, x2, /, *, axis=-1)` ([`linalg.cross`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.cross.html))
///   — the dtype is `result_type(x1.dtype, x2.dtype)`, as this function computes it.
/// - NumPy: `numpy.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)` ([`numpy.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)).
/// - RSTSR: `rt::ext_cross(&a, &b, axis)`, method `a.ext_cross(&b, axis)`.
///
/// # Panics
///
/// - Panics if the two operands are not on the same device, if `axis` is out of bounds for either
///   operand, or if `axis` does not have length 3 in both operands.
///
/// For a fallible version, use [`ext_cross_f`].
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
/// - [`cross`]: the same-dtype form, for operands that already share one dtype.
///
/// ## Variants of this function
///
/// - [`ext_cross`] / [`ext_cross_f`]: Returning a new tensor.
pub fn ext_cross<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Tensor<TC, B, DC>
where
    // dimension
    DA: DimAPI + DimMaxAPI<DB, Max = DC>,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtCrossAPI<TA, TB, TC, DC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC>,
{
    ext_cross_f(a, b, axis).rstsr_unwrap()
}

/// Cross product of 3-element vectors along one axis, promoting mixed operands to their common
/// dtype.
///
/// See also [`ext_cross`].
pub fn ext_cross_f<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TC, B, DC>>
where
    // dimension
    DA: DimAPI + DimMaxAPI<DB, Max = DC>,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtCrossAPI<TA, TB, TC, DC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC>,
{
    op_refa_refb_ext_cross(a, b, axis)
}

/// Device-level driver of promoting cross product, allocating the output; see also [`ext_cross`].
pub fn op_refa_refb_ext_cross<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TC, B, DC>>
where
    // dimension
    DA: DimAPI + DimMaxAPI<DB, Max = DC>,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtCrossAPI<TA, TB, TC, DC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC>,
{
    let (a, b) = (a.view(), b.view());

    // check devices
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // the array-API contract: the axis is size 3 in both operands, before broadcasting
    let ndim = a.ndim().max(b.ndim());
    let axis = cross_axis_in_output(axis.try_into().map_err(Into::into)?, ndim)?;
    let order = device.default_order();
    cross_check_axis(a.shape(), a.ndim(), axis, ndim, order)?;
    cross_check_axis(b.shape(), b.ndim(), axis, ndim, order)?;

    let (la, lb) = broadcast_layout(a.layout(), b.layout(), order)?;
    // the freshly allocated result follows the input's device default order
    let lc = match order {
        RowMajor => la.shape().c(),
        ColMajor => la.shape().f(),
    };
    let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
    device.ext_cross(storage_c.raw_mut(), &lc, a.raw(), &la, b.raw(), &lb, axis)?;
    // SAFETY: `device.ext_cross` above wrote every element of `lc`, covering the
    // fresh storage exactly.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, lc) }
}

/* #endregion */

/* #region ext_cross tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Cross product of 3-element vectors along one axis, promoting mixed operands to their common
    /// dtype.
    ///
    /// See also [`ext_cross`].
    pub fn ext_cross_f<TB, TC, DC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    ) -> Result<Tensor<TC, B, DC>>
    where
        // dimension
        D: DimAPI + DimMaxAPI<DB, Max = DC>,
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtCrossAPI<T, TB, TC, DC> + DeviceAPI<TB> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC>,
    {
        op_refa_refb_ext_cross(self.view(), b, axis)
    }

    /// Cross product of 3-element vectors along one axis, promoting mixed operands to their common
    /// dtype.
    ///
    /// See also [`ext_cross`].
    pub fn ext_cross<TB, TC, DC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    ) -> Tensor<TC, B, DC>
    where
        // dimension
        D: DimAPI + DimMaxAPI<DB, Max = DC>,
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtCrossAPI<T, TB, TC, DC> + DeviceAPI<TB> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC>,
    {
        op_refa_refb_ext_cross(self.view(), b, axis).rstsr_unwrap()
    }
}

/* #endregion */
