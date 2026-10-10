//! Tensor contraction with Array-API dtype promotion.

use crate::prelude_dev::*;
use crate::tensor::linalg::tensordot::{resolve_tensordot_axes, split_tensordot_free};

/* #region ext_tensordot by function */

/// Tensor contraction over specified axes, promoting mixed-dtype operands to their common dtype.
///
/// Contracts the axes of `a` given by `axes` with the corresponding axes of `b`; the result holds
/// the non-contracted axes of `a` (in order) followed by the non-contracted axes of `b` (in
/// order). No conjugation is applied, unlike [`ext_vecdot`]. The operands may have different
/// dtypes: each pair is promoted to its common dtype ([`DTypePromoteAPI`], the same rule as NumPy)
/// before the product.
///
/// [`tensordot`] instead requires the operands to share one dtype; this function is the
/// array-API-fulfilment form. The axes and shape rules are those of [`tensordot`].
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// <div class="warning">
///
/// **Array-API Compliance Form**
///
/// This function exists only for array-API compliance, not as the idiomatic rstsr surface. Prefer
/// [`tensordot`] whenever the operands already share a dtype.
///
/// </div>
///
/// <div class="warning">
///
/// **Efficiency Notice**
///
/// General axis pairs fall back to a naive kernel with no BLAS-backed path; `ext_tensordot` is
/// therefore not recommended when efficiency matters. Prefer
/// [`rt::tblis::einsum`](https://docs.rs/rstsr-tblis/latest/rstsr_tblis/einsum_impl/fn.einsum.html),
/// which requires the user to build and install the TBLIS library themselves.
///
/// </div>
///
/// # Parameters
///
/// - `a`, `b`: the input operands (views and owned tensors both accepted). Corresponding contracted
///   axes must have equal sizes.
/// - `axes`: the number of axes to contract, or the pair of explicit axis sequences (see
///   [`tensordot`] for the overloads). Default: `2`.
///
/// # Returns
///
/// - [`Tensor<TC, B, IxD>`][`Tensor`]: the contraction in the promoted dtype, owning its data.
///
/// # Examples
///
/// Mixed `u8` × `u16` operands give the promoted `u16` result:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1u8, 2], [3, 4]], &device);
/// let b = rt::tensor_from_nested!([[5u16, 6], [7, 8]], &device);
/// let c = rt::ext_tensordot(&a, &b, 1);
/// println!("{c}");
/// // [[ 19 22]
/// //  [ 43 50]]
/// # assert_eq!(format!("{c}"), "[[ 19 22]\n [ 43 50]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `tensordot(x1, x2, /, *, axes=2)` ([`tensordot`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.tensordot.html)),
///   with mixed-dtype operands.
/// - NumPy: `numpy.tensordot(a, b, axes=2)` ([`numpy.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html))
///   — NumPy computes the product in the promoted dtype.
/// - RSTSR: `rt::ext_tensordot(&a, &b, axes)`, method `a.ext_tensordot(&b, axes)`.
///
/// # Panics
///
/// - Panics if the paired contracted axes do not have equal sizes, or an integer `axes` is negative
///   or exceeds the dimensionality of either input.
///
/// For a fallible version, use [`ext_tensordot_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`tensordot`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.tensordot.html)
/// - NumPy: [`numpy.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html)
///
/// ## Related functions in RSTSR
///
/// - [`tensordot`]: the same-dtype entry, without promotion.
/// - [`ext_matmul`]: matrix product with the same promotion rule.
///
/// ## Variants of this function
///
/// - [`ext_tensordot_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::ext_tensordot`] /
///   [`TensorAny::ext_tensordot_f`].
pub fn ext_tensordot<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Tensor<TC, B, IxD>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtTensordotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>
        + DeviceRawAPI<MaybeUninit<TC>>,
{
    ext_tensordot_f(a, b, axes).rstsr_unwrap()
}

/// Tensor contraction over specified axes, promoting mixed operands to their common dtype.
///
/// See also [`ext_tensordot`].
pub fn ext_tensordot_f<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TC, B, IxD>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtTensordotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>
        + DeviceRawAPI<MaybeUninit<TC>>,
{
    let (a, b) = (a.view(), b.view());
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    let mut axes = axes.try_into().map_err(Into::into)?;
    if axes == AxesPairIndex::None {
        axes = AxesPairIndex::Val(2);
    }
    let (axes_a, axes_b) = resolve_tensordot_axes(&axes, a.ndim(), b.ndim())?;

    let (lam, lbm) = split_tensordot_free(a.layout(), &axes_a, b.layout(), &axes_b)?;

    let mut shape_c = lam.shape().clone();
    shape_c.extend_from_slice(lbm.shape());
    let layout_c = shape_c.new_contig(None, device.default_order());
    let mut storage_c = device.uninit_impl(layout_c.bounds_index()?.1)?;
    device.ext_tensordot(storage_c.raw_mut(), &layout_c, a.raw(), a.layout(), b.raw(), b.layout(), &axes_a, &axes_b)?;
    // SAFETY: `device.ext_tensordot` wrote every element of `layout_c`.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
}

/* #endregion */

/* #region ext_tensordot tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Tensor contraction over specified axes, promoting mixed operands to their common dtype.
    ///
    /// See also [`ext_tensordot`].
    pub fn ext_tensordot_f<TB, TC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Result<Tensor<TC, B, IxD>>
    where
        // dimension
        DB: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtTensordotAPI<T, TB, TC, D, DB, IxD>
            + DeviceAPI<TC>
            + DeviceCreationAnyAPI<TC>
            + DeviceRawAPI<MaybeUninit<TC>>,
    {
        ext_tensordot_f(self.view(), b, axes)
    }

    /// Tensor contraction over specified axes, promoting mixed operands to their common dtype.
    ///
    /// See also [`ext_tensordot`].
    pub fn ext_tensordot<TB, TC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Tensor<TC, B, IxD>
    where
        // dimension
        DB: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtTensordotAPI<T, TB, TC, D, DB, IxD>
            + DeviceAPI<TC>
            + DeviceCreationAnyAPI<TC>
            + DeviceRawAPI<MaybeUninit<TC>>,
    {
        ext_tensordot_f(self.view(), b, axes).rstsr_unwrap()
    }
}

/* #endregion */
