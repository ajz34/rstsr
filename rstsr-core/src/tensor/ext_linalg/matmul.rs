//! Matrix multiplication with Array-API dtype promotion.

use crate::prelude_dev::*;
use num::{One, Zero};

/* #region ext_matmul by function */

/// Matrix multiplication of two arrays, promoting mixed-dtype operands to their common dtype.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The operands may have different dtypes: each pair is promoted to its common dtype
/// ([`DTypePromoteAPI`], the same rule as NumPy) and the product is accumulated in that
/// dtype. [`matmul`] instead requires the operands to share one dtype; this function is the
/// array-API-fulfilment form, like the `ext_*` element-wise arithmetic functions.
///
/// The shape rules are those of [`matmul`] (1-D / 2-D / stacked matrix products, with the same
/// broadcasting); only the dtype contract differs. As for [`matmul`], the matrix dimensions are
/// the last two axes under [`RowMajor`] and the first two under [`ColMajor`].
///
/// <div class="warning">
///
/// **Array-API Compliance Form**
///
/// This function exists only for array-API compliance, not as the idiomatic rstsr surface. Prefer
/// [`matmul`] whenever the operands already share a dtype.
///
/// </div>
///
/// # Parameters
///
/// - `a`: the left operand (views and owned tensors both accepted).
/// - `b`: the right operand (views and owned tensors both accepted).
///
/// # Returns
///
/// - [`Tensor<TC, B, DC>`][`Tensor`]: the matrix product in the promoted dtype, owning its data.
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
/// let c = rt::ext_matmul(&a, &b);
/// println!("{c}");
/// // [[ 19 22]
/// //  [ 43 50]]
/// # assert_eq!(format!("{c}"), "[[ 19 22]\n [ 43 50]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `matmul(x1, x2, /)` ([`matmul`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.matmul.html)),
///   with mixed-dtype operands.
/// - NumPy: `numpy.matmul(x1, x2)` ([`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html))
///   — NumPy computes the product in the promoted dtype.
/// - RSTSR: `rt::ext_matmul(&a, &b)`, method `a.ext_matmul(&b)`.
///
/// # Panics
///
/// - Panics if the operand shapes do not fit the [`matmul`] rule table.
///
/// For a fallible version, use [`ext_matmul_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`matmul`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.matmul.html)
/// - NumPy: [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
///
/// ## Related functions in RSTSR
///
/// - [`matmul`]: the same-dtype entry, without promotion.
/// - [`ext_mul`](crate::tensor::operators::exports::ext_mul()): element-wise product with the same
///   promotion rule.
///
/// ## Variants of this function
///
/// - [`ext_matmul_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::ext_matmul`] / [`TensorAny::ext_matmul_f`].
pub fn ext_matmul<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Tensor<TC, B, DC>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Zero + One,
    B: DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceExtMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_refa_refb_ext_matmul(a, b, TC::one()).rstsr_unwrap()
}

/// Matrix multiplication of two arrays, promoting mixed operands to their common dtype.
///
/// See also [`ext_matmul`].
pub fn ext_matmul_f<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<Tensor<TC, B, DC>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Zero + One,
    B: DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceExtMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_refa_refb_ext_matmul(a, b, TC::one())
}

/// Device-level driver of promoting matmul, allocating the output; see also [`ext_matmul`].
pub fn op_refa_refb_ext_matmul<TA, TB, TC, DA, DB, DC, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    alpha: TC,
) -> Result<Tensor<TC, B, DC>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceExtMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    let (a, b) = (a.view(), b.view());
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let default_order = a.device().default_order();
    let cfg = LayoutMatMulConfig::<DA, DB>::layout_matmul(a.layout(), b.layout(), default_order)?;
    let lc = cfg.lc;
    // fresh-output path: uninit storage + the write-only `ext_matmul_uninit`, then one
    // `assume_init` — the same shape as `op_refa_refb_matmul`.
    let device = a.device().clone();
    let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
    device.ext_matmul_uninit(storage_c.raw_mut(), &lc, a.raw(), a.layout(), b.raw(), b.layout(), alpha)?;
    // SAFETY: `ext_matmul_uninit` initialized every element of `lc` (its contract).
    let storage_c = unsafe { B::assume_init_impl(storage_c)? };
    Tensor::new_f(storage_c, lc)
}

/* #endregion */

/* #region ext_matmul tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Matrix multiplication of two tensors, promoting mixed operands to their common dtype.
    ///
    /// See also [`ext_matmul`].
    pub fn ext_matmul_f<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Result<Tensor<TC, B, DC>>
    where
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        TC: Zero + One,
        B: DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
        B: DeviceExtMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_refa_refb_ext_matmul(self.view(), rhs, TC::one())
    }

    /// Matrix multiplication of two tensors, promoting mixed operands to their common dtype.
    ///
    /// See also [`ext_matmul`].
    pub fn ext_matmul<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Tensor<TC, B, DC>
    where
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        TC: Zero + One,
        B: DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
        B: DeviceExtMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_refa_refb_ext_matmul(self.view(), rhs, TC::one()).rstsr_unwrap()
    }
}

/* #endregion */
