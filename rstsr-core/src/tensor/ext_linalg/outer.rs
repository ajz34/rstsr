//! Outer product with Array-API dtype promotion.

use crate::prelude_dev::*;

/* #region ext_outer by function */

/// Outer product of two one-dimensional arrays, promoting mixed-dtype operands to their common
/// dtype.
///
/// Let $\mathbf{a}$ be a vector of length $N$ in `a` and $\mathbf{b}$ be a vector of length $M$ in
/// `b`. The outer product is the $N \times M$ array
///
/// $$C_{ij} = a_i b_j$$
///
/// without any conjugation. The operands may have different dtypes: each pair is promoted to its
/// common dtype ([`DTypePromoteAPI`], the same rule as NumPy) before the product.
///
/// [`outer`] instead requires the operands to share one dtype; this function is the
/// array-API-fulfilment form, like the `ext_*` element-wise arithmetic functions. The shape
/// contract is that of [`outer`]: both operands must be one-dimensional.
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// <div class="warning">
///
/// **Array-API Compliance Form**
///
/// This function exists only for array-API compliance, not as the idiomatic rstsr surface. Prefer
/// [`outer`] whenever the operands already share a dtype.
///
/// </div>
///
/// # Parameters
///
/// - `a`: the first operand; must be one-dimensional (views and owned tensors both accepted).
/// - `b`: the second operand; must be one-dimensional (views and owned tensors both accepted).
///
/// # Returns
///
/// - [`Tensor<TC, B, Ix2>`][`Tensor`]: the outer product in the promoted dtype, of shape `(a.size,
///   b.size)`, owning its data.
///
/// # Examples
///
/// Mixed `u8` × `u16` operands give the promoted `u16` result:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1u8, 2, 3], &device);
/// let b = rt::tensor_from_nested!([4u16, 5], &device);
/// let c = rt::ext_outer(&a, &b);
/// println!("{c}");
/// // [[ 4 5]
/// //  [ 8 10]
/// //  [ 12 15]]
/// # assert_eq!(format!("{c}"), "[[ 4 5]\n [ 8 10]\n [ 12 15]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `linalg.outer(x1, x2, /)` ([`linalg.outer`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.outer.html)),
///   with mixed-dtype operands.
/// - NumPy: `numpy.linalg.outer(x1, x2)` ([`numpy.linalg.outer`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html))
///   — the same one-dimensional contract; the product is computed in the promoted dtype.
/// - RSTSR: `rt::ext_outer(&a, &b)`, method `a.ext_outer(&b)`.
///
/// # Panics
///
/// - Panics if either operand is not one-dimensional.
///
/// For a fallible version, use [`ext_outer_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`linalg.outer`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.outer.html)
/// - NumPy: [`numpy.linalg.outer`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html)
///
/// ## Related functions in RSTSR
///
/// - [`outer`]: the same-dtype entry, without promotion.
/// - [`ext_matmul`]: matrix product with the same promotion rule.
///
/// ## Variants of this function
///
/// - [`ext_outer_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::ext_outer`] / [`TensorAny::ext_outer_f`].
pub fn ext_outer<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Tensor<TC, B, Ix2>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtOuterAPI<TA, TB, TC> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
{
    ext_outer_f(a, b).rstsr_unwrap()
}

/// Outer product of two one-dimensional arrays, promoting mixed operands to their common dtype.
///
/// See also [`ext_outer`].
pub fn ext_outer_f<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<Tensor<TC, B, Ix2>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtOuterAPI<TA, TB, TC> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
{
    op_refa_refb_ext_outer(a, b)
}

/// Device-level driver of promoting outer product, allocating the output; see also [`ext_outer`].
pub fn op_refa_refb_ext_outer<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<Tensor<TC, B, Ix2>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtOuterAPI<TA, TB, TC> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
{
    let (a, b) = (a.view(), b.view());

    // check devices
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // the array-API contract: both operands are one-dimensional
    rstsr_assert!(a.ndim() == 1 && b.ndim() == 1, InvalidValue, "outer expects one-dimensional operands")?;
    let la = a.layout().to_dim::<Ix1>()?;
    let lb = b.layout().to_dim::<Ix1>()?;
    let (n, m) = (la.shape()[0], lb.shape()[0]);

    // the freshly allocated result follows the input's device default order
    let layout_c = match device.default_order() {
        RowMajor => [n, m].c(),
        ColMajor => [n, m].f(),
    };
    let mut storage_c = device.uninit_impl(layout_c.bounds_index()?.1)?;
    device.ext_outer(storage_c.raw_mut(), &layout_c, a.raw(), &la, b.raw(), &lb)?;
    // SAFETY: `device.ext_outer` above wrote every element of `layout_c`, covering
    // the fresh storage exactly.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
}

/* #endregion */

/* #region ext_outer tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Outer product of two one-dimensional tensors, promoting mixed operands to their common
    /// dtype.
    ///
    /// See also [`ext_outer`].
    pub fn ext_outer_f<TB, TC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Result<Tensor<TC, B, Ix2>>
    where
        // dimension
        DB: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtOuterAPI<T, TB, TC> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
    {
        op_refa_refb_ext_outer(self.view(), b)
    }

    /// Outer product of two one-dimensional tensors, promoting mixed operands to their common
    /// dtype.
    ///
    /// See also [`ext_outer`].
    pub fn ext_outer<TB, TC, DB>(&self, b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>) -> Tensor<TC, B, Ix2>
    where
        // dimension
        DB: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtOuterAPI<T, TB, TC> + DeviceAPI<TC> + DeviceCreationAnyAPI<TC> + DeviceRawAPI<MaybeUninit<TC>>,
    {
        op_refa_refb_ext_outer(self.view(), b).rstsr_unwrap()
    }
}

/* #endregion */
