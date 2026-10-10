//! Vector dot product with Array-API dtype promotion.

use crate::prelude_dev::*;

/* #region ext_vecdot by function */

/// Vector dot product of two arrays, promoting mixed-dtype operands to their common dtype.
///
/// Let $\mathbf{a}$ be a vector in `a` and $\mathbf{b}$ be a corresponding vector in `b`. The dot
/// product is defined as:
///
/// $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} \overline{a_i}b_i$$
///
/// where the sum is over the contracted axes and where $\overline{a_i}$ denotes the complex
/// conjugate if $a_i$ is complex and the identity otherwise. The operands may have different
/// dtypes: each pair is promoted to its common dtype ([`DTypePromoteAPI`], the same rule as NumPy)
/// before the product, and the first operand is conjugated in that promoted dtype.
///
/// [`vecdot`] instead requires the operands to share one dtype; this function is the
/// array-API-fulfilment form, like the `ext_*` element-wise arithmetic functions. The axes and
/// broadcasting rules are those of [`vecdot`].
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// <div class="warning">
///
/// **Array-API Compliance Form**
///
/// This function exists only for array-API compliance, not as the idiomatic rstsr surface. Prefer
/// [`vecdot`] whenever the operands already share a dtype.
///
/// </div>
///
/// # Parameters
///
/// - `a`: the first operand (conjugated; views and owned tensors both accepted).
/// - `b`: the second operand (views and owned tensors both accepted).
/// - `axes_pair`: the axis or axes over which to compute the dot product (see [`vecdot`] for the
///   overloads). Default: `-1`.
///
/// # Returns
///
/// - [`Tensor<TC, B, IxD>`][`Tensor`]: the dot product in the promoted dtype, owning its data.
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
/// let b = rt::tensor_from_nested!([4u16, 5, 6], &device);
/// let c = rt::ext_vecdot(&a, &b, None);
/// println!("{c}");
/// // 32
/// # assert_eq!(format!("{c}"), "32");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `vecdot(x1, x2, /, *, axis=-1)` ([`vecdot`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.vecdot.html)),
///   with mixed-dtype operands.
/// - NumPy: `numpy.vecdot(x1, x2, axis=-1)` ([`numpy.vecdot`](https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html))
///   — NumPy computes the product in the promoted dtype.
/// - RSTSR: `rt::ext_vecdot(&a, &b, axes_pair)`, method `a.ext_vecdot(&b, axes_pair)`.
///
/// # Panics
///
/// - Panics if the contracted axis dimensions do not match, or the non-contracted parts cannot be
///   broadcast together.
///
/// For a fallible version, use [`ext_vecdot_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`vecdot`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.vecdot.html)
/// - NumPy: [`numpy.vecdot`](https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html)
///
/// ## Related functions in RSTSR
///
/// - [`vecdot`]: the same-dtype entry, without promotion.
/// - [`ext_matmul`]: matrix product with the same promotion rule.
///
/// ## Variants of this function
///
/// - [`ext_vecdot_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::ext_vecdot`] / [`TensorAny::ext_vecdot_f`].
pub fn ext_vecdot<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes_pair: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Tensor<TC, B, IxD>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtVecdotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>
        + DeviceRawAPI<MaybeUninit<TC>>,
{
    ext_vecdot_f(a, b, axes_pair).rstsr_unwrap()
}

/// Vector dot product of two arrays, promoting mixed operands to their common dtype.
///
/// See also [`ext_vecdot`].
pub fn ext_vecdot_f<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes_pair: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TC, B, IxD>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtVecdotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>
        + DeviceRawAPI<MaybeUninit<TC>>,
{
    op_refa_refb_ext_vecdot(a, b, axes_pair)
}

/// Device-level driver of promoting vecdot, allocating the output; see also [`ext_vecdot`].
pub fn op_refa_refb_ext_vecdot<TA, TB, TC, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axes_pair: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<TC, B, IxD>>
where
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: DTypePromoteAPI<TB, Res = TC>,
    B: DeviceExtVecdotAPI<TA, TB, TC, DA, DB, IxD>
        + DeviceAPI<TC>
        + DeviceCreationAnyAPI<TC>
        + DeviceRawAPI<MaybeUninit<TC>>,
{
    let (a, b) = (a.view(), b.view());

    // check devices
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // check axis
    let mut axes_pair = axes_pair.try_into().map_err(Into::into)?;
    if axes_pair == AxesPairIndex::None {
        axes_pair = AxesPairIndex::Val(-1);
    }

    let (axes_a, axes_b) = match axes_pair {
        AxesPairIndex::None => unreachable!("already handled above"),
        AxesPairIndex::Val(axis) => {
            if axis < 0 {
                rstsr_pattern!(
                    axis,
                    -(a.ndim().min(b.ndim()) as isize)..=-1,
                    InvalidValue,
                    "axis should be [-N, -1] where N is min(a.ndim, b.ndim)"
                )?;
                let axis_a = axis + a.ndim() as isize;
                let axis_b = axis + b.ndim() as isize;
                (vec![axis_a], vec![axis_b])
            } else {
                rstsr_pattern!(
                    axis,
                    0..(a.ndim().min(b.ndim()) as isize),
                    InvalidValue,
                    "axis should be [0, N) where N is min(a.ndim, b.ndim)"
                )?;
                (vec![axis], vec![axis])
            }
        },
        AxesPairIndex::Pair(axes_a, axes_b) => {
            let axes_a = normalize_axes_index(axes_a, a.ndim(), false, false)?;
            let axes_b = normalize_axes_index(axes_b, b.ndim(), false, false)?;
            rstsr_assert_eq!(
                axes_a.len(),
                axes_b.len(),
                InvalidValue,
                "axes_a and axes_b should have the same length"
            )?;
            (axes_a, axes_b)
        },
    };

    let (las, lam) = a.layout().dim_split_axes(&axes_a)?;
    let (lbs, lbm) = b.layout().dim_split_axes(&axes_b)?;

    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;

    let default_order = a.device().default_order();
    let (lam_b, lbm_b) = broadcast_layout(&lam, &lbm, default_order)?;
    // generate output layout
    let layout_c = match TensorIterOrder::default() {
        TensorIterOrder::C => lam_b.shape().c(),
        TensorIterOrder::F => lam_b.shape().f(),
        _ => get_layout_for_binary_op(&lam_b, &lbm_b, default_order)?,
    };
    let mut storage_c = device.uninit_impl(layout_c.bounds_index()?.1)?;
    device.ext_vecdot(storage_c.raw_mut(), &layout_c, a.raw(), a.layout(), b.raw(), b.layout(), &axes_a, &axes_b)?;
    // SAFETY: `device.ext_vecdot` above wrote every element of `layout_c`, covering
    // the fresh storage exactly.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
}

/* #endregion */

/* #region ext_vecdot tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Vector dot product of two tensors, promoting mixed operands to their common dtype.
    ///
    /// See also [`ext_vecdot`].
    pub fn ext_vecdot_f<TB, TC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes_pair: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Result<Tensor<TC, B, IxD>>
    where
        // dimension
        DB: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtVecdotAPI<T, TB, TC, D, DB, IxD>
            + DeviceAPI<TC>
            + DeviceCreationAnyAPI<TC>
            + DeviceRawAPI<MaybeUninit<TC>>,
    {
        op_refa_refb_ext_vecdot(self.view(), b, axes_pair)
    }

    /// Vector dot product of two tensors, promoting mixed operands to their common dtype.
    ///
    /// See also [`ext_vecdot`].
    pub fn ext_vecdot<TB, TC, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
        axes_pair: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    ) -> Tensor<TC, B, IxD>
    where
        // dimension
        DB: DimAPI,
        // operation specific
        T: DTypePromoteAPI<TB, Res = TC>,
        B: DeviceExtVecdotAPI<T, TB, TC, D, DB, IxD>
            + DeviceAPI<TC>
            + DeviceCreationAnyAPI<TC>
            + DeviceRawAPI<MaybeUninit<TC>>,
    {
        op_refa_refb_ext_vecdot(self.view(), b, axes_pair).rstsr_unwrap()
    }
}

/* #endregion */
