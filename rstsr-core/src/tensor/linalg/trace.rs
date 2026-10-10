use crate::prelude_dev::*;

/* #region helpers */

/// The `(offset, axis1, axis2)` diagonal of `tensor`, with the axes following
/// the device default order when the caller leaves them unset (last two under
/// row-major, first two under column-major).
fn trace_diagonal<'a, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    offset: impl Into<DiagonalArgs>,
) -> Result<TensorView<'a, T, B, D::SmallerOne>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    let DiagonalArgs { offset, axis1, axis2 } = offset.into();
    let (axis1_default, axis2_default) = match tensor.device().default_order() {
        RowMajor => (-2, -1),
        ColMajor => (0, 1),
    };
    let args = DiagonalArgs {
        offset,
        axis1: Some(axis1.unwrap_or(axis1_default)),
        axis2: Some(axis2.unwrap_or(axis2_default)),
    };
    diagonal_f(tensor, args)
}

/* #endregion */

/* #region trace by function */

/// Sum along the diagonal of a tensor.
///
/// Let $\mathbf{A}$ be the matrix spanned by the two axes of `tensor` selected by `offset`. The
/// trace is the sum over the diagonal shifted by `offset`,
///
/// $$\mathrm{tr}(\mathbf{A}) = \sum_i A_{i,\, i + \text{offset}}$$
///
/// The two selected axes are removed, so for a stacked input the result has shape
/// `tensor.shape()[..ndim - 2]`; a two-dimensional input gives a scalar-shaped (zero-dimensional)
/// result.
/// This function's **default axes follow the device default order**: the last two under
/// [`RowMajor`] and the first two under [`ColMajor`]. Explicit axes select the same diagonal under
/// either order.
///
/// # Parameters
///
/// - `tensor`: impl [`TensorViewAPI`]
///
///   - The input tensor; it must be at least two-dimensional.
///
/// - `offset`: impl [`Into<DiagonalArgs>`][DiagonalArgs]
///
///   - Which diagonal and over which axes; accepts `()` or `None` (main diagonal, default axes), an
///     integer offset (same axes), or `(offset, axis1, axis2)`.
///
/// # Returns
///
/// - [`Tensor<B::TOut, B, IxD>`][`Tensor`]: the summed diagonal. The element type is the backend's
///   accumulation type `B::TOut`, the same choice as [`sum`]; use [`trace_with_dtype`] to request a
///   specific dtype.
///
/// # Examples
///
/// Under a row-major device the trace of a stack is taken over each matrix's diagonal, collapsing
/// the last two axes:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
/// let t = rt::trace(&a, ());
/// println!("{t}");
/// // [ 5 13]
/// # let expected = rt::tensor_from_nested!([5, 13], &device);
/// # assert!(rt::allclose(&t, &expected, None));
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `linalg.trace(x, /, *, offset=0, dtype=None)` ([`linalg.trace`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.trace.html))
///   — under [`RowMajor`] the default axes are its last two; the `dtype` keyword is
///   [`trace_with_dtype`].
/// - NumPy: `numpy.trace(a, offset=0, axis1=0, axis2=1, ...)` ([`numpy.trace`](https://numpy.org/doc/stable/reference/generated/numpy.trace.html))
///   — the same operation; NumPy's default axes are the first two, matching the [`ColMajor`]
///   default here, while [`RowMajor`] (the NumPy-like order) uses the last two. Pass an explicit
///   `(offset, axis1, axis2)` to pin the axes.
/// - RSTSR: `rt::trace(&a, offset)`, method `a.trace(offset)`.
///
/// # Panics
///
/// - Panics if the input has fewer than two dimensions.
///
/// For a fallible version, use [`trace_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`linalg.trace`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.trace.html)
/// - NumPy: [`numpy.trace`](https://numpy.org/doc/stable/reference/generated/numpy.trace.html)
///
/// ## Related functions in RSTSR
///
/// - [`diagonal`]: the diagonal view that this function sums.
/// - [`sum`]: the reduction it is built from.
///
/// ## Variants of this function
///
/// - [`trace`] / [`trace_f`]: Returning a new tensor.
/// - [`trace_with_dtype`] / [`trace_with_dtype_f`]: Requesting an explicit output dtype (the
///   Array-API `dtype` keyword).
pub fn trace<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    offset: impl Into<DiagonalArgs>,
) -> Tensor<B::TOut, B, IxD>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: OpSumAPI<T, D::SmallerOne> + DeviceCreationAnyAPI<B::TOut>,
{
    trace_f(tensor, offset).rstsr_unwrap()
}

/// Sum along the diagonal of a tensor.
///
/// See also [`trace`].
pub fn trace_f<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    offset: impl Into<DiagonalArgs>,
) -> Result<Tensor<B::TOut, B, IxD>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: OpSumAPI<T, D::SmallerOne> + DeviceCreationAnyAPI<B::TOut>,
{
    let tensor = tensor.view();
    let diag = trace_diagonal(&tensor, offset)?;
    // `diagonal` appends the diagonal as the last axis
    sum_axes_f(diag, -1)
}

/// Sum along the diagonal of a tensor, accumulating in an explicit dtype.
///
/// Let $\mathbf{A}$ and the diagonal be as in [`trace`]. The sum is accumulated in `TOut`
/// (elements are cast inside the accumulation, so no cast copy of the input is materialized) —
/// the same rule as [`sum_with_dtype`], and the Array-API `dtype` keyword of `linalg.trace`.
///
/// See also [`trace`].
pub fn trace_with_dtype<T, TOut, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    offset: impl Into<DiagonalArgs>,
) -> Tensor<TOut, B, IxD>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: OpSumDtypeAPI<T, TOut, D::SmallerOne> + DeviceCreationAnyAPI<TOut>,
{
    trace_with_dtype_f(tensor, offset).rstsr_unwrap()
}

/// Sum along the diagonal of a tensor, accumulating in an explicit dtype.
///
/// See also [`trace`].
pub fn trace_with_dtype_f<T, TOut, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    offset: impl Into<DiagonalArgs>,
) -> Result<Tensor<TOut, B, IxD>>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: OpSumDtypeAPI<T, TOut, D::SmallerOne> + DeviceCreationAnyAPI<TOut>,
{
    let tensor = tensor.view();
    let diag = trace_diagonal(&tensor, offset)?;
    // `diagonal` appends the diagonal as the last axis
    sum_with_dtype_f(diag, -1)
}

/* #endregion */

/* #region trace tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Sum along the diagonal of a tensor.
    ///
    /// See also [`trace`].
    pub fn trace(&self, offset: impl Into<DiagonalArgs>) -> Tensor<B::TOut, B, IxD>
    where
        D: DimSmallerOneAPI,
        D::SmallerOne: DimAPI,
        B: OpSumAPI<T, D::SmallerOne> + DeviceCreationAnyAPI<B::TOut>,
    {
        trace(self, offset)
    }

    /// Sum along the diagonal of a tensor.
    ///
    /// See also [`trace`].
    pub fn trace_f(&self, offset: impl Into<DiagonalArgs>) -> Result<Tensor<B::TOut, B, IxD>>
    where
        D: DimSmallerOneAPI,
        D::SmallerOne: DimAPI,
        B: OpSumAPI<T, D::SmallerOne> + DeviceCreationAnyAPI<B::TOut>,
    {
        trace_f(self, offset)
    }

    /// Sum along the diagonal of a tensor, accumulating in an explicit dtype.
    ///
    /// See also [`trace`].
    pub fn trace_with_dtype<TOut>(&self, offset: impl Into<DiagonalArgs>) -> Tensor<TOut, B, IxD>
    where
        D: DimSmallerOneAPI,
        D::SmallerOne: DimAPI,
        B: OpSumDtypeAPI<T, TOut, D::SmallerOne> + DeviceCreationAnyAPI<TOut>,
    {
        trace_with_dtype(self, offset)
    }

    /// Sum along the diagonal of a tensor, accumulating in an explicit dtype.
    ///
    /// See also [`trace`].
    pub fn trace_with_dtype_f<TOut>(&self, offset: impl Into<DiagonalArgs>) -> Result<Tensor<TOut, B, IxD>>
    where
        D: DimSmallerOneAPI,
        D::SmallerOne: DimAPI,
        B: OpSumDtypeAPI<T, TOut, D::SmallerOne> + DeviceCreationAnyAPI<TOut>,
    {
        trace_with_dtype_f(self, offset)
    }
}

/* #endregion */

/* #region tests */

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_trace_matrix() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        // a two-dimensional input gives a zero-dimensional result (both axes
        // conventions coincide here)
        let t = rt::trace(&a, ());
        assert_eq!(t.ndim(), 0);
        assert_eq!(format!("{t}"), "5");
    }

    #[test]
    fn test_trace_offsets_and_stacks() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);
        assert_eq!(format!("{}", rt::trace(&a, ())), "15"); // 1 + 5 + 9
        assert_eq!(format!("{}", rt::trace(&a, 1)), "8"); // 2 + 6
        assert_eq!(format!("{}", rt::trace(&a, -1)), "12"); // 4 + 8
                                                            // an out-of-range offset yields an
                                                            // empty diagonal, i.e. a zero trace
        assert_eq!(format!("{}", rt::trace(&a, 5)), "0");

        // under row-major a stacked input collapses its last two axes
        let b = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        let t = rt::trace(&b, ());
        let expected = rt::tensor_from_nested!([5, 13], &device);
        assert!(rt::allclose(&t, &expected, None));

        // the associated method is equivalent
        assert!(rt::allclose(b.trace(()), &expected, None));
    }

    #[test]
    fn test_trace_follows_device_default_order() {
        // the default axes follow the device default order: the last two under
        // row-major, the first two under column-major
        let mut device_row = DeviceCpuSerial::default();
        device_row.set_default_order(RowMajor);
        let a_row = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device_row);
        let expected_row = rt::tensor_from_nested!([5, 13], &device_row); // (0,0)+(1,1) per
                                                                          // trailing axis
        assert!(rt::allclose(rt::trace(&a_row, ()), &expected_row, None));

        let mut device_col = DeviceCpuSerial::default();
        device_col.set_default_order(ColMajor);
        let a_col = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device_col);
        let expected_col = rt::tensor_from_nested!([8, 10], &device_col); // (0,0)+(1,1) per leading
                                                                          // axis
        assert!(rt::allclose(rt::trace(&a_col, ()), &expected_col, None));

        // explicit axes override the order-dependent default
        let trailing_on_col = rt::tensor_from_nested!([5, 13], &device_col);
        assert!(rt::allclose(rt::trace(&a_col, (0, -2, -1)), &trailing_on_col, None));
    }

    #[test]
    fn test_trace_explicit_axes() {
        let device = DeviceCpuSerial::default();
        // axes (0, 1) are the NumPy default; here they are requested explicitly.
        // the remaining axis (2) survives, so the result is [a[0,0,0] + a[1,1,0],
        // a[0,0,1] + a[1,1,1]] = [8, 10]
        let a = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        let t = rt::trace(&a, (0, 0, 1));
        let expected = rt::tensor_from_nested!([8, 10], &device);
        assert!(rt::allclose(&t, &expected, None));
    }

    #[test]
    fn test_trace_requires_2d() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        assert!(rt::trace_f(&a, ()).is_err());
    }

    #[test]
    fn test_trace_with_dtype() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[[1u8, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        // the default accumulation dtype is the input dtype
        let t: Tensor<u8, _, IxD> = rt::trace(&a, ());
        assert!(rt::allclose(&t, &rt::tensor_from_nested!([5u8, 13], &device), None));
        // an explicit accumulation dtype via the type parameter
        let t32: Tensor<u32, _, IxD> = rt::trace_with_dtype(&a, ());
        let expected = rt::tensor_from_nested!([5u32, 13], &device);
        assert!(rt::allclose(&t32, &expected, None));
        // the associated method is equivalent
        assert!(rt::allclose(a.trace_with_dtype::<u32>(()), &expected, None));
    }
}

/* #endregion */
