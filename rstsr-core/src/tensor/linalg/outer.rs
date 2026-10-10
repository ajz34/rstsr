use crate::prelude_dev::*;
use core::mem::transmute;

/* #region outer by function */

/// Outer product of two one-dimensional arrays.
///
/// Let $\mathbf{a}$ be a vector of length $N$ in `a` and $\mathbf{b}$ be a vector of length $M$ in
/// `b`. The outer product is the $N \times M$ array
///
/// $$C_{ij} = a_i b_j$$
///
/// without any conjugation. Both operands must be one-dimensional, following the Array-API
/// `linalg.outer` contract, and the result is always two-dimensional of shape $(N, M)$.
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Parameters
///
/// - `a`: impl [`TensorViewAPI`]
///
///   - The first input array; must be one-dimensional.
///
/// - `b`: impl [`TensorViewAPI`]
///
///   - The second input array; must be one-dimensional.
///
/// # Returns
///
/// [`Tensor<TA::Output, B, Ix2>`]
///
/// - The outer product, of shape `(a.size, b.size)`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 2, 3], &device);
/// let b = rt::tensor_from_nested!([4, 5], &device);
/// let c = rt::outer(&a, &b);
/// println!("{c}");
/// // [[ 4 5]
/// //  [ 8 10]
/// //  [ 12 15]]
/// # assert_eq!(format!("{c}"), "[[ 4 5]\n [ 8 10]\n [ 12 15]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `linalg.outer(x1, x2, /)` ([`linalg.outer`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.outer.html))
///   — this is the `xp.linalg` extension member.
/// - NumPy: `numpy.linalg.outer(x1, x2)`
///   ([`numpy.linalg.outer`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html))
///   — the same one-dimensional contract, raising `ValueError` for a non-vector operand.
///   The top-level `numpy.outer`
///   ([`numpy.outer`](https://numpy.org/doc/stable/reference/generated/numpy.outer.html))
///   instead flattens its inputs.
/// - RSTSR: `rt::outer(&a, &b)`, method `a.outer(&b)`.
///
/// # Panics
///
/// - Panics if either operand is not one-dimensional.
///
/// For a fallible version, use [`outer_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Array-API standard: [`linalg.outer`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.outer.html)
/// - NumPy: [`numpy.linalg.outer`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html)
///
/// ## Related functions in RSTSR
///
/// - [`ext_outer`]: the array-API-fulfilment form, which promotes mixed-dtype operands.
/// - [`tensordot`]: the general contraction, of which an outer product is the `axes = 0` case.
///
/// ## Variants of this function
///
/// - [`outer`] / [`outer_f`]: Returning a new tensor.
/// - [`outer_from`] / [`outer_from_f`]: Writing the result into an existing tensor.
pub fn outer<TA, TB, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Tensor<TA::Output, B, Ix2>
where
    TA: Mul<TB>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceOuterAPI<TA, TB, TA::Output>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    outer_f(a, b).rstsr_unwrap()
}

/// Outer product of two one-dimensional arrays.
///
/// See also [`outer`].
pub fn outer_f<TA, TB, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<Tensor<TA::Output, B, Ix2>>
where
    TA: Mul<TB>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceOuterAPI<TA, TB, TA::Output>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    op_refa_refb_outer(a, b)
}

/// Device-level driver of the outer product, allocating the output; see also [`outer`].
pub fn op_refa_refb_outer<TA, TB, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<Tensor<TA::Output, B, Ix2>>
where
    TA: Mul<TB>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceOuterAPI<TA, TB, TA::Output>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
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
    device.outer(storage_c.raw_mut(), &layout_c, a.raw(), &la, b.raw(), &lb)?;
    // SAFETY: `device.outer` above wrote every element of `layout_c`, covering
    // the fresh storage exactly.
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
}

/// Outer product of two one-dimensional arrays, written into an existing two-dimensional tensor.
///
/// The output `c` must already have shape `(a.size, b.size)` and is fully overwritten: this
/// function computes `c[i, j] = a[i] * b[j]` without accumulating onto `c`'s previous contents.
///
/// See also [`outer`].
pub fn outer_from<TA, TB, TC, DA, DB, B>(
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = Ix2>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) where
    TA: Mul<TB, Output = TC>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceOuterAPI<TA, TB, TC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    outer_from_f(c, a, b).rstsr_unwrap()
}

/// Outer product of two one-dimensional arrays, written into an existing two-dimensional tensor.
///
/// See also [`outer`].
pub fn outer_from_f<TA, TB, TC, DA, DB, B>(
    mut c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = Ix2>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
) -> Result<()>
where
    TA: Mul<TB, Output = TC>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceOuterAPI<TA, TB, TC> + DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    let (a, b, mut c) = (a.view(), b.view(), c.view_mut());
    // writing through a broadcast layout would alias elements
    rstsr_assert!(!c.layout().is_broadcasted(), InvalidLayout, "cannot write into broadcasted tensor")?;

    // check devices
    let device = c.device().clone();
    rstsr_assert!(device.same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // the array-API contract: both operands are one-dimensional
    rstsr_assert!(a.ndim() == 1 && b.ndim() == 1, InvalidValue, "outer expects one-dimensional operands")?;
    let la = a.layout().to_dim::<Ix1>()?;
    let lb = b.layout().to_dim::<Ix1>()?;
    let (n, m) = (la.shape()[0], lb.shape()[0]);
    rstsr_assert_eq!(c.shape(), &[n, m], InvalidLayout, "the outer-product output should have shape (a.len, b.len)")?;

    let c_layout = c.layout().clone();
    // SAFETY: `<B as DeviceRawAPI<TC>>::Raw` and `<B as DeviceRawAPI<MaybeUninit<TC>>>::Raw`
    // are the same buffer type (a container of `TC`-sized slots); `c` is an initialized,
    // writable output buffer, which the device op fully overwrites.
    let c_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<TC>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<TC>>>::Raw>(c.raw_mut())
    };
    device.outer(c_raw_mut, &c_layout, a.raw(), &la, b.raw(), &lb)
}

/* #endregion */

/* #region outer tensor trait */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Outer product of two one-dimensional tensors.
    ///
    /// See also [`outer`].
    pub fn outer_f<TB, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Result<Tensor<<T as Mul<TB>>::Output, B, Ix2>>
    where
        T: Mul<TB>,
        DB: DimAPI,
        B: DeviceOuterAPI<T, TB, <T as Mul<TB>>::Output>
            + DeviceAPI<T>
            + DeviceAPI<TB>
            + DeviceAPI<<T as Mul<TB>>::Output>
            + DeviceCreationAnyAPI<<T as Mul<TB>>::Output>,
    {
        op_refa_refb_outer(self.view(), b)
    }

    /// Outer product of two one-dimensional tensors.
    ///
    /// See also [`outer`].
    pub fn outer<TB, DB>(
        &self,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Tensor<<T as Mul<TB>>::Output, B, Ix2>
    where
        T: Mul<TB>,
        DB: DimAPI,
        B: DeviceOuterAPI<T, TB, <T as Mul<TB>>::Output>
            + DeviceAPI<T>
            + DeviceAPI<TB>
            + DeviceAPI<<T as Mul<TB>>::Output>
            + DeviceCreationAnyAPI<<T as Mul<TB>>::Output>,
    {
        op_refa_refb_outer(self.view(), b).rstsr_unwrap()
    }
}

/* #endregion */

/* #region outer_from tensor trait */

impl<R, T, B> TensorAny<R, T, B, Ix2>
where
    R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
{
    /// Outer product of two one-dimensional tensors, writing into this tensor.
    ///
    /// See also [`outer`].
    pub fn outer_from<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) where
        TA: Mul<TB, Output = T>,
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceOuterAPI<TA, TB, T> + DeviceAPI<TA> + DeviceAPI<TB>,
    {
        outer_from_f(self, a, b).rstsr_unwrap()
    }

    /// Outer product of two one-dimensional tensors, writing into this tensor.
    ///
    /// See also [`outer`].
    pub fn outer_from_f<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    ) -> Result<()>
    where
        TA: Mul<TB, Output = T>,
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceOuterAPI<TA, TB, T> + DeviceAPI<TA> + DeviceAPI<TB>,
    {
        outer_from_f(self, a, b)
    }
}

/* #endregion */

/* #region tests */

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_outer_matches_broadcast_mul() {
        // the array-API definition: outer(x1, x2) == x1[:, None] * x2[None, :]
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceCpuSerial::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
            let b = rt::tensor_from_nested!([4.0f64, 5.], &device);
            let a2 = rt::tensor_from_nested!([[1.0f64], [2.], [3.]], &device);
            let b2 = rt::tensor_from_nested!([[4.0f64, 5.]], &device);
            assert_eq!(format!("{}", rt::outer(&a, &b)), format!("{}", rt::mul(&a2, &b2)));
        }
    }

    #[test]
    fn test_outer_strided_operands() {
        // a non-contiguous (stride-2) first operand
        let device = DeviceCpuSerial::default();
        let a2 = rt::tensor_from_nested!([[1.0f64, 2.], [3., 4.], [5., 6.]], &device);
        let a = a2.i((.., 0));
        let b = rt::tensor_from_nested!([7.0f64, 8.], &device);
        let expected = rt::tensor_from_nested!([[7.0f64, 8.], [21., 24.], [35., 40.]], &device);
        assert_eq!(format!("{}", rt::outer(&a, &b)), format!("{expected}"));
    }

    #[test]
    fn test_outer_follows_device_default_order() {
        // the freshly allocated result's memory arrangement follows the input's
        // device default order, like its sibling linalg entries
        for (order, expect_c_contig) in [(RowMajor, true), (ColMajor, false)] {
            let mut device = DeviceCpuSerial::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
            let b = rt::tensor_from_nested!([4.0f64, 5.], &device);
            let c = rt::outer(&a, &b);
            assert_eq!(c.layout().c_contig(), expect_c_contig);
            assert_eq!(c.layout().f_contig(), !expect_c_contig);
            // and it agrees with tensordot(a, b, 0), which is the same math
            let sibling = rt::tensordot(&a, &b, 0);
            assert_eq!(c.layout().c_contig(), sibling.layout().c_contig());
        }
    }

    #[test]
    fn test_outer_requires_1d() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1.0f64, 2.], &device);
        let b2 = rt::tensor_from_nested!([[1.0f64, 2.]], &device);
        assert!(rt::outer_f(&a, &b2).is_err());
        assert!(rt::outer_f(&b2, &a).is_err());
    }

    #[test]
    fn test_outer_from_overwrites_output() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
        let b = rt::tensor_from_nested!([4.0f64, 5.], &device);
        // a pre-filled output is fully overwritten (outer has no accumulate form)
        let expected = rt::tensor_from_nested!([[4.0f64, 5.], [8., 10.], [12., 15.]], &device);

        let mut c: Tensor<f64, _, Ix2> = rt::ones(([3, 2], &device)).into_dim();
        rt::outer_from(&mut c, &a, &b);
        assert_eq!(format!("{c}"), format!("{expected}"));

        // the associated method is equivalent
        let mut c2: Tensor<f64, _, Ix2> = rt::zeros(([3, 2], &device)).into_dim();
        c2.outer_from(&a, &b);
        assert_eq!(format!("{c2}"), format!("{expected}"));
    }

    #[test]
    fn test_outer_from_rejects_bad_output() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
        let b = rt::tensor_from_nested!([4.0f64, 5.], &device);

        // wrong output shape
        let mut c: Tensor<f64, _, Ix2> = rt::zeros(([2, 3], &device)).into_dim();
        assert!(rt::outer_from_f(&mut c, &a, &b).is_err());

        // a broadcast (stride-0) output aliases elements; writing through it is rejected
        let ai = rt::arange((3, &device));
        let bi = rt::arange((2, &device));
        let mut c = Tensor::new(rt::arange((2, &device)).into_raw_parts().0, Layout::new([3, 2], [0, 1], 0).unwrap());
        assert!(rt::outer_from_f(c.view_mut(), &ai, &bi).is_err());
    }

    #[cfg(all(feature = "faer", feature = "rayon"))]
    #[test]
    fn test_outer_parallel_path() {
        // 60 x 40 = 2400 elements, above the rayon PARALLEL_SWITCH, so this
        // crosses the parallel kernel; grade it against the serial device.
        let device = DeviceFaer::default();
        let serial = DeviceCpuSerial::default();
        let a = rt::arange((60, &device));
        let b = rt::arange((40, &device));
        let a_s = rt::arange((60, &serial));
        let b_s = rt::arange((40, &serial));
        let c = rt::outer(&a, &b);
        assert_eq!(c.shape(), &[60, 40]);
        assert_eq!(format!("{c}"), format!("{}", rt::outer(&a_s, &b_s)));
    }
}

/* #endregion */
