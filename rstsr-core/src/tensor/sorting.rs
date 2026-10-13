//! Sorting tensor API: [`sort`], [`argsort`], and the `_custom` comparator
//! variants, with argument type [`SortArgs`].

use core::cmp::Ordering;

use num::Complex;

use crate::prelude_dev::*;

/* #region argsort */

/// Returns the indices that sort a tensor along an axis.
///
/// See also [`argsort`].
pub fn argsort_f<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
) -> Result<Tensor<usize, B, IxD>>
where
    D: DimAPI,
    T: 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpArgSortAPI<T, D>,
{
    let args = args.into();
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(args.axis, tensor.ndim())?;
    decline_complex_sort::<T>()?;
    let (storage, layout) = device.argsort_axes(tensor.raw(), tensor.layout(), axis, args.descending, args.stable)?;
    Tensor::new_f(storage, layout)
}

/// Complex dtypes are declined at the tensor layer (array-api restricts
/// sort/argsort to real-valued data types; complex ordering is available
/// through [`sort_custom`]). Runtime gate via TypeId, the same technique as
/// DeviceFaer's matmul dispatch.
fn decline_complex_sort<T: 'static>() -> Result<()> {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<Complex<f32>>()
        || core::any::TypeId::of::<T>() == core::any::TypeId::of::<Complex<f64>>()
    {
        return rstsr_raise!(
            UnImplemented,
            "sort/argsort do not support complex data type; use sort_custom with an explicit comparator."
        );
    }
    Ok(())
}

/// Returns the indices that sort a tensor along an axis.
///
/// The returned indices are of dtype [`usize`] and have the same shape as the
/// input; gathering with them along `axis` reproduces [`sort`]. Ties keep the
/// input order (stable).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - `args`: any [`SortArgs`] conversion:
///
///   - `()` / `None`: defaults (`axis = -1`, ascending, stable).
///   - `axis` (`isize`): sort along that axis (negative counts from the back).
///   - `(axis, descending)` / `(axis, descending, stable)`: explicit forms.
///   - `descending = true` reverses the value comparison; ties (and NaN, see below) keep their
///     input order.
///
/// # Returns
///
/// - [`Tensor<usize, B, IxD>`][`Tensor`]
///
///   - Indices along `axis` such that gathering the input with them yields the sorted tensor. NaN
///     is treated as greater than every other value (last in both ascending and descending order),
///     and equal values (including `-0.0` and `0.0`) keep input order.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 2], &device);
/// println!("{}", rt::argsort(&a, None));
/// // [ 1 2 0]
/// # assert_eq!(format!("{}", rt::argsort(&a, None)), "[ 1 2 0]");
/// ```
///
/// # Overload List
///
/// `args` accepts any [`SortArgs`] conversion:
///
/// - `()` / `None`: defaults (`axis = -1`, ascending, stable).
/// - `axis` (integer): sort along that axis (negative counts from the back).
/// - `descending` (`bool`): sort descending along the last axis.
/// - `(axis, descending)`.
/// - `(axis, descending, stable)` (`stable` is accepted for parity; the sort is always stable).
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 2], &device);
/// println!("{}", rt::argsort(&a, None));             // defaults
/// // [ 1 2 0]
/// println!("{}", rt::argsort(&a, 0));                // integer axis
/// // [ 1 2 0]
/// println!("{}", rt::argsort(&a, true));             // descending
/// // [ 0 2 1]
/// println!("{}", rt::argsort(&a, (0, true)));        // axis + descending
/// // [ 0 2 1]
/// println!("{}", rt::argsort(&a, (0, false, true))); // axis + ascending + stable
/// // [ 1 2 0]
/// # assert_eq!(format!("{}", rt::argsort(&a, None)), "[ 1 2 0]");
/// # assert_eq!(format!("{}", rt::argsort(&a, true)), "[ 0 2 1]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `argsort(x, /, *, axis=-1, descending=False, stable=True)` ([`argsort`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.argsort.html))
/// - NumPy: `numpy.argsort(a, axis=-1)` ([`numpy.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html))
/// - RSTSR: `rt::argsort(tensor, args)`
///
/// Deviation from NumPy: NumPy's `kind` parameter is not supported (the sort
/// is always stable). Complex dtypes are declined: the array-api standard
/// scopes `argsort` to real-valued dtypes, whereas NumPy accepts complex and
/// sorts lexicographically (real part first) — use [`argsort_custom`] with
/// the [`ExtSortCmp`] comparator for NumPy-parity complex ordering.
///
/// # Panics
///
/// - Panics if the axis is out of range, or if the device does not implement sorting for the dtype
///   (complex and other non-[`ExtSortCmp`] dtypes).
///
/// For a fallible version, use [`argsort_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`argsort`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.argsort.html)
/// - NumPy: [`numpy.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)
///
/// ## Related functions in RSTSR
///
/// - [`sort`]: the sorted values themselves.
/// - [`argsort_custom`]: user-comparator variant.
///
/// ## Variants of this function
///
/// - [`argsort_f`]: fallible version.
/// - [`argsort_custom`] / [`argsort_custom_f`]: user comparator.
/// - [`TensorAny::argsort`]: associated method.
/// - [`TensorAny::argsort_f`]: associated fallible method.
pub fn argsort<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
) -> Tensor<usize, B, IxD>
where
    D: DimAPI,
    T: 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpArgSortAPI<T, D>,
{
    argsort_f(tensor, args).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpArgSortAPI<T, D>,
{
    /// Returns the indices that sort a tensor along an axis.
    ///
    /// See also [`argsort`].
    pub fn argsort_f<AArg>(&self, args: AArg) -> Result<Tensor<usize, B, IxD>>
    where
        AArg: Into<SortArgs>,
    {
        argsort_f(self, args)
    }

    /// Returns the indices that sort a tensor along an axis.
    ///
    /// See also [`argsort`].
    pub fn argsort<AArg>(&self, args: AArg) -> Tensor<usize, B, IxD>
    where
        AArg: Into<SortArgs>,
    {
        argsort_f(self, args).rstsr_unwrap()
    }
}

/* #endregion */

/* #region sort */

/// Sort a tensor along an axis.
///
/// See also [`sort`].
pub fn sort_f<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
) -> Result<Tensor<T, B, IxD>>
where
    D: DimAPI,
    T: 'static,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortAPI<T, D>,
{
    let args = args.into();
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(args.axis, tensor.ndim())?;
    decline_complex_sort::<T>()?;
    let (storage, layout) = device.sort_axes(tensor.raw(), tensor.layout(), axis, args.descending, args.stable)?;
    Tensor::new_f(storage, layout)
}

/// Sort a tensor along an axis.
///
/// Values are sorted with the total order of [`ExtSortCmp`]: NaN (or a NaN
/// component, for complex) is treated as greater than every other value —
/// last in both ascending and descending order — and values equal under `==`
/// (including `-0.0` and `0.0`) keep their input order (stable).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - `args`: any [`SortArgs`] conversion — see [`argsort`] for the accepted forms.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]
///
///   - A new owned tensor with the same shape as the input; elements along `axis` are sorted.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 2], &device);
/// println!("{}", rt::sort(&a, None));
/// // [ 1 2 3]
/// # assert_eq!(format!("{}", rt::sort(&a, None)), "[ 1 2 3]");
/// ```
///
/// Sorting a 2-D tensor along `axis = 0` and along the last axis:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let b = rt::tensor_from_nested!([[9, 1, 5], [2, 8, 3]], &device);
/// println!("{}", rt::sort(&b, 0));
/// // [[ 2 1 3]
/// //  [ 9 8 5]]
/// println!("{}", rt::sort(&b, None));
/// // [[ 1 5 9]
/// //  [ 2 3 8]]
/// # assert_eq!(format!("{}", rt::sort(&b, 0)), "[[ 2 1 3]\n [ 9 8 5]]");
/// # assert_eq!(format!("{}", rt::sort(&b, None)), "[[ 1 5 9]\n [ 2 3 8]]");
/// ```
///
/// # Overload List
///
/// `args` accepts any [`SortArgs`] conversion:
///
/// - `()` / `None`: defaults (`axis = -1`, ascending, stable).
/// - `axis` (integer): sort along that axis (negative counts from the back).
/// - `descending` (`bool`): sort descending along the last axis.
/// - `(axis, descending)`.
/// - `(axis, descending, stable)`.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 2], &device);
/// println!("{}", rt::sort(&a, None));            // defaults
/// // [ 1 2 3]
/// println!("{}", rt::sort(&a, 0));               // integer axis
/// // [ 1 2 3]
/// println!("{}", rt::sort(&a, true));            // descending
/// // [ 3 2 1]
/// println!("{}", rt::sort(&a, (0, true)));       // axis + descending
/// // [ 3 2 1]
/// println!("{}", rt::sort(&a, (0, true, true))); // axis + descending + stable
/// // [ 3 2 1]
/// # assert_eq!(format!("{}", rt::sort(&a, None)), "[ 1 2 3]");
/// # assert_eq!(format!("{}", rt::sort(&a, true)), "[ 3 2 1]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `sort(x, /, *, axis=-1, descending=False, stable=True)` ([`sort`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.sort.html))
/// - NumPy: `numpy.sort(a, axis=-1)` ([`numpy.sort`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html))
/// - RSTSR: `rt::sort(tensor, args)`
///
/// Deviation from NumPy: NumPy's `kind` parameter is not supported (the sort
/// is always stable); NaN orders last in descending sorts as well (NumPy
/// 2.x behavior). Complex dtypes are declined (the array-api standard scopes
/// `sort` to real-valued dtypes), whereas NumPy accepts complex and sorts
/// lexicographically — use [`sort_custom`] with the [`ExtSortCmp`]
/// comparator for NumPy-parity complex ordering.
///
/// # Panics
///
/// - Panics if the axis is out of range, or if the device does not implement sorting for the dtype
///   (complex and other non-[`ExtSortCmp`] dtypes).
///
/// For a fallible version, use [`sort_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`sort`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.sort.html)
/// - NumPy: [`numpy.sort`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
///
/// ## Related functions in RSTSR
///
/// - [`argsort`]: the sorting permutation instead of the values.
/// - [`sort_custom`]: user-comparator variant.
///
/// ## Variants of this function
///
/// - [`sort_f`]: fallible version.
/// - [`sort_custom`] / [`sort_custom_f`]: user comparator.
/// - [`TensorAny::sort`]: associated method.
/// - [`TensorAny::sort_f`]: associated fallible method.
pub fn sort<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
) -> Tensor<T, B, IxD>
where
    D: DimAPI,
    T: 'static,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortAPI<T, D>,
{
    sort_f(tensor, args).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: 'static,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortAPI<T, D>,
{
    /// Sort a tensor along an axis.
    ///
    /// See also [`sort`].
    pub fn sort_f<AArg>(&self, args: AArg) -> Result<Tensor<T, B, IxD>>
    where
        AArg: Into<SortArgs>,
    {
        sort_f(self, args)
    }

    /// Sort a tensor along an axis.
    ///
    /// See also [`sort`].
    pub fn sort<AArg>(&self, args: AArg) -> Tensor<T, B, IxD>
    where
        AArg: Into<SortArgs>,
    {
        sort_f(self, args).rstsr_unwrap()
    }
}

/* #endregion */

/* #region sort_custom / argsort_custom */

/// Sort a tensor along an axis with a user comparator.
///
/// See also [`sort_custom`].
pub fn sort_custom_f<T, B, D, F>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
    f: F,
) -> Result<Tensor<T, B, IxD>>
where
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortCustomAPI<T, D>,
{
    let args = args.into();
    let descending = args.descending;
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(args.axis, tensor.ndim())?;
    // `stable` is accepted for parity with `sort` (the custom kernel is always
    // stable); `descending` reverses the comparator outcome.
    let f = move |a: &T, b: &T| if descending { f(a, b).reverse() } else { f(a, b) };
    let (storage, layout) = device.sort_axes_custom(tensor.raw(), tensor.layout(), axis, f)?;
    Tensor::new_f(storage, layout)
}

/// Sort a tensor along an axis with a user-supplied comparator
/// `f: Fn(&T, &T) -> Ordering`.
///
/// The comparator fully decides the order; ties under `f` keep input order (the
/// kernel is always stable). This is the escape hatch for orderings rstsr does
/// not ship — for example, complex values compared by squared magnitude.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `args`: any [`SortArgs`] conversion; see [`sort`] for the accepted forms.
/// - `f`: the comparator; must be a consistent total order for correct results.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: a new owned tensor with the input's shape; elements along
///   `axis` are ordered by `f`.
///
/// # Examples
///
/// Complex values ordered by squared magnitude:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # use core::cmp::Ordering;
/// # use num::Complex;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::asarray((
///     vec![Complex::new(3.0, 0.0), Complex::new(1.0, 1.0), Complex::new(0.0, 2.0)],
///     &device,
/// ));
/// let by_norm = |x: &Complex<f64>, y: &Complex<f64>| {
///     let n1 = x.re * x.re + x.im * x.im;
///     let n2 = y.re * y.re + y.im * y.im;
///     n1.partial_cmp(&n2).unwrap_or(Ordering::Equal)
/// };
/// println!("{}", rt::sort_custom(&a, None, by_norm));
/// // [ 1+1i 0+2i 3+0i]
/// # assert_eq!(format!("{}", rt::sort_custom(&a, None, by_norm)), "[ 1+1i 0+2i 3+0i]");
/// ```
///
/// # Overload List
///
/// `args` accepts the same [`SortArgs`] forms as [`sort`]: `()` / `None` (defaults), an integer
/// `axis`, a `bool` `descending`, `(axis, descending)`, or `(axis, descending, stable)`.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # use core::cmp::Ordering;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([5, 1, 4, 2, 3], &device);
/// let by_mod3 = |x: &i32, y: &i32| (x % 3).cmp(&(y % 3));
/// println!("{}", rt::sort_custom(&a, None, by_mod3)); // defaults
/// // [ 3 1 4 5 2]
/// let b = rt::tensor_from_nested!([[3, 1, 2], [6, 4, 5]], &device);
/// let by_mod2 = |x: &i32, y: &i32| (x % 2).cmp(&(y % 2));
/// println!("{}", rt::sort_custom(&b, 0, by_mod2)); // explicit axis
/// // [[ 6 4 2]
/// //  [ 3 1 5]]
/// println!("{}", rt::sort_custom(&b, (0, true), by_mod2)); // axis + descending
/// // [[ 3 1 5]
/// //  [ 6 4 2]]
/// # assert_eq!(format!("{}", rt::sort_custom(&a, None, by_mod3)), "[ 3 1 4 5 2]");
/// # assert_eq!(format!("{}", rt::sort_custom(&b, 0, by_mod2)), "[[ 6 4 2]\n [ 3 1 5]]");
/// # assert_eq!(format!("{}", rt::sort_custom(&b, (0, true), by_mod2)), "[[ 3 1 5]\n [ 6 4 2]]");
/// ```
///
/// # Panics
///
/// - Panics if the axis is out of range.
///
/// For a fallible version, use [`sort_custom_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`sort`]: the built-in order ([`ExtSortCmp`]).
/// - [`argsort_custom`]: the permutation instead of the values.
///
/// ## Variants of this function
///
/// - [`sort_custom_f`]: fallible version.
/// - [`TensorAny::sort_custom`]: associated method.
/// - [`TensorAny::sort_custom_f`]: associated fallible method.
pub fn sort_custom<T, B, D, F>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
    f: F,
) -> Tensor<T, B, IxD>
where
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortCustomAPI<T, D>,
{
    sort_custom_f(tensor, args, f).rstsr_unwrap()
}

/// Returns the indices that sort a tensor along an axis with a user comparator.
///
/// See also [`argsort_custom`].
pub fn argsort_custom_f<T, B, D, F>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
    f: F,
) -> Result<Tensor<usize, B, IxD>>
where
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSortCustomAPI<T, D>,
{
    let args = args.into();
    let descending = args.descending;
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(args.axis, tensor.ndim())?;
    // `stable` is accepted for parity with `argsort` (the custom kernel is
    // always stable); `descending` reverses the comparator outcome.
    let f = move |a: &T, b: &T| if descending { f(a, b).reverse() } else { f(a, b) };
    let (storage, layout) = device.argsort_axes_custom(tensor.raw(), tensor.layout(), axis, f)?;
    Tensor::new_f(storage, layout)
}

/// Returns the indices that sort a tensor along an axis with a user-supplied
/// comparator `f: Fn(&T, &T) -> Ordering` (ties keep input order).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `args`: any [`SortArgs`] conversion; see [`sort`] for the accepted forms.
/// - `f`: the comparator; must be a consistent total order for correct results.
///
/// # Returns
///
/// - [`Tensor<usize, B, IxD>`][`Tensor`]: positions within `axis` such that gathering the input
///   with them reproduces [`sort_custom`].
///
/// # Examples
///
/// Indices that order the input by residue mod 2:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 2, 0], &device);
/// let by_mod2 = |x: &i32, y: &i32| (x % 2).cmp(&(y % 2));
/// println!("{}", rt::argsort_custom(&a, None, by_mod2));
/// // [ 2 3 0 1]
/// # assert_eq!(format!("{}", rt::argsort_custom(&a, None, by_mod2)), "[ 2 3 0 1]");
/// ```
///
/// # Overload List
///
/// `args` accepts the same [`SortArgs`] forms as [`sort`]: `()` / `None` (defaults), an integer
/// `axis`, a `bool` `descending`, `(axis, descending)`, or `(axis, descending, stable)`.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 2, 0], &device);
/// let by_mod2 = |x: &i32, y: &i32| (x % 2).cmp(&(y % 2));
/// println!("{}", rt::argsort_custom(&a, 0, by_mod2)); // explicit axis
/// // [ 2 3 0 1]
/// println!("{}", rt::argsort_custom(&a, (0, true), by_mod2)); // axis + descending
/// // [ 0 1 2 3]
/// # assert_eq!(format!("{}", rt::argsort_custom(&a, 0, by_mod2)), "[ 2 3 0 1]");
/// # assert_eq!(format!("{}", rt::argsort_custom(&a, (0, true), by_mod2)), "[ 0 1 2 3]");
/// ```
///
/// # Panics
///
/// - Panics if the axis is out of range.
///
/// For a fallible version, use [`argsort_custom_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`argsort`]: the built-in order ([`ExtSortCmp`]).
/// - [`sort_custom`]: the values instead of the permutation.
///
/// ## Variants of this function
///
/// - [`argsort_custom_f`]: fallible version.
/// - [`TensorAny::argsort_custom`]: associated method.
/// - [`TensorAny::argsort_custom_f`]: associated fallible method.
pub fn argsort_custom<T, B, D, F>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    args: impl Into<SortArgs>,
    f: F,
) -> Tensor<usize, B, IxD>
where
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSortCustomAPI<T, D>,
{
    argsort_custom_f(tensor, args, f).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortCustomAPI<T, D>,
{
    /// Sort a tensor along an axis with a user comparator.
    ///
    /// See also [`sort_custom`].
    pub fn sort_custom_f<AArg, F>(&self, args: AArg, f: F) -> Result<Tensor<T, B, IxD>>
    where
        AArg: Into<SortArgs>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        sort_custom_f(self, args, f)
    }

    /// Sort a tensor along an axis with a user comparator.
    ///
    /// See also [`sort_custom`].
    pub fn sort_custom<AArg, F>(&self, args: AArg, f: F) -> Tensor<T, B, IxD>
    where
        AArg: Into<SortArgs>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        sort_custom_f(self, args, f).rstsr_unwrap()
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSortCustomAPI<T, D>,
{
    /// Returns the indices that sort a tensor along an axis with a user
    /// comparator.
    ///
    /// See also [`argsort_custom`].
    pub fn argsort_custom_f<AArg, F>(&self, args: AArg, f: F) -> Result<Tensor<usize, B, IxD>>
    where
        AArg: Into<SortArgs>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        argsort_custom_f(self, args, f)
    }

    /// Returns the indices that sort a tensor along an axis with a user
    /// comparator.
    ///
    /// See also [`argsort_custom`].
    pub fn argsort_custom<AArg, F>(&self, args: AArg, f: F) -> Tensor<usize, B, IxD>
    where
        AArg: Into<SortArgs>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        argsort_custom_f(self, args, f).rstsr_unwrap()
    }
}

/* #endregion */
