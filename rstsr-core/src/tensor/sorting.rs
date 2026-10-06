//! Sorting tensor API: [`sort`], [`argsort`], and the `_custom` comparator
//! variants, with argument type [`SortArgs`].

use core::cmp::Ordering;

use crate::prelude_dev::*;

/* #region argsort */

/// Returns the indices that sort a tensor along an axis.
///
/// See also [`argsort`].
pub fn argsort_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, args: impl Into<SortArgs>) -> Result<Tensor<usize, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
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
    let (storage, layout) = device.argsort_axes(tensor.raw(), tensor.layout(), axis, args.descending, args.stable)?;
    Tensor::new_f(storage, layout)
}

/// Returns the indices that sort a tensor along an axis.
///
/// The returned indices are of dtype [`usize`] and have the same shape as the
/// input; gathering with them along `axis` (`take`) reproduces [`sort`]. Ties
/// keep the input order (stable), matching NumPy's default `quicksort`
/// guarantee for `argsort` only in the stable sense — rstsr always provides
/// the stable behavior.
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
/// - `args`: TryInto [`SortArgs`]
///
///   - `()`: defaults (`axis = -1`, ascending, stable).
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
/// println!("{}", rt::argsort((&a, ())));
/// // [ 1 2 0]
/// # assert_eq!(format!("{}", rt::argsort((&a, ()))), "[ 1 2 0]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `argsort(x, /, *, axis=-1, descending=False, stable=True)` ([`argsort`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.argsort.html))
/// - NumPy: `numpy.argsort(a, axis=-1)` ([`numpy.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html))
/// - RSTSR: `rt::argsort((tensor, args))`
///
/// Deviation from NumPy: NumPy's `kind` parameter is not supported (the sort
/// is always stable). Complex dtypes sort lexicographically (real part first),
/// whereas NumPy raises for complex sort — rstsr declines complex here to
/// follow the array-api standard (complex is accepted by
/// [`argsort_custom`] with a user comparator).
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
pub fn argsort<Args, Inp>(args: Args) -> Args::Out
where
    Args: ArgSortAPI<Inp>,
{
    Args::argsort(args)
}

/// API trait backing [`argsort`].
pub trait ArgSortAPI<Inp> {
    type Out;

    fn argsort_f(self) -> Result<Self::Out>;
    fn argsort(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::argsort_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D, AArg> ArgSortAPI<()> for (&TensorAny<R, T, B, D>, AArg)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    AArg: Into<SortArgs>,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpArgSortAPI<T, D>,
{
    type Out = Tensor<usize, B, IxD>;

    fn argsort_f(self) -> Result<Self::Out> {
        let (tensor, args) = self;
        argsort_f(tensor, args)
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
pub fn sort_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, args: impl Into<SortArgs>) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortAPI<T, D>,
{
    let args = args.into();
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(args.axis, tensor.ndim())?;
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
/// - `args`: TryInto [`SortArgs`] — see [`argsort`] for the accepted forms.
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
/// println!("{}", rt::sort((&a, ())));
/// // [ 1 2 3]
/// # assert_eq!(format!("{}", rt::sort((&a, ()))), "[ 1 2 3]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `sort(x, /, *, axis=-1, descending=False, stable=True)` ([`sort`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.sort.html))
/// - NumPy: `numpy.sort(a, axis=-1)` ([`numpy.sort`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html))
/// - RSTSR: `rt::sort((tensor, args))`
///
/// Deviation from NumPy: NumPy's `kind` parameter is not supported (the sort
/// is always stable); NaN orders last in descending sorts as well (NumPy
/// 2.x behavior).
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
pub fn sort<Args, Inp>(args: Args) -> Args::Out
where
    Args: SortAPI<Inp>,
{
    Args::sort(args)
}

/// API trait backing [`sort`].
pub trait SortAPI<Inp> {
    type Out;

    fn sort_f(self) -> Result<Self::Out>;
    fn sort(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::sort_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D, AArg> SortAPI<()> for (&TensorAny<R, T, B, D>, AArg)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    AArg: Into<SortArgs>,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortAPI<T, D>,
{
    type Out = Tensor<T, B, IxD>;

    fn sort_f(self) -> Result<Self::Out> {
        let (tensor, args) = self;
        sort_f(tensor, args)
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
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
pub fn sort_custom_f<R, T, B, D, F>(
    tensor: &TensorAny<R, T, B, D>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    f: F,
) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortCustomAPI<T, D>,
{
    let axis = axis.try_into().map_err(Into::into)?.into_inner();
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(axis, tensor.ndim())?;
    let (storage, layout) = device.sort_axes_custom(tensor.raw(), tensor.layout(), axis, f)?;
    Tensor::new_f(storage, layout)
}

/// Sort a tensor along an axis with a user-supplied comparator
/// `f: Fn(&T, &T) -> Ordering`.
///
/// The comparator fully decides the order (stability: ties under `f` keep
/// input order). This is the escape hatch for orderings rstsr does not ship —
/// for example, complex values compared by squared magnitude:
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
/// println!("{}", rt::sort_custom((&a, -1, by_norm)));
/// // sorted by |z|^2 ascending: (1+1i), (2i), (3)
/// ```
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `axis`: TryInto [`AxisIndex<isize>`]: the axis to sort along.
/// - `f`: the comparator; must be a consistent total order for correct results.
///
/// # Panics
///
/// - Panics if the axis is out of range.
///
/// For a fallible version, use [`sort_custom_f`].
///
/// # See also
///
/// ## Variants of this function
///
/// - [`sort_custom_f`]: fallible version.
/// - [`argsort_custom`]: the permutation instead of the values.
pub fn sort_custom<Args, Inp>(args: Args) -> Args::Out
where
    Args: SortCustomAPI<Inp>,
{
    Args::sort_custom(args)
}

/// API trait backing [`sort_custom`].
pub trait SortCustomAPI<Inp> {
    type Out;

    fn sort_custom_f(self) -> Result<Self::Out>;
    fn sort_custom(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::sort_custom_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D, AArg, F> SortCustomAPI<()> for (&TensorAny<R, T, B, D>, AArg, F)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpSortCustomAPI<T, D>,
{
    type Out = Tensor<T, B, IxD>;

    fn sort_custom_f(self) -> Result<Self::Out> {
        let (tensor, axis, f) = self;
        sort_custom_f(tensor, axis, f)
    }
}

/// Returns the indices that sort a tensor along an axis with a user
/// comparator.
///
/// See also [`argsort_custom`].
pub fn argsort_custom_f<R, T, B, D, F>(
    tensor: &TensorAny<R, T, B, D>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    f: F,
) -> Result<Tensor<usize, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSortCustomAPI<T, D>,
{
    let axis = axis.try_into().map_err(Into::into)?.into_inner();
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let axis = rstsr_check_axis!(axis, tensor.ndim())?;
    let (storage, layout) = device.argsort_axes_custom(tensor.raw(), tensor.layout(), axis, f)?;
    Tensor::new_f(storage, layout)
}

/// Returns the indices that sort a tensor along an axis with a user-supplied
/// comparator `f: Fn(&T, &T) -> Ordering` (ties keep input order).
///
/// # Panics
///
/// - Panics if the axis is out of range.
///
/// For a fallible version, use [`argsort_custom_f`].
///
/// # See also
///
/// ## Variants of this function
///
/// - [`argsort_custom_f`]: fallible version.
/// - [`sort_custom`]: the values instead of the permutation.
pub fn argsort_custom<Args, Inp>(args: Args) -> Args::Out
where
    Args: ArgSortCustomAPI<Inp>,
{
    Args::argsort_custom(args)
}

/// API trait backing [`argsort_custom`].
pub trait ArgSortCustomAPI<Inp> {
    type Out;

    fn argsort_custom_f(self) -> Result<Self::Out>;
    fn argsort_custom(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::argsort_custom_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D, AArg, F> ArgSortCustomAPI<()> for (&TensorAny<R, T, B, D>, AArg, F)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSortCustomAPI<T, D>,
{
    type Out = Tensor<usize, B, IxD>;

    fn argsort_custom_f(self) -> Result<Self::Out> {
        let (tensor, axis, f) = self;
        argsort_custom_f(tensor, axis, f)
    }
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
    pub fn sort_custom_f<AArg, F>(&self, axis: AArg, f: F) -> Result<Tensor<T, B, IxD>>
    where
        AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        sort_custom_f(self, axis, f)
    }

    /// Sort a tensor along an axis with a user comparator.
    ///
    /// See also [`sort_custom`].
    pub fn sort_custom<AArg, F>(&self, axis: AArg, f: F) -> Tensor<T, B, IxD>
    where
        AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        sort_custom_f(self, axis, f).rstsr_unwrap()
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
    pub fn argsort_custom_f<AArg, F>(&self, axis: AArg, f: F) -> Result<Tensor<usize, B, IxD>>
    where
        AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        argsort_custom_f(self, axis, f)
    }

    /// Returns the indices that sort a tensor along an axis with a user
    /// comparator.
    ///
    /// See also [`argsort_custom`].
    pub fn argsort_custom<AArg, F>(&self, axis: AArg, f: F) -> Tensor<usize, B, IxD>
    where
        AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
        F: Fn(&T, &T) -> Ordering + Send + Sync,
    {
        argsort_custom_f(self, axis, f).rstsr_unwrap()
    }
}

/* #endregion */
