//! Set-function tensor API: the [`unique_*`](unique_values) family and
//! [`isin`], with named multi-output structs.

use crate::prelude_dev::*;

/* #region unique output structs */

/// Output of [`unique_counts`]: unique values and their multiplicities.
pub struct UniqueCounts<T, B>
where
    B: DeviceAPI<T> + DeviceRawAPI<usize>,
{
    /// Unique values (ascending for orderable dtypes, first-occurrence
    /// otherwise — the [`unique_values`] ordering contract).
    pub values: Tensor<T, B, IxD>,
    /// Multiplicity of each unique value, aligned with [`Self::values`].
    pub counts: Tensor<usize, B, IxD>,
}

impl<T, B> From<UniqueCounts<T, B>> for (Tensor<T, B, IxD>, Tensor<usize, B, IxD>)
where
    B: DeviceAPI<T> + DeviceRawAPI<usize>,
{
    fn from(value: UniqueCounts<T, B>) -> Self {
        (value.values, value.counts)
    }
}

/// Output of [`unique_inverse`]: unique values and the inverse mapping.
pub struct UniqueInverse<T, B>
where
    B: DeviceAPI<T> + DeviceRawAPI<usize>,
{
    /// Unique values (ascending for orderable dtypes, first-occurrence
    /// otherwise — the [`unique_values`] ordering contract).
    pub values: Tensor<T, B, IxD>,
    /// For every input element (row-major), the index of its unique entry;
    /// shape equals the input shape.
    pub inverse_indices: Tensor<usize, B, IxD>,
}

impl<T, B> From<UniqueInverse<T, B>> for (Tensor<T, B, IxD>, Tensor<usize, B, IxD>)
where
    B: DeviceAPI<T> + DeviceRawAPI<usize>,
{
    fn from(value: UniqueInverse<T, B>) -> Self {
        (value.values, value.inverse_indices)
    }
}

/// Output of [`unique_all`].
pub struct UniqueAll<T, B>
where
    B: DeviceAPI<T> + DeviceRawAPI<usize>,
{
    /// Unique values (ascending for orderable dtypes, first-occurrence
    /// otherwise — the [`unique_values`] ordering contract).
    pub values: Tensor<T, B, IxD>,
    /// First-occurrence flat C-order index of each unique value.
    pub indices: Tensor<usize, B, IxD>,
    /// Unique-entry index for every input element (input's shape).
    pub inverse_indices: Tensor<usize, B, IxD>,
    /// Multiplicity of each unique value, aligned with [`Self::values`].
    pub counts: Tensor<usize, B, IxD>,
}

/* #endregion */

/* #region shared implementation */

#[allow(clippy::type_complexity)]
fn unique_impl<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    with_all: bool,
) -> Result<(
    usize,
    Tensor<T, B, IxD>,
    Option<Tensor<usize, B, IxD>>,
    Option<Tensor<usize, B, IxD>>,
    Option<Tensor<usize, B, IxD>>,
)>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let n = tensor.size();
    // caller-allocated scratch of the input's count; the kernels return the
    // unique count and the tensor level truncates
    let values_layout = vec![n].new_contig(None, device.default_order());
    let (v_max,) = (values_layout.bounds_index()?.1,);
    let mut values = device.uninit_impl(v_max)?;
    if !with_all {
        let u = device.unique_values(tensor.raw(), tensor.layout(), values.raw_mut())?;
        // SAFETY: the device impl truncated the raw Vec to the `u` written
        // entries before returning (OpUniqueAPI contract).
        let storage = unsafe { <B as DeviceCreationAnyAPI<T>>::assume_init_impl(values)? };
        let t = Tensor::new_f(storage, vec![u].new_contig(None, device.default_order()))?;
        return Ok((u, t, None, None, None));
    }
    let mut indices = device.uninit_impl(v_max)?;
    let mut inverse = device.uninit_impl(v_max)?;
    let mut counts = device.uninit_impl(v_max)?;
    let u = device.unique_all(
        tensor.raw(),
        tensor.layout(),
        values.raw_mut(),
        indices.raw_mut(),
        inverse.raw_mut(),
        counts.raw_mut(),
    )?;
    // SAFETY: the device impl truncated values/indices/counts to the `u`
    // written entries and inverse to `n` (OpUniqueAPI contract).
    let values_t = Tensor::new_f(
        unsafe { <B as DeviceCreationAnyAPI<T>>::assume_init_impl(values)? },
        vec![u].new_contig(None, device.default_order()),
    )?;
    let indices_t = Tensor::new_f(
        unsafe { <B as DeviceCreationAnyAPI<usize>>::assume_init_impl(indices)? },
        vec![u].new_contig(None, device.default_order()),
    )?;
    let shape_in: Vec<usize> = tensor.shape().as_ref().to_vec();
    // inverse is written in row-major visit order → C-contig layout
    let inverse_t = Tensor::new_f(
        unsafe { <B as DeviceCreationAnyAPI<usize>>::assume_init_impl(inverse)? },
        shape_in.new_c_contig(None),
    )?;
    let counts_t = Tensor::new_f(
        unsafe { <B as DeviceCreationAnyAPI<usize>>::assume_init_impl(counts)? },
        vec![u].new_contig(None, device.default_order()),
    )?;
    Ok((u, values_t, Some(indices_t), Some(inverse_t), Some(counts_t)))
}

/* #endregion */

/* #region unique_values */

/// Returns the unique values of a tensor.
///
/// See also [`unique_values`].
pub fn unique_values_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    let (_, values, _, _, _) = unique_impl(tensor, false)?;
    Ok(values)
}

/// Returns the unique values of the flattened tensor.
///
/// The ordering depends on the dtype: orderable scalar dtypes (bool,
/// integers, real floats) return values in **ascending order** (NumPy
/// parity); other dtypes (e.g. complex) return values in **first-occurrence
/// order** over the strict row-major visit sequence. NaNs are distinct
/// entries (tail of the ascending order); signed zeros merge (the first-
/// seen encoding is kept, in both paths). Output shape is data-dependent
/// (1-D).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: unique values, 1-D.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);
/// println!("{}", rt::unique_values(&a));
/// // [ 1 2 3]
/// # assert_eq!(format!("{}", rt::unique_values(&a)), "[ 1 2 3]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `unique_values(x, /)` ([`unique_values`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_values.html))
/// - NumPy: `numpy.unique(x)` ([`numpy.unique`](https://numpy.org/doc/stable/reference/generated/numpy.unique.html))
/// - RSTSR: `rt::unique_values(tensor)`
///
/// Ordering note: rstsr returns ascending order for orderable scalar dtypes
/// (bool/integers/real floats) and first-occurrence order for others (e.g.
/// complex). This matches `numpy.unique` but not the array-api aliases
/// (`numpy.unique_values`/`unique_all`/...), which since NumPy 2.3 do not
/// guarantee an order. NaNs are distinct entries (parity with the array-api
/// aliases, which pass `equal_nan=False`; differs from `numpy.unique`, which
/// collapses them). See `tests/tracking/numpy_differences.md`.
///
/// # See also
///
/// ## Variants of this function
///
/// - [`unique_values_f`]: fallible version.
pub fn unique_values<Inp>(inp: Inp) -> Inp::Out
where
    Inp: UniqueValuesAPI,
{
    Inp::unique_values(inp)
}

/// API trait backing [`unique_values`].
pub trait UniqueValuesAPI {
    type Out;

    fn unique_values_f(self) -> Result<Self::Out>;
    fn unique_values(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::unique_values_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D> UniqueValuesAPI for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    type Out = Tensor<T, B, IxD>;

    fn unique_values_f(self) -> Result<Self::Out> {
        unique_values_f(self)
    }
}

/* #endregion */

/* #region unique_counts */

/// Returns the unique values of a tensor and their counts.
///
/// See also [`unique_counts`].
pub fn unique_counts_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<UniqueCounts<T, B>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    let (_, values, _, _, counts) = unique_impl(tensor, true)?;
    Ok(UniqueCounts { values, counts: counts.expect("unique_all returns counts") })
}

/// Returns the unique values of the flattened tensor and, for each, the
/// number of times it appears. Values follow the [`unique_values`]
/// ordering contract; `counts` is aligned with `values`.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
///
/// # Returns
///
/// - [`UniqueCounts<T, B>`][`UniqueCounts`]: `values` (1-D) and `counts` (1-D, same length);
///   converts `Into<(Tensor, Tensor)>`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);
/// let res = rt::unique_counts(&a);
/// println!("{}", res.values);
/// // [ 1 2 3]
/// println!("{}", res.counts);
/// // [ 2 1 2]
/// # assert_eq!(format!("{}", res.values), "[ 1 2 3]");
/// # assert_eq!(format!("{}", res.counts), "[ 2 1 2]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `unique_counts(x, /)` ([`unique_counts`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_counts.html))
/// - NumPy: `numpy.unique(x, return_counts=True)`
/// - RSTSR: `rt::unique_counts(tensor)`
///
/// # See also
///
/// ## Variants of this function
///
/// - [`unique_counts_f`]: fallible version.
pub fn unique_counts<Inp>(inp: Inp) -> Inp::Out
where
    Inp: UniqueCountsAPI,
{
    Inp::unique_counts(inp)
}

/// API trait backing [`unique_counts`].
pub trait UniqueCountsAPI {
    type Out;

    fn unique_counts_f(self) -> Result<Self::Out>;
    fn unique_counts(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::unique_counts_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D> UniqueCountsAPI for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    type Out = UniqueCounts<T, B>;

    fn unique_counts_f(self) -> Result<Self::Out> {
        unique_counts_f(self)
    }
}

/* #endregion */

/* #region unique_inverse */

/// Returns the unique values of a tensor and the inverse mapping.
///
/// See also [`unique_inverse`].
pub fn unique_inverse_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<UniqueInverse<T, B>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    let (_, values, _, inverse, _) = unique_impl(tensor, true)?;
    Ok(UniqueInverse { values, inverse_indices: inverse.expect("unique_all returns inverse") })
}

/// Returns the unique values of the flattened tensor and, for every input
/// element, the index of its unique entry. Values follow the [`unique_values`]
/// ordering contract; `inverse_indices` has the input's shape.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
///
/// # Returns
///
/// - [`UniqueInverse<T, B>`][`UniqueInverse`]: `values` (1-D) and `inverse_indices` (input's
///   shape); converts `Into<(Tensor, Tensor)>`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 3], &device);
/// let res = rt::unique_inverse(&a);
/// println!("{}", res.values);
/// // [ 1 3]
/// println!("{}", res.inverse_indices);
/// // [ 1 0 1]
/// # assert_eq!(format!("{}", res.values), "[ 1 3]");
/// # assert_eq!(format!("{}", res.inverse_indices), "[ 1 0 1]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `unique_inverse(x, /)` ([`unique_inverse`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_inverse.html))
/// - NumPy: `numpy.unique(x, return_inverse=True)` — NumPy flattens `inverse_indices`; the
///   array-api standard keeps the input shape (rstsr follows the standard).
/// - RSTSR: `rt::unique_inverse(tensor)`
///
/// # See also
///
/// ## Variants of this function
///
/// - [`unique_inverse_f`]: fallible version.
pub fn unique_inverse<Inp>(inp: Inp) -> Inp::Out
where
    Inp: UniqueInverseAPI,
{
    Inp::unique_inverse(inp)
}

/// API trait backing [`unique_inverse`].
pub trait UniqueInverseAPI {
    type Out;

    fn unique_inverse_f(self) -> Result<Self::Out>;
    fn unique_inverse(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::unique_inverse_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D> UniqueInverseAPI for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    type Out = UniqueInverse<T, B>;

    fn unique_inverse_f(self) -> Result<Self::Out> {
        unique_inverse_f(self)
    }
}

/* #endregion */

/* #region unique_all */

/// Returns the unique values, first-occurrence indices, inverse mapping,
/// and counts of a tensor.
///
/// See also [`unique_all`].
pub fn unique_all_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<UniqueAll<T, B>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    let (_, values, indices, inverse, counts) = unique_impl(tensor, true)?;
    Ok(UniqueAll {
        values,
        indices: indices.expect("unique_all returns indices"),
        inverse_indices: inverse.expect("unique_all returns inverse"),
        counts: counts.expect("unique_all returns counts"),
    })
}

/// Returns the unique values, first-occurrence flat C-order indices, the
/// inverse mapping, and multiplicities in one pass. Values follow the
/// [`unique_values`] ordering contract; `indices` is the first occurrence
/// (flattened row-major position) of each unique value;
/// `inverse_indices` has the input's shape.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
///
/// # Returns
///
/// - [`UniqueAll<T, B>`][`UniqueAll`]: fields `values`, `indices`, `inverse_indices`, `counts`.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([3, 1, 3], &device);
/// let res = rt::unique_all(&a);
/// println!("{}", res.values);
/// // [ 1 3]
/// println!("{}", res.indices);
/// // [ 1 0]
/// println!("{}", res.inverse_indices);
/// // [ 1 0 1]
/// println!("{}", res.counts);
/// // [ 1 2]
/// # assert_eq!(format!("{}", res.values), "[ 1 3]");
/// # assert_eq!(format!("{}", res.indices), "[ 1 0]");
/// # assert_eq!(format!("{}", res.inverse_indices), "[ 1 0 1]");
/// # assert_eq!(format!("{}", res.counts), "[ 1 2]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `unique_all(x, /)` returning a namedtuple with fields
///   `values`, `indices`, `inverse_indices`, `counts`
///   ([`unique_all`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_all.html))
/// - NumPy: `numpy.unique(x, return_index=True, return_inverse=True, return_counts=True)` — NumPy's
///   `indices` refer to the *sorted* unique sequence; rstsr refers to the first-occurrence
///   sequence.
/// - RSTSR: `rt::unique_all(tensor)`
///
/// # See also
///
/// ## Variants of this function
///
/// - [`unique_all_f`]: fallible version.
pub fn unique_all<Inp>(inp: Inp) -> Inp::Out
where
    Inp: UniqueAllAPI,
{
    Inp::unique_all(inp)
}

/// API trait backing [`unique_all`].
pub trait UniqueAllAPI {
    type Out;

    fn unique_all_f(self) -> Result<Self::Out>;
    fn unique_all(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::unique_all_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D> UniqueAllAPI for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + OpUniqueAPI<T, D>,
{
    type Out = UniqueAll<T, B>;

    fn unique_all_f(self) -> Result<Self::Out> {
        unique_all_f(self)
    }
}

/* #endregion */

/* #region isin */

/// Element membership of `x1` in `x2`.
///
/// See also [`isin`].
pub fn isin_f<R1, R2, T, B, D1, D2>(
    x1: &TensorAny<R1, T, B, D1>,
    x2: &TensorAny<R2, T, B, D2>,
    invert: bool,
) -> Result<Tensor<bool, B, IxD>>
where
    R1: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    R2: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D1: DimAPI,
    D2: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<bool>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceRawAPI<MaybeUninit<bool>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + DeviceCreationAnyAPI<bool>
        + OpIsinAPI<T, D1>,
{
    let device = x1.device().clone();
    rstsr_assert!(device.same_device(x2.device()), DeviceMismatch, "isin requires x1 and x2 on the same device.")?;
    let l2: Layout<IxD> = x2.layout().to_dim()?;
    let (storage, layout) = device.isin(x1.raw(), x1.layout(), x2.raw(), &l2, invert)?;
    Tensor::new_f(storage, layout)
}

/// Calculates the element-wise membership of `x1` in `x2`: the output is
/// `true` where an element of `x1` appears in `x2` (and inverted with
/// `invert = true`). Output shape equals `x1`'s shape. Membership is value
/// equality, so a NaN (or NaN-bearing complex) element is never a member,
/// even of a set containing NaN (NumPy parity).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders.
///
/// # Parameters
///
/// - `x1`: [`&TensorAny<R, T, B, D>`](TensorAny): the values to test.
/// - `x2`: [`&TensorAny<R, T, B, D>`](TensorAny): the membership set.
/// - `invert`: if `true`, the output is inverted.
///
/// # Returns
///
/// - [`Tensor<bool, B, IxD>`][`Tensor`]: with `x1`'s shape.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
/// let b = rt::tensor_from_nested!([2, 4], &device);
/// println!("{}", rt::isin((&a, &b, false)));
/// // [ false true false true]
/// # assert_eq!(format!("{}", rt::isin((&a, &b, false))), "[ false true false true]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `isin(x1, x2, /, *, invert=False)` (2025.12) ([`isin`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.isin.html))
/// - NumPy: `numpy.isin(element, test_elements, invert=False)`
/// - RSTSR: `rt::isin((x1, x2, invert))`
///
/// Deviation from NumPy: scalar `x1`/`x2` should be wrapped with
/// [`asarray`](asarray()); NumPy broadcasts `element` against
/// `test_elements`, the array-api standard does not (rstsr follows the
/// standard).
///
/// # See also
///
/// ## Variants of this function
///
/// - [`isin_f`]: fallible version.
pub fn isin<Args, Inp>(args: Args) -> Args::Out
where
    Args: IsinAPI<Inp>,
{
    Args::isin(args)
}

/// API trait backing [`isin`].
pub trait IsinAPI<Inp> {
    type Out;

    fn isin_f(self) -> Result<Self::Out>;
    fn isin(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::isin_f(self).rstsr_unwrap()
    }
}

impl<R1, R2, T, B, D1, D2> IsinAPI<()> for (&TensorAny<R1, T, B, D1>, &TensorAny<R2, T, B, D2>, bool)
where
    R1: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    R2: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D1: DimAPI,
    D2: DimAPI,
    T: Clone + PartialEq + ExtSortCmp + 'static,
    B: DeviceAPI<T>
        + DeviceAPI<bool>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceRawAPI<MaybeUninit<bool>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + DeviceCreationAnyAPI<bool>
        + OpIsinAPI<T, D1>,
{
    type Out = Tensor<bool, B, IxD>;

    fn isin_f(self) -> Result<Self::Out> {
        let (x1, x2, invert) = self;
        isin_f(x1, x2, invert)
    }
}

/* #endregion */
