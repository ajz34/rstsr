//! Searchsorted tensor API: [`searchsorted`] with [`SearchSortedArgs`].

use crate::prelude_dev::*;

/* #region searchsorted */

/// Find the positions where values of `x2` would insert into sorted `x1`.
///
/// See also [`searchsorted`].
pub fn searchsorted_f<T, B, D1, D2, AArg>(
    x1: impl TensorViewAPI<Type = T, Backend = B, Dim = D1>,
    x2: impl TensorViewAPI<Type = T, Backend = B, Dim = D2>,
    args: AArg,
) -> Result<Tensor<usize, B, IxD>>
where
    D1: DimAPI,
    D2: DimAPI,
    AArg: TryInto<SearchSortedArgs, Error: Into<Error>>,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSearchSortedAPI<T, T, D2>,
{
    let (x1, x2) = (x1.view(), x2.view());
    let args = args.try_into().map_err(Into::into)?;
    let device = x1.device().clone();
    rstsr_assert!(
        device.same_device(x2.device()),
        DeviceMismatch,
        "searchsorted requires x1 and x2 on the same device."
    )?;
    if let Some(sorter) = &args.sorter {
        rstsr_assert_eq!(
            sorter.len(),
            x1.size(),
            InvalidValue,
            "searchsorted sorter must have the same length as x1."
        )?;
        for &idx in sorter.iter() {
            rstsr_pattern!(idx, 0..x1.size(), InvalidValue, "searchsorted sorter entries must be within x1 bounds.")?;
        }
    }
    let l1: Layout<IxD> = x1.layout().to_dim()?;
    rstsr_assert_eq!(l1.ndim(), 1, InvalidLayout, "searchsorted requires a one-dimensional x1.")?;
    let (storage, layout) =
        device.searchsorted(x1.raw(), &l1, x2.raw(), x2.layout(), args.side.into(), args.sorter.as_deref())?;
    Tensor::new_f(storage, layout)
}

/// Finds the indices into a sorted one-dimensional array `x1` such that, if
/// the corresponding elements in `x2` were inserted before the indices, the
/// order of `x1` would be preserved.
///
/// Output dtype is [`usize`] and the output shape equals `x2`'s shape.
/// Elements of `x2` are searched as-is (they need not be sorted). Real NaN
/// values land after all finite elements of `x1`, consistent with the sort
/// order of [`ExtSortCmp`]. Complex keys containing a NaN component are an
/// edge: the binary search hoists any NaN-bearing key after all finite
/// entries, which deviates from the part-wise lexicographic order of
/// [`ExtSortCmp`] (and NumPy); complex `searchsorted` remains a registered
/// follow-up.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `x1`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The one-dimensional array to search into. Must be sorted ascending; otherwise pass `sorter`
///     (a permutation sorting `x1` ascending).
///   - Complex and other dtypes are admitted; the comparison is the total order of [`ExtSortCmp`].
///
/// - `x2`: [`&TensorAny<R, T, B, D>`](TensorAny): the values to insert (any shape).
///
/// - `args`: TryInto [`SearchSortedArgs`]
///
///   - `()`: defaults (`side = "left"`, no sorter).
///   - `"left"` / `"right"` (or [`SearchSide`]): insertion side — `'left'` gives `x1[i-1] < v <=
///     x1[i]`, `'right'` gives `x1[i-1] <= v < x1[i]`.
///   - A `Vec<usize>`: the `sorter` permutation.
///   - `("left" | SearchSide, Vec<usize>)`: both.
///
/// # Returns
///
/// - [`Tensor<usize, B, IxD>`][`Tensor`]
///
///   - Insertion positions, same shape as `x2`; with `sorter`, positions index the permuted
///     sequence (NumPy-compatible).
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x1 = rt::tensor_from_nested!([11, 12, 14, 15, 16], &device);
/// let x2 = rt::tensor_from_nested!([10, 13, 17], &device);
/// println!("{}", rt::searchsorted(&x1, &x2, ()));
/// // [ 0 2 5]
/// # assert_eq!(format!("{}", rt::searchsorted(&x1, &x2, ())), "[ 0 2 5]");
/// ```
///
/// Insertion side (`side = "right"`):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x1 = rt::tensor_from_nested!([10, 20, 30], &device);
/// let v = rt::tensor_from_nested!([20], &device);
/// println!("{}", rt::searchsorted(&x1, &v, "left"));
/// // [ 1]
/// println!("{}", rt::searchsorted(&x1, &v, "right"));
/// // [ 2]
/// # assert_eq!(format!("{}", rt::searchsorted(&x1, &v, "left")), "[ 1]");
/// # assert_eq!(format!("{}", rt::searchsorted(&x1, &v, "right")), "[ 2]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `searchsorted(x1, x2, /, *, side='left', sorter=None)` ([`searchsorted`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.searchsorted.html))
/// - NumPy: `numpy.searchsorted(a, v, side='left', sorter=None)` ([`numpy.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html))
/// - RSTSR: `rt::searchsorted(x1, x2, args)`
///
/// Deviation from NumPy: scalar `x2` should be wrapped with
/// [`asarray`](asarray()) (rstsr functions take tensors). The returned index
/// dtype is [`usize`].
///
/// # Panics
///
/// - Panics if `x1` is not one-dimensional, or the devices differ.
///
/// For a fallible version, use [`searchsorted_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`sort`]: produces the sorted order `searchsorted` assumes.
/// - [`argsort`]: produces a valid `sorter` for unsorted `x1`.
///
/// ## Variants of this function
///
/// - [`searchsorted_f`]: fallible version.
/// - [`TensorAny::searchsorted`]: associated method.
/// - [`TensorAny::searchsorted_f`]: associated fallible method.
pub fn searchsorted<T, B, D1, D2, AArg>(
    x1: impl TensorViewAPI<Type = T, Backend = B, Dim = D1>,
    x2: impl TensorViewAPI<Type = T, Backend = B, Dim = D2>,
    args: AArg,
) -> Tensor<usize, B, IxD>
where
    D1: DimAPI,
    D2: DimAPI,
    AArg: TryInto<SearchSortedArgs, Error: Into<Error>>,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpSearchSortedAPI<T, T, D2>,
{
    searchsorted_f(x1, x2, args).rstsr_unwrap()
}

impl<R1, T, B, D1> TensorAny<R1, T, B, D1>
where
    R1: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D1: DimAPI,
    B: DeviceAPI<T> + DeviceAPI<usize> + DeviceRawAPI<MaybeUninit<usize>> + DeviceCreationAnyAPI<usize>,
{
    /// Finds the positions where values of `x2` would insert into sorted `x1`.
    ///
    /// See also [`searchsorted`].
    pub fn searchsorted_f<D2, AArg>(
        &self,
        x2: impl TensorViewAPI<Type = T, Backend = B, Dim = D2>,
        args: AArg,
    ) -> Result<Tensor<usize, B, IxD>>
    where
        D2: DimAPI,
        AArg: TryInto<SearchSortedArgs, Error: Into<Error>>,
        B: DeviceAPI<T>
            + DeviceAPI<usize>
            + DeviceRawAPI<MaybeUninit<usize>>
            + DeviceCreationAnyAPI<usize>
            + OpSearchSortedAPI<T, T, D2>,
    {
        searchsorted_f(self, x2, args)
    }

    /// Finds the positions where values of `x2` would insert into sorted `x1`.
    ///
    /// See also [`searchsorted`].
    pub fn searchsorted<D2, AArg>(
        &self,
        x2: impl TensorViewAPI<Type = T, Backend = B, Dim = D2>,
        args: AArg,
    ) -> Tensor<usize, B, IxD>
    where
        D2: DimAPI,
        AArg: TryInto<SearchSortedArgs, Error: Into<Error>>,
        B: DeviceAPI<T>
            + DeviceAPI<usize>
            + DeviceRawAPI<MaybeUninit<usize>>
            + DeviceCreationAnyAPI<usize>
            + OpSearchSortedAPI<T, T, D2>,
    {
        searchsorted_f(self, x2, args).rstsr_unwrap()
    }
}

/* #endregion */
