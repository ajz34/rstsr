//! Advanced indexing related tensor manipulations.
//!
//! Currently, full support of advanced indexing is not available. However, it
//! is still possible to index one axis by list.

use crate::prelude_dev::*;

/* #region index_select */

/// Returns a new tensor, which indexes the input tensor along dimension `axis` using the entries in
/// `indices`.
///
/// See also [`index_select`].
pub fn index_select_f<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axis: isize, indices: I) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    // TODO: output layout control (TensorIterOrder::K or default layout)
    let device = tensor.device().clone();
    let tensor_layout = tensor.layout();
    let ndim = tensor_layout.ndim();
    // check axis and index
    let axis = rstsr_check_axis!(axis, ndim)?;
    let nshape: usize = tensor_layout.shape()[axis];
    let indices = indices.try_into().map_err(Into::into)?;
    let indices = indices
        .as_ref()
        .iter()
        .map(|&i| -> Result<usize> {
            let i = if i < 0 { nshape as isize + i } else { i };
            rstsr_pattern!(
                i,
                0..nshape as isize,
                IndexError,
                "Invalid index that exceeds shape length at axis {}.",
                axis
            )?;
            Ok(i as usize)
        })
        .collect::<Result<Vec<usize>>>()?;
    let mut out_shape = tensor_layout.shape().as_ref().to_vec();
    out_shape[axis] = indices.len();
    let out_layout = out_shape.new_contig(None, device.default_order()).into_dim()?;
    let mut out_storage = device.uninit_impl(out_layout.size())?;
    device.index_select(out_storage.raw_mut(), &out_layout, tensor.storage().raw(), tensor_layout, axis, &indices)?;
    // SAFETY: `device.index_select` above wrote all `out_layout.size()` elements
    // of the fresh storage.
    let out_storage = unsafe { B::assume_init_impl(out_storage)? };
    TensorBase::new_f(out_storage, out_layout)
}

/// Returns a new tensor, which indexes the input tensor along dimension `axis`
/// using the entries in `indices`.
///
/// The output has the same shape as the input except on `axis`, whose length
/// becomes `indices.len()`. Entries may repeat, and negative values count from
/// the back. The output is an owned tensor, contiguous in the device default
/// order.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `axis`: the axis to index along; negative values count from the back.
/// - `indices`: the indices to select, anything that converts into [`AxesIndex<isize>`][AxesIndex]
///   (array, vector, or slice of integers).
///
/// # Returns
///
/// - [`Tensor<T, B, D>`][`Tensor`]: the gathered tensor, owning its data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// println!("{}", rt::index_select(&a, 0, [0, 2]));
/// // [[ 0 1 2 3]
/// //  [ 8 9 10 11]]
/// println!("{}", rt::index_select(&a, 1, vec![3, 1]));
/// // [[ 3 1]
/// //  [ 7 5]
/// //  [ 11 9]]
/// # assert_eq!(format!("{}", rt::index_select(&a, 1, vec![3, 1])), "[[ 3 1]\n [ 7 5]\n [ 11 9]]");
/// ```
///
/// Negative indices count from the back:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// # let a = rt::arange((12, &device)).into_shape([3, 4]);
/// println!("{}", rt::index_select(&a, 0, [-1]));
/// // [[ 8 9 10 11]]
/// # assert_eq!(format!("{}", rt::index_select(&a, 0, [-1])), "[[ 8 9 10 11]]");
/// ```
///
/// # Notes of API accordance
///
/// - PyTorch: `torch.index_select(input, dim, index)` ([`torch.index_select`](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html))
/// - RSTSR: `rt::index_select(&tensor, axis, indices)`; indices are integers (negative allowed),
///   not a tensor.
///
/// # Panics
///
/// - Panics if `axis` is out of range, or if any index (after resolving negative values) is out of
///   bound on `axis`.
///
/// For a fallible version, use [`index_select_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - PyTorch: [`torch.index_select`](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html)
/// - NumPy: [`numpy.take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html) (see
///   also [`take`])
///
/// ## Related functions in RSTSR
///
/// - [`take`]: the same operation with NumPy's argument order.
/// - [`bool_select`]: select by boolean mask.
/// - [`slice`](slice()): basic indexing (views).
///
/// ## Variants of this function
///
/// - [`index_select_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::index_select`] /
///   [`TensorAny::index_select_f`].
pub fn index_select<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axis: isize, indices: I) -> Tensor<T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    index_select_f(tensor, axis, indices).rstsr_unwrap()
}

/// Take elements from a tensor along an axis.
///
/// See also [`take`].
pub fn take_f<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, indices: I, axis: isize) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    index_select_f(tensor, axis, indices)
}

/// Take elements from a tensor along an axis.
///
/// The same operation as [`index_select`] (indices may repeat, negative values
/// count from the back), with NumPy's argument order: `indices` before `axis`.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `indices`: the indices to take, anything that converts into [`AxesIndex<isize>`][AxesIndex]
///   (array, vector, or slice of integers).
/// - `axis`: the axis to take along; negative values count from the back.
///
/// # Returns
///
/// - [`Tensor<T, B, D>`][`Tensor`]: the gathered tensor, owning its data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// println!("{}", rt::take(&a, [0, 2], 0));
/// // [[ 0 1 2 3]
/// //  [ 8 9 10 11]]
/// # assert_eq!(format!("{}", rt::take(&a, [0, 2], 0)), "[[ 0 1 2 3]\n [ 8 9 10 11]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `take(x, indices, /, *, axis=None)` ([`take`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.take.html))
/// - NumPy: `numpy.take(a, indices, axis=None)` ([`numpy.take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html))
/// - RSTSR: `rt::take(&tensor, indices, axis)`; `axis` is mandatory (no flattened whole-tensor
///   form), and `mode`/`out` are not supported.
///
/// # Panics
///
/// - Panics if `axis` is out of range, or if any index (after resolving negative values) is out of
///   bound on `axis`.
///
/// For a fallible version, use [`take_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`take`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.take.html)
/// - NumPy: [`numpy.take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
///
/// ## Related functions in RSTSR
///
/// - [`index_select`]: the same operation with PyTorch's argument order.
/// - [`bool_select`]: select by boolean mask.
///
/// ## Variants of this function
///
/// - [`take_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::take`] / [`TensorAny::take_f`].
pub fn take<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, indices: I, axis: isize) -> Tensor<T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
{
    index_select(tensor, axis, indices)
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
{
    /// See also [`index_select`].
    pub fn index_select_f<I>(&self, axis: isize, indices: I) -> Result<Tensor<T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        index_select_f(self, axis, indices)
    }

    /// Returns a new tensor, which indexes the input tensor along dimension
    /// `axis` using the entries in `indices`.
    ///
    /// # See also
    ///
    /// This function should be similar to PyTorch's [`torch.index_select`](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html).
    pub fn index_select<I>(&self, axis: isize, indices: I) -> Tensor<T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        index_select(self, axis, indices)
    }

    pub fn take_f<I>(&self, indices: I, axis: isize) -> Result<Tensor<T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        take_f(self, indices, axis)
    }

    /// Take elements from an array along an axis.
    ///
    /// # See also
    ///
    /// [Python Array API standard: take](https://data-apis.org/array-api/latest/API_specification/generated/array_api.take.html#array_api.take)
    pub fn take<I>(&self, indices: I, axis: isize) -> Tensor<T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        take(self, indices, axis)
    }
}

/* #endregion */

/* #region bool_select */

/// Returns a new tensor, which indexes the input tensor along dimension `axis` using the boolean
/// entries in `mask`.
///
/// See also [`bool_select`].
pub fn bool_select_f<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axis: isize, mask: I) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<bool>, Error: Into<Error>>,
{
    // transform bool to index
    let indices = mask
        .try_into()
        .map_err(Into::into)?
        .as_ref()
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| m.then_some(i))
        .collect::<Vec<usize>>();
    index_select_f(tensor, axis, indices)
}

/// Returns a new tensor, which indexes the input tensor along dimension `axis`
/// using the boolean entries in `mask`.
///
/// Positions where `mask` is true are selected along `axis`; the output length
/// on `axis` equals the number of true entries. The output is an owned tensor,
/// contiguous in the device default order.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
/// - `axis`: the axis to select along; negative values count from the back.
/// - `mask`: boolean flags, anything that converts into [`AxesIndex<bool>`][AxesIndex] (array,
///   vector, or slice of `bool`), one entry per position on `axis`.
///
/// # Returns
///
/// - [`Tensor<T, B, D>`][`Tensor`]: the selected tensor, owning its data.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// println!("{}", rt::bool_select(&a, 1, [true, false, true, false]));
/// // [[ 0 2]
/// //  [ 4 6]
/// //  [ 8 10]]
/// # assert_eq!(format!("{}", rt::bool_select(&a, 1, [true, false, true, false])), "[[ 0 2]\n [ 4 6]\n [ 8 10]]");
/// ```
///
/// # Notes of API accordance
///
/// - PyTorch: `torch.index_select` applied on a boolean mask's positions ([`torch.masked_select`](https://docs.pytorch.org/docs/stable/generated/torch.masked_select.html)
///   flattens instead; rstsr selects along one axis)
/// - RSTSR: `rt::bool_select(&tensor, axis, mask)`; the mask must match the length of `axis`.
///
/// # Panics
///
/// - Panics if `axis` is out of range, or if any selected position exceeds the length of `axis`
///   (possible only when the mask is longer than the axis).
///
/// For a fallible version, use [`bool_select_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`index_select`] / [`take`]: select by integer indices.
///
/// ## Variants of this function
///
/// - [`bool_select_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::bool_select`] /
///   [`TensorAny::bool_select_f`].
pub fn bool_select<R, T, B, D, I>(tensor: &TensorAny<R, T, B, D>, axis: isize, mask: I) -> Tensor<T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
    I: TryInto<AxesIndex<bool>, Error: Into<Error>>,
{
    bool_select_f(tensor, axis, mask).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + DeviceIndexSelectAPI<T, D> + DeviceCreationAnyAPI<T>,
{
    pub fn bool_select_f<I>(&self, axis: isize, indices: I) -> Result<Tensor<T, B, D>>
    where
        I: TryInto<AxesIndex<bool>, Error: Into<Error>>,
    {
        bool_select_f(self, axis, indices)
    }

    /// Returns a new tensor, which indexes the input tensor along dimension
    /// `axis` using the boolean entries in `mask`.
    pub fn bool_select<I>(&self, axis: isize, indices: I) -> Tensor<T, B, D>
    where
        I: TryInto<AxesIndex<bool>, Error: Into<Error>>,
    {
        bool_select(self, axis, indices)
    }
}

/* #endregion */

/* #region take_along_axis */

/// Gather values along an axis using an index tensor.
///
/// See also [`take_along_axis`].
pub fn take_along_axis_f<T, B, DA, DI>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = DA>,
    indices: impl TensorViewAPI<Type = isize, Backend = B, Dim = DI>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<T, B, IxD>>
where
    DA: DimAPI,
    DI: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceAPI<isize, Raw = Vec<isize>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceTakeAlongAxisAPI<T, DA, DI>,
{
    let (tensor, indices) = (tensor.view(), indices.view());
    let axis: AxisIndex<isize> = axis.try_into().map_err(Into::into)?;
    let device = tensor.device().clone();
    rstsr_assert!(
        device.same_device(indices.device()),
        DeviceMismatch,
        "take_along_axis requires tensor and indices on the same device."
    )?;
    let axis = axis.into_normalized(tensor.ndim())?;
    // shape checks: same rank; non-axis dims must be broadcast-compatible
    // (array-api 2025.12: indices "must be compatible with x, except for the
    // axis specified by axis (see broadcasting)"; the output shape follows
    // that broadcasting)
    let la = tensor.layout();
    let lidx = indices.layout();
    rstsr_assert_eq!(
        lidx.ndim(),
        la.ndim(),
        InvalidLayout,
        "take_along_axis requires indices with the same ndim as the tensor."
    )?;
    let mut out_shape: Vec<usize> = la.shape().as_ref().to_vec();
    #[allow(clippy::needless_range_loop)] // reads and writes out_shape[i]
    for i in 0..la.ndim() {
        if i != axis {
            let d1 = out_shape[i];
            let d2 = lidx.shape()[i];
            let bcast = if d1 == d2 {
                d1
            } else if d1 == 1 {
                d2
            } else if d2 == 1 {
                d1
            } else {
                rstsr_assert!(
                    false,
                    InvalidLayout,
                    "take_along_axis requires broadcast-compatible shapes outside the indexed axis; got {} and {} along axis {}.",
                    d1,
                    d2,
                    i
                )?;
                unreachable!()
            };
            out_shape[i] = bcast;
        }
    }
    // validate + resolve index entries in logical row-major order (the index
    // tensor may be strided, offset, or broadcast — never read the raw
    // buffer): negatives count from the back (array-api/NumPy semantics; the
    // suite draws ~half negative indices)
    let axis_size = la.shape()[axis];
    let mut resolved: Vec<usize> = Vec::with_capacity(indices.size());
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&lidx.to_dim()?, RowMajor)?;
    for (_, off) in iter {
        let v = indices.raw()[off];
        let v = if v < 0 { v + axis_size as isize } else { v };
        rstsr_pattern!(
            v,
            0..axis_size as isize,
            IndexError,
            "take_along_axis index out of range along axis {}.",
            axis
        )?;
        resolved.push(v as usize);
    }
    // output shape: broadcast outside the axis, indices' length along it
    out_shape[axis] = lidx.shape()[axis];
    let layout_c = out_shape.new_contig(None, device.default_order());
    let (_, idx_max) = layout_c.bounds_index()?;
    let mut storage = device.uninit_impl(idx_max)?;
    // a fresh C-contig layout over the (unchanged) index shape addresses the
    // resolved index vector
    let lidx_resolved: Layout<DI> = lidx.shape().as_ref().to_vec().new_c_contig(None).to_dim()?;
    device.take_along_axis(storage.raw_mut(), &layout_c, tensor.raw(), la, &resolved, &lidx_resolved, axis)?;
    // SAFETY: `take_along_axis` above wrote every element of the fresh
    // storage exactly once (each (rest, j) position is filled from one
    // indexed source element).
    let storage = unsafe { <B as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
    Tensor::new_f(storage, layout_c)
}

/// Gather values along an axis using an index tensor: at every position
/// outside `axis`, `out[i_0, ..., j, ..., i_n] = x[i_0, ..., indices[i_0,
/// ..., j, ..., i_n], ..., i_n]`. This is the companion of [`argsort`]:
/// gathering `x` with `argsort(x, axis)` reproduces [`sort`].
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, DA>`](TensorAny): the source tensor.
/// - `indices`: [`&TensorAny<RI, isize, B, DI>`](TensorAny): integer indices along `axis`; same
///   rank, broadcast-compatible shapes outside `axis`. Negative entries count from the back of the
///   indexed axis.
/// - `axis`: TryInto [`AxisIndex<isize>`]: the axis to gather along (negative counts from the
///   back).
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: broadcast of the shapes outside `axis`, with `indices`'
///   length along it.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
/// let idx = rt::tensor_from_nested!([[2, 0], [1, 1]], &device);
/// println!("{}", rt::take_along_axis(&a, &idx, -1));
/// // [[ 30 10]
/// //  [ 50 50]]
/// # let out = rt::take_along_axis(&a, &idx, -1);
/// # assert_eq!(out.reshape([-1]).to_vec(), vec![30, 10, 50, 50]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `take_along_axis(x, indices, /, *, axis=-1)` ([`take_along_axis`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.take_along_axis.html))
/// - NumPy: `numpy.take_along_axis(arr, indices, axis)`
/// - RSTSR: `rt::take_along_axis(tensor, indices, axis)`
///
/// Shapes outside `axis` must be broadcast-compatible: a size-1 dimension on
/// either side broadcasts against the other (array-api 2025.12 and NumPy
/// parity — the output shape follows that broadcast).
///
/// # Panics
///
/// - Panics if `axis` is out of range, shapes mismatch outside `axis`, an index is out of range, or
///   the devices differ.
///
/// For a fallible version, use [`take_along_axis_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`argsort`]: produces valid index tensors.
/// - [`take`]: a single host-side index list along one axis.
///
/// ## Variants of this function
///
/// - [`take_along_axis_f`]: fallible version.
pub fn take_along_axis<T, B, DA, DI>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = DA>,
    indices: impl TensorViewAPI<Type = isize, Backend = B, Dim = DI>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
) -> Tensor<T, B, IxD>
where
    DA: DimAPI,
    DI: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceAPI<isize, Raw = Vec<isize>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceTakeAlongAxisAPI<T, DA, DI>,
{
    take_along_axis_f(tensor, indices, axis).rstsr_unwrap()
}

impl<R, T, B, DA> TensorAny<R, T, B, DA>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    DA: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceAPI<isize, Raw = Vec<isize>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>,
{
    /// Gather values along an axis using an index tensor.
    ///
    /// See also [`take_along_axis`].
    pub fn take_along_axis_f<DI, AArg>(
        &self,
        indices: impl TensorViewAPI<Type = isize, Backend = B, Dim = DI>,
        axis: AArg,
    ) -> Result<Tensor<T, B, IxD>>
    where
        DI: DimAPI,
        AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
        B: DeviceAPI<T>
            + DeviceAPI<usize, Raw = Vec<usize>>
            + DeviceAPI<isize, Raw = Vec<isize>>
            + DeviceRawAPI<MaybeUninit<T>>
            + DeviceCreationAnyAPI<T>
            + DeviceTakeAlongAxisAPI<T, DA, DI>,
    {
        take_along_axis_f(self, indices, axis)
    }

    /// Gather values along an axis using an index tensor.
    ///
    /// See also [`take_along_axis`].
    pub fn take_along_axis<DI, AArg>(
        &self,
        indices: impl TensorViewAPI<Type = isize, Backend = B, Dim = DI>,
        axis: AArg,
    ) -> Tensor<T, B, IxD>
    where
        DI: DimAPI,
        AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
        B: DeviceAPI<T>
            + DeviceAPI<usize, Raw = Vec<usize>>
            + DeviceAPI<isize, Raw = Vec<isize>>
            + DeviceRawAPI<MaybeUninit<T>>
            + DeviceCreationAnyAPI<T>
            + DeviceTakeAlongAxisAPI<T, DA, DI>,
    {
        take_along_axis_f(self, indices, axis).rstsr_unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_index_select() {
        #[cfg(not(feature = "col_major"))]
        {
            let device = DeviceCpuSerial::default();
            let a = linspace((1.0, 24.0, 24, &device)).into_shape((2, 3, 4));
            let b = a.index_select(0, [0, 0, 1, -1]);
            assert!(fingerprint(&b) - -31.94175930917264 < 1e-8);
            let b = a.index_select(1, [0, 0, 1, -1]);
            assert!(fingerprint(&b) - 3.5719025258942088 < 1e-8);
            let b = a.index_select(2, [0, 0, 1, -1]);
            assert!(fingerprint(&b) - -25.648600916145096 < 1e-8);
        }
        #[cfg(feature = "col_major")]
        {
            let device = DeviceCpuSerial::default();
            let a = linspace((1.0, 24.0, 24, &device)).into_shape((4, 3, 2));
            let b = a.index_select(2, [0, 0, 1, -1]);
            assert!(fingerprint(&b) - -31.94175930917264 < 1e-8);
            let b = a.index_select(1, [0, 0, 1, -1]);
            assert!(fingerprint(&b) - 3.5719025258942088 < 1e-8);
            let b = a.index_select(0, [0, 0, 1, -1]);
            assert!(fingerprint(&b) - -25.648600916145096 < 1e-8);
        }

        // 1-dim select with empty index
        let device = DeviceCpuSerial::default();
        let a = linspace((1.0, 4.0, 4, &device));
        let mask: Vec<usize> = vec![];
        let b = a.index_select(0, &mask);
        assert_eq!(b.raw(), &[]);
    }

    #[test]
    fn test_index_select_default_device() {
        #[cfg(not(feature = "col_major"))]
        {
            let device = DeviceCpu::default();
            let a = linspace((1.0, 2.0, 256 * 256 * 256, &device)).into_shape((256, 256, 256));
            let sel = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];
            let b = a.index_select(0, sel);
            assert!(fingerprint(&b) - 0.9357016252766746 < 1e-10);
            let b = a.index_select(1, sel);
            assert!(fingerprint(&b) - 1.012193909979973 < 1e-10);
            let b = a.index_select(2, sel);
            assert!(fingerprint(&b) - 1.010735112247236 < 1e-10);
        }
        #[cfg(feature = "col_major")]
        {
            let device = DeviceCpu::default();
            let a = linspace((1.0, 2.0, 256 * 256 * 256, &device)).into_shape((256, 256, 256));
            let sel = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];
            let b = a.index_select(2, sel);
            assert!(fingerprint(&b) - 0.9357016252766746 < 1e-10);
            let b = a.index_select(1, sel);
            assert!(fingerprint(&b) - 1.012193909979973 < 1e-10);
            let b = a.index_select(0, sel);
            assert!(fingerprint(&b) - 1.010735112247236 < 1e-10);
        }
    }

    #[test]
    fn test_bool_select_workable() {
        let a = arange(24).into_shape((2, 3, 4));
        let b = a.bool_select(-2, [true, false, true]);
        println!("{b:?}");
    }
}
