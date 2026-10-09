//! Advanced indexing related tensor manipulations: one-axis gathers (by an
//! integer list or an index tensor) and whole-tensor boolean-mask selection.
//!
//! Array indexing (*fancy indexing*, integer index arrays possibly mixed with
//! basic indexers) lives in [`array_index`].

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

/* #region mask_select / mask_fill */

/// A boolean-mask index is valid when it has no more axes than the tensor and
/// each of its axes matches the tensor's leading axis, or is zero (NumPy
/// allows a zero-size mask axis; it selects nothing).
fn check_mask_axes<DA, DM>(la: &Layout<DA>, lm: &Layout<DM>) -> Result<()>
where
    DA: DimAPI,
    DM: DimAPI,
{
    rstsr_assert!(
        lm.ndim() <= la.ndim(),
        IndexError,
        "boolean-mask index has {} axes, but the indexed tensor has only {}",
        lm.ndim(),
        la.ndim()
    )?;
    #[allow(clippy::needless_range_loop)] // reads both shapes
    for d in 0..lm.ndim() {
        let (m, x) = (lm.shape()[d], la.shape()[d]);
        rstsr_assert!(
            m == x || m == 0,
            IndexError,
            "boolean-mask axis {} has size {}, but the indexed tensor has size {} (must match, or be 0)",
            d,
            m,
            x
        )?;
    }
    Ok(())
}

/// Select the elements of `tensor` where `mask` is true (Array API `x[mask]`).
///
/// See also [`mask_select`].
pub fn mask_select_f<T, B, DA, DM>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = DA>,
    mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>,
) -> Result<Tensor<T, B, IxD>>
where
    DA: DimAPI,
    DM: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<bool>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + OpNonzeroAPI<bool, DM>
        + DeviceMaskIndexAPI<T, DA, DM>,
{
    let (tensor, mask) = (tensor.view(), mask.view());
    let device = tensor.device().clone();
    rstsr_assert!(device.same_device(mask.device()), DeviceMismatch)?;
    let (la, lm) = (tensor.layout(), mask.layout());
    check_mask_axes(la, lm)?;
    // the leading output length is data-dependent: one element per true entry
    let count = device.nonzero_count(mask.raw(), lm)?;
    let mut out_shape: Vec<usize> = Vec::with_capacity(la.ndim() - lm.ndim() + 1);
    out_shape.push(count);
    out_shape.extend_from_slice(&la.shape().as_ref()[lm.ndim()..]);
    let layout_c: Layout<IxD> = out_shape.new_contig(None, device.default_order());
    let (_, size) = layout_c.bounds_index()?;
    let mut storage = device.uninit_impl(size)?;
    device.mask_select(storage.raw_mut(), tensor.raw(), la, mask.raw(), lm)?;
    // SAFETY: `mask_select` wrote exactly `count * prod(shape[lm.ndim()..])`
    // elements = `size`.
    let storage = unsafe { <B as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
    Tensor::new_f(storage, layout_c)
}

/// Select the elements of `tensor` where `mask` is true (Array API `x[mask]`).
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// The mask may have at most `x.ndim()` axes; each of its axes must match the
/// corresponding leading axis of `x` (a zero-size mask axis selects nothing).
/// The result is `(count,) + x.shape[mask.ndim() ..]`, where `count` is the
/// number of true entries.
///
/// The mask entries are visited in the device default order, and the selected
/// elements (for a prefix mask, each trailing block) are emitted in that same
/// order — row-major under [`RowMajor`], column-major under [`ColMajor`]. The
/// result's memory arrangement likewise follows the device default order. This
/// is intentionally *not* the Python array API rule, which prescribes
/// row-major (C-style) iteration of a boolean index array regardless of
/// device; RSTSR keeps its device-order convention for flattened visit
/// orders, as in [`crate::tensor::nonzero::nonzero`].
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// let mask = rt::tensor_from_nested!([[true, false, true, false], [false, false, false, true], [false, false, true, false]], &device);
/// println!("{}", rt::mask_select(&a, &mask));
/// // [ 0 2 7 10]
/// # assert_eq!(rt::mask_select(&a, &mask).to_vec(), vec![0, 2, 7, 10]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `x[mask]` ([`indexing`](https://data-apis.org/array-api/2024.12/API_specification/indexing.html))
/// - NumPy: `x[mask]` (`numpy.ndarray.__getitem__`)
/// - RSTSR: `rt::mask_select(x, mask)`
///
/// # Panics
///
/// - Panics if the mask has more axes than `x`, or an axis whose size is neither `x`'s nor zero.
///
/// For a fallible version, use [`mask_select_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`bool_select`]: select along one axis by a per-axis mask.
/// - [`crate::tensor::nonzero::nonzero`]: the coordinates of the true entries.
///
/// ## Variants of this function
///
/// - [`mask_select_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::mask_select`] /
///   [`TensorAny::mask_select_f`].
pub fn mask_select<T, B, DA, DM>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = DA>,
    mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>,
) -> Tensor<T, B, IxD>
where
    DA: DimAPI,
    DM: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<bool>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + OpNonzeroAPI<bool, DM>
        + DeviceMaskIndexAPI<T, DA, DM>,
{
    mask_select_f(tensor, mask).rstsr_unwrap()
}

/// Write `value` into every element of `tensor` where `mask` is true (Array API
/// `x[mask] = value`).
///
/// See also [`mask_fill`].
pub fn mask_fill_f<RA, T, B, DA, DM>(
    tensor: &mut TensorAny<RA, T, B, DA>,
    mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>,
    value: T,
) -> Result<()>
where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    DA: DimAPI,
    DM: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceAPI<bool> + DeviceRawAPI<MaybeUninit<T>> + DeviceMaskIndexAPI<T, DA, DM>,
{
    let mask = mask.view();
    let device = tensor.device().clone();
    rstsr_assert!(device.same_device(mask.device()), DeviceMismatch)?;
    check_mask_axes(tensor.layout(), mask.layout())?;
    let la = tensor.layout().clone();
    device.mask_fill(tensor.raw_mut(), &la, mask.raw(), mask.layout(), value)
}

/// Write `value` into every element of `tensor` where `mask` is true (Array API
/// `x[mask] = value`).
///
/// See [`mask_select`] for the mask-shape contract. Only a scalar `value` is
/// supported.
///
/// # Notes of API accordance
///
/// - Array-API: `x[mask] = value` ([`indexing`](https://data-apis.org/array-api/2024.12/API_specification/indexing.html))
/// - NumPy: `x[mask] = value`
/// - RSTSR: `rt::mask_fill(&mut x, mask, value)`
///
/// # Panics
///
/// - Panics if the mask has more axes than `x`, or an axis whose size is neither `x`'s nor zero.
///
/// For a fallible version, use [`mask_fill_f`].
///
/// # See also
///
/// ## Variants of this function
///
/// - [`mask_fill_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::mask_fill`] / [`TensorAny::mask_fill_f`].
pub fn mask_fill<RA, T, B, DA, DM>(
    tensor: &mut TensorAny<RA, T, B, DA>,
    mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>,
    value: T,
) where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    DA: DimAPI,
    DM: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceAPI<bool> + DeviceRawAPI<MaybeUninit<T>> + DeviceMaskIndexAPI<T, DA, DM>,
{
    mask_fill_f(tensor, mask, value).rstsr_unwrap()
}

impl<R, T, B, DA> TensorAny<R, T, B, DA>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    DA: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<bool>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>,
{
    pub fn mask_select_f<DM>(
        &self,
        mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>,
    ) -> Result<Tensor<T, B, IxD>>
    where
        DM: DimAPI,
        B: OpNonzeroAPI<bool, DM> + DeviceMaskIndexAPI<T, DA, DM>,
    {
        mask_select_f(self, mask)
    }

    pub fn mask_select<DM>(&self, mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>) -> Tensor<T, B, IxD>
    where
        DM: DimAPI,
        B: OpNonzeroAPI<bool, DM> + DeviceMaskIndexAPI<T, DA, DM>,
    {
        mask_select(self, mask)
    }
}

impl<RA, T, B, DA> TensorAny<RA, T, B, DA>
where
    RA: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    DA: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceAPI<bool> + DeviceRawAPI<MaybeUninit<T>>,
{
    pub fn mask_fill_f<DM>(
        &mut self,
        mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>,
        value: T,
    ) -> Result<()>
    where
        DM: DimAPI,
        B: DeviceMaskIndexAPI<T, DA, DM>,
    {
        mask_fill_f(self, mask, value)
    }

    pub fn mask_fill<DM>(&mut self, mask: impl TensorViewAPI<Type = bool, Backend = B, Dim = DM>, value: T)
    where
        DM: DimAPI,
        B: DeviceMaskIndexAPI<T, DA, DM>,
    {
        mask_fill(self, mask, value)
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

    #[test]
    fn test_mask_indexing_workable() {
        #[cfg(not(feature = "col_major"))]
        {
            let device = DeviceCpu::default();
            let a = arange((12, &device)).into_shape([3, 4]);

            // full mask: 1-D result in row-major visit order
            let m = asarray((
                vec![true, false, true, false, false, false, false, true, false, false, true, false],
                &device,
            ))
            .into_shape([3, 4]);
            let out = a.mask_select(&m);
            assert_eq!(out.shape(), &vec![4]);
            assert_eq!(out.to_vec(), vec![0, 2, 7, 10]);

            // prefix mask: (count,) + the trailing shape
            let pm = asarray((vec![true, false, true], &device));
            let out = a.mask_select(&pm);
            assert_eq!(out.shape(), &vec![2, 4]);
            assert_eq!(out.into_shape([-1]).to_vec(), vec![0, 1, 2, 3, 8, 9, 10, 11]);

            // scatter a scalar into the true positions
            let mut c = zeros(([3, 4], &device));
            c.mask_fill(&m, 9);
            assert_eq!(c.into_shape([-1]).to_vec(), vec![9, 0, 9, 0, 0, 0, 0, 9, 0, 0, 9, 0]);
        }
    }

    // `DeviceCpu` is the rayon device under the default features; the sizes
    // below cross its parallel switch, so these compare the parallel mask /
    // take_along_axis kernels with the serial ones.
    #[test]
    #[cfg(feature = "rayon")]
    fn test_mask_indexing_parallel_matches_serial() {
        let mut d_rayon = DeviceCpu::default();
        d_rayon.set_default_order(RowMajor);
        let mut d_serial = DeviceCpuSerial::default();
        d_serial.set_default_order(RowMajor);

        // a prefix mask over the leading axis of a 2-D array: the selected
        // trailing blocks are large enough to cross the switch
        let (rows, cols) = (200_usize, 256_usize);
        let host: Vec<f64> = (0..rows * cols).map(|x| x as f64).collect();
        let sel: Vec<bool> = (0..rows).map(|i| i % 3 != 0).collect();
        let a_r = asarray((host.clone(), &d_rayon)).into_shape((rows, cols));
        let m_r = asarray((sel.clone(), &d_rayon));
        let a_s = asarray((host, &d_serial)).into_shape((rows, cols));
        let m_s = asarray((sel.clone(), &d_serial));

        let v_r = mask_select(&a_r, &m_r).into_shape([-1]).to_vec();
        let v_s = mask_select(&a_s, &m_s).into_shape([-1]).to_vec();
        // the expected blocks are the rows not divisible by 3, in visit order
        let expected: Vec<f64> =
            (0..rows).filter(|i| i % 3 != 0).flat_map(|i| (0..cols).map(move |j| (i * cols + j) as f64)).collect();
        assert_eq!(v_r, expected);
        assert_eq!(v_r, v_s);

        // scatter a scalar into the same selection
        let mut c_r = a_r.clone();
        let mut c_s = a_s.clone();
        c_r.mask_fill(&m_r, -1.0);
        c_s.mask_fill(&m_s, -1.0);
        let v_r = c_r.into_shape([-1]).to_vec();
        let v_s = c_s.into_shape([-1]).to_vec();
        let expected: Vec<f64> = (0..rows * cols).map(|x| if (x / cols) % 3 != 0 { -1.0 } else { x as f64 }).collect();
        assert_eq!(v_r, expected);
        assert_eq!(v_r, v_s);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_take_along_axis_parallel_matches_serial() {
        let mut d_rayon = DeviceCpu::default();
        d_rayon.set_default_order(RowMajor);
        let mut d_serial = DeviceCpuSerial::default();
        d_serial.set_default_order(RowMajor);

        let (rows, cols) = (300_usize, 300_usize);
        let host: Vec<f64> = (0..rows * cols).map(|x| x as f64).collect();
        let idx: Vec<isize> = (0..rows * cols).map(|x| ((x * 7) % cols) as isize).collect();
        let a_r = asarray((host.clone(), &d_rayon)).into_shape((rows, cols));
        let i_r = asarray((idx.clone(), &d_rayon)).into_shape((rows, cols));
        let a_s = asarray((host, &d_serial)).into_shape((rows, cols));
        let i_s = asarray((idx.clone(), &d_serial)).into_shape((rows, cols));

        let v_r = a_r.take_along_axis(&i_r, 1).into_shape([-1]).to_vec();
        let v_s = a_s.take_along_axis(&i_s, 1).into_shape([-1]).to_vec();
        // `a[r, c] = r * cols + c` with `c` the taken column
        let expected: Vec<f64> = (0..rows * cols).map(|x| ((x / cols) * cols + (x * 7) % cols) as f64).collect();
        assert_eq!(v_r, expected);
        assert_eq!(v_r, v_s);

        // take along the leading axis as well (a different rest/axis split)
        let v_r = a_r.take_along_axis(&i_r, 0).into_shape([-1]).to_vec();
        let v_s = a_s.take_along_axis(&i_s, 0).into_shape([-1]).to_vec();
        assert_eq!(v_r, v_s);
    }
}
