//! Repeat elements of a tensor: [`repeat`], [`repeat_f`], with argument type
//! [`RepeatArg`].

use crate::prelude_dev::*;

/* #region RepeatArg */

/// Repeats argument for [`repeat`]: a single count for every element, or
/// per-element counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepeatArg {
    /// Every element repeats this many times.
    All(usize),
    /// Per-element repeat counts; see [`repeat`] for the accepted lengths.
    Elems(Vec<usize>),
}

impl From<usize> for RepeatArg {
    fn from(value: usize) -> Self {
        Self::All(value)
    }
}

impl From<&usize> for RepeatArg {
    fn from(value: &usize) -> Self {
        Self::All(*value)
    }
}

impl From<Vec<usize>> for RepeatArg {
    fn from(value: Vec<usize>) -> Self {
        Self::Elems(value)
    }
}

impl From<&Vec<usize>> for RepeatArg {
    fn from(value: &Vec<usize>) -> Self {
        Self::Elems(value.clone())
    }
}

impl From<&[usize]> for RepeatArg {
    fn from(value: &[usize]) -> Self {
        Self::Elems(value.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for RepeatArg {
    fn from(value: [usize; N]) -> Self {
        Self::Elems(value.into_iter().collect())
    }
}

impl<const N: usize> From<&[usize; N]> for RepeatArg {
    fn from(value: &[usize; N]) -> Self {
        Self::Elems(value.to_vec())
    }
}

/* #endregion */

/* #region repeat */

/// Repeat elements of a tensor.
///
/// See also [`repeat`].
pub fn repeat_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    repeats: impl Into<RepeatArg>,
    axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    let repeats = repeats.into();
    let axis = axis.try_into().map_err(Into::into)?;
    let device = tensor.device().clone();
    let ndim = tensor.ndim();

    // normalize axis: at most one axis; None means flatten
    let axis = match axis {
        AxesIndex::None => None,
        AxesIndex::Val(v) => Some(rstsr_check_axis!(v, ndim)?),
        AxesIndex::Vec(_) => return rstsr_raise!(InvalidValue, "repeat accepts at most one axis."),
    };

    // resolve repeat counts per repeated position; the scalar forms are kept
    // lazy (no O(size) materialization) via a count-at accessor
    let n_rep = match axis {
        None => tensor.size(),
        Some(axis) => tensor.shape()[axis],
    };
    let counts_vec: Option<Vec<usize>> = match &repeats {
        RepeatArg::All(_) => None,
        RepeatArg::Elems(v) if v.len() == 1 => None,
        RepeatArg::Elems(v) => {
            rstsr_assert_eq!(
                v.len(),
                n_rep,
                InvalidLayout,
                "repeat: number of repeat counts must equal the repeated axis size (or total size when axis is None)."
            )?;
            Some(v.clone())
        },
    };
    // broadcast count for the scalar / length-1 forms (None => lazy count_at)
    let all_count: Option<usize> = match (&repeats, &counts_vec) {
        (RepeatArg::All(n), _) => Some(*n),
        (RepeatArg::Elems(v), None) => Some(v[0]),
        (_, Some(_)) => None,
    };
    let count_at = |pos: usize| match (&counts_vec, all_count) {
        (Some(v), _) => v[pos],
        (None, Some(n)) => n,
        (None, None) => unreachable!("scalar forms always set all_count"),
    };
    let out_size: usize = match &counts_vec {
        Some(v) => v.iter().sum(),
        None => all_count.unwrap_or(0) * n_rep,
    };

    // allocate output, contiguous in device default order
    let out_shape = match axis {
        None => vec![out_size],
        Some(axis) => {
            let mut shape = tensor.shape().as_ref().to_vec();
            shape[axis] = out_size;
            shape
        },
    };
    let layout_c = out_shape.new_contig(None, device.default_order());
    let (_, idx_max) = layout_c.bounds_index()?;
    let mut storage = device.uninit_impl(idx_max)?;

    match axis {
        None => {
            // flattened repeat: strict row-major (C-order) visit order
            let layout: Layout<IxD> = tensor.layout().to_dim()?;
            let ndim_in = layout.ndim();
            if ndim_in == 0 {
                // 0-d input: the single element repeats into a 1-D output
                let layout_result = layout_c.dim_narrow(0, slice!(0, out_size as isize))?;
                let (layout_result, layout_src) =
                    broadcast_layout_to_first(&layout_result, &layout, device.default_order())?;
                device.assign_arbitary_uninit(storage.raw_mut(), &layout_result, tensor.raw(), &layout_src)?;
            } else {
                let iter = IndexedIterLayout::new(&layout, RowMajor)?;
                let mut offset = 0;
                for (flat, (index, _)) in iter.enumerate() {
                    let count = count_at(flat);
                    if count == 0 {
                        continue;
                    }
                    let layout_result = layout_c.dim_narrow(0, slice!(offset as isize, (offset + count) as isize))?;
                    // select the element: eliminate axes in descending order so
                    // each dim_select's axis index stays valid as axes are removed
                    let index_vec: &[usize] = index.as_ref();
                    let mut layout_src: Layout<IxD> = layout.clone();
                    for (axis_i, &pos) in index_vec.iter().enumerate().rev() {
                        layout_src = layout_src.dim_select(axis_i as isize, pos as isize)?;
                    }
                    // broadcast the 1-element source over the output slice
                    let (layout_result, layout_src) =
                        broadcast_layout_to_first(&layout_result, &layout_src, device.default_order())?;
                    device.assign_arbitary_uninit(storage.raw_mut(), &layout_result, tensor.raw(), &layout_src)?;
                    offset += count;
                }
            }
        },
        Some(axis) => {
            // repeat along one axis: for each element k of the axis, assign
            // `count` copies as one slab spanning all other axes — the slab's
            // source keeps the input's real strides on the non-repeated axes
            // and reads a single (stride-0) element along the repeated axis
            let layout: Layout<IxD> = tensor.layout().to_dim()?;
            let axis_stride = layout.stride()[axis];
            let axis_base = layout.offset();
            let mut offset = 0;
            let layout_in = layout;
            let stride_ref: &[isize] = layout_in.stride().as_ref();
            let src_stride_base: Vec<isize> = stride_ref.to_vec();
            for k in 0..tensor.shape()[axis] {
                let count = count_at(k);
                if count == 0 {
                    continue;
                }
                let elem_offset = (axis_base as isize + axis_stride * k as isize) as usize;
                let layout_result =
                    layout_c.dim_narrow(axis as isize, slice!(offset as isize, (offset + count) as isize))?;
                // source layout: output-slice shape, input strides, with the
                // repeated axis stride forced to 0 — every element of the
                // repeated axis reads the same source element
                let mut src_shape: Vec<usize> = tensor.shape().as_ref().to_vec();
                let mut src_stride: Vec<isize> = src_stride_base.clone();
                src_shape[axis] = count;
                src_stride[axis] = 0;
                // SAFETY: strides and offset derive from the validated input
                // layout with only the repeated axis stride forced to 0 and
                // offset advanced along the (validated) axis — all addresses
                // stay within the input's own storage.
                let layout_elem = unsafe { Layout::new_unchecked(src_shape, src_stride, elem_offset) };
                device.assign_arbitary_uninit(storage.raw_mut(), &layout_result, tensor.raw(), &layout_elem)?;
                offset += count;
            }
        },
    }

    // SAFETY: the `assign_uninit` calls above initialized every element of
    // `layout_c` exactly once (slices partition the repeated axis / flat axis).
    let storage = unsafe { B::assume_init_impl(storage)? };
    Tensor::new_f(storage, layout_c)
}

/// Repeat elements of a tensor.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order; the element visit order of the flattened form is
/// always row-major.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - `repeats`: TryInto [`RepeatArg`]
///
///   - A single count (`usize`): every element repeats this many times.
///   - A list of counts (`Vec<usize>` or slice): per-element counts. Its length must be `1`
///     (treated as the single-count form), the size of the selected axis, or the total size when
///     `axis` is `None`.
///   - A count of zero drops the element (or empties the selected axis).
///
/// - `axis`: TryInto [`AxesIndex<isize>`]
///
///   - A single axis: only elements along that axis are repeated; other dimensions are preserved.
///   - `None` (default): the tensor is flattened in strict row-major (C-order) sequence, elements
///     are repeated, and a 1-D tensor is returned. The visit order does not depend on the device
///     default order.
///   - Negative values count from the back.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]
///
///   - A new owned tensor; the input is not modified.
///   - Shape is the input shape with the selected axis replaced by the sum of the repeat counts
///     (1-D when `axis` is `None`).
///
/// # Examples
///
/// Repeating every element (scalar count):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// println!("{}", rt::repeat((&a, 2, None)));
/// // [ 0 0 1 1 2 2]
/// # assert_eq!(format!("{}", rt::repeat((&a, 2, None))), "[ 0 0 1 1 2 2]");
/// ```
///
/// Per-element counts along a given axis:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// println!("{}", rt::repeat((&a, [2, 1], 0)));
/// // [[ 0 1 2]
/// //  [ 0 1 2]
/// //  [ 3 4 5]]
/// # assert_eq!(format!("{}", rt::repeat((&a, [2, 1], 0))), "[[ 0 1 2]\n [ 0 1 2]\n [ 3 4 5]]");
/// ```
///
/// Flattening repeat (`axis = None`):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// println!("{}", rt::repeat((&a, 1, None)));
/// // [ 0 1 2 3 4 5]
/// # assert_eq!(format!("{}", rt::repeat((&a, 1, None))), "[ 0 1 2 3 4 5]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `repeat(x, repeats, /, *, axis=None)` ([`repeat`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.repeat.html))
/// - NumPy: `numpy.repeat(a, repeats, axis=None)` ([`numpy.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html))
/// - RSTSR: `rt::repeat((tensor, repeats, axis))`
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, IxD>`][`Tensor`].
///
/// - `repeat((tensor, repeats)) -> Tensor<T, B, IxD>` (implicit `axis = None`, flattened)
/// - `repeat((tensor, repeats, axis)) -> Tensor<T, B, IxD>` where `axis` is any
///   `TryInto<AxesIndex<isize>>` form (integer, tuple, list, `None`)
///
/// RSTSR's behavior matches NumPy and Array-API; `repeats` of an integer
/// array should be passed as host-side counts (`Vec<usize>`), mirroring
/// rstsr's convention that index-like host arguments are plain sequences.
///
/// # Panics
///
/// - Panics if `axis` (after normalization) is not within range, if more than one axis is given, or
///   if the number of repeat counts matches neither `1` nor the repeated axis size.
///
/// For a fallible version, use [`repeat_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`tile`]: repeat the whole tensor (block-wise) instead of its elements.
/// - [`concat`]: join tensors along an axis (the composition building block).
///
/// ## Variants of this function
///
/// - [`repeat_f`]: fallible version.
/// - [`TensorAny::repeat`]: associated method.
/// - [`TensorAny::repeat_f`]: associated fallible method.
pub fn repeat<Args, Inp>(args: Args) -> Args::Out
where
    Args: RepeatAPI<Inp>,
{
    Args::repeat(args)
}

/// API trait backing [`repeat`].
pub trait RepeatAPI<Inp> {
    type Out;

    fn repeat_f(self) -> Result<Self::Out>;
    fn repeat(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::repeat_f(self).rstsr_unwrap()
    }
}

impl<RA, T, B, D, RArg, AArg> RepeatAPI<()> for (&TensorAny<RA, T, B, D>, RArg, AArg)
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    RArg: Into<RepeatArg>,
    AArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn repeat_f(self) -> Result<Self::Out> {
        let (tensor, repeats, axis) = self;
        repeat_f(tensor, repeats, axis)
    }
}

impl<RA, T, B, D, RArg> RepeatAPI<()> for (&TensorAny<RA, T, B, D>, RArg)
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    RArg: Into<RepeatArg>,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn repeat_f(self) -> Result<Self::Out> {
        let (tensor, repeats) = self;
        repeat_f(tensor, repeats, AxesIndex::<isize>::None)
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    /// Repeat elements of a tensor.
    ///
    /// See also [`repeat`].
    pub fn repeat_f(
        &self,
        repeats: impl Into<RepeatArg>,
        axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<Tensor<T, B, IxD>> {
        repeat_f(self, repeats, axis)
    }

    /// Repeat elements of a tensor.
    ///
    /// See also [`repeat`].
    pub fn repeat(
        &self,
        repeats: impl Into<RepeatArg>,
        axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Tensor<T, B, IxD> {
        repeat_f(self, repeats, axis).rstsr_unwrap()
    }
}

/* #endregion */
