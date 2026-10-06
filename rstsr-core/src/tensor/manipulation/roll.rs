//! Roll tensor elements along axes: [`roll`], [`roll_f`].

use crate::prelude_dev::*;

/* #region roll */

/// Roll tensor elements along axes.
///
/// See also [`roll`].
pub fn roll_f<'a, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shift: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    axis: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<T, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    T: Clone,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignArbitaryAPI<T, IxD, D>,
{
    let device = tensor.device().clone();
    let ndim = tensor.ndim();
    let shift = shift.try_into().map_err(Into::into)?;
    let axis = axis.try_into().map_err(Into::into)?;

    match axis {
        AxesIndex::None => {
            // flatten-roll-reshape: C-order visit contract, shape preserved
            let flat: Tensor<T, B, IxD> = {
                let layout: Layout<IxD> = tensor.layout().to_dim()?;
                let out_shape = vec![layout.size()];
                let layout_c = out_shape.new_contig(None, device.default_order());
                let (_, idx_max) = layout_c.bounds_index()?;
                let mut storage = device.uninit_impl(idx_max)?;
                // read in row-major order regardless of storage arrangement:
                // walk row-major offsets and write one element each
                let iter = IndexedIterLayout::new(&layout, RowMajor)?;
                let mut offset = 0_usize;
                if layout.ndim() == 0 {
                    // 0-d input: single element, no axis to select
                    let layout_result = layout_c.dim_narrow(0, slice!(0, 1))?;
                    let (layout_result, layout_src) =
                        broadcast_layout_to_first(&layout_result, &layout, device.default_order())?;
                    device.assign_arbitary_uninit(storage.raw_mut(), &layout_result, tensor.raw(), &layout_src)?;
                } else {
                    for (index, _) in iter {
                        let layout_result = layout_c.dim_narrow(0, slice!(offset as isize, (offset + 1) as isize))?;
                        // `index` is the row-major multi-index of this element
                        let index_vec: &[usize] = index.as_ref();
                        let mut layout_src: Layout<IxD> = layout.clone();
                        // select in descending axis order: each dim_select
                        // removes one axis, keeping lower indices valid
                        for (axis, &pos) in index_vec.iter().enumerate().rev() {
                            layout_src = layout_src.dim_select(axis as isize, pos as isize)?;
                        }
                        let (layout_result, layout_src) =
                            broadcast_layout_to_first(&layout_result, &layout_src, device.default_order())?;
                        device.assign_arbitary_uninit(storage.raw_mut(), &layout_result, tensor.raw(), &layout_src)?;
                        offset += 1;
                    }
                }
                // SAFETY: the row-major iterator above visited every element of
                // the validated layout exactly once, filling `layout_c` fully.
                let storage = unsafe { B::assume_init_impl(storage)? };
                Tensor::new_f(storage, layout_c)?
            };
            // shifts normalized mod size; 0-d/1-elem tensors roll to themselves
            let size = flat.size().max(1);
            let mut shift_all: isize = 0;
            let shifts = match &shift {
                AxesIndex::None => vec![0_isize],
                AxesIndex::Val(v) => vec![*v],
                AxesIndex::Vec(v) => v.clone(),
            };
            for &s in &shifts {
                shift_all = shift_all.wrapping_add(s);
            }
            let shift_norm = shift_all.rem_euclid(size as isize) as usize;
            let rolled = roll_axis_1d(&flat, shift_norm)?;
            let shape_out = tensor.shape().as_ref().to_vec();
            // freshly owned data: reinterpret to the input shape; reading order
            // is C-order by construction (the flatten above), so copy=false with
            // RowMajor is always viewable
            let reshaped = into_shape_with_args(rolled, shape_out, ReshapeArgs::from((TensorOrder::RowMajor, false)));
            Ok(reshaped.into_dim())
        },
        AxesIndex::Val(axis) => {
            let axis = rstsr_check_axis!(axis, ndim)?;
            // a tuple shift on a single axis is summed (NumPy broadcasts it)
            let shift_list = match &shift {
                AxesIndex::None => 0_isize,
                AxesIndex::Val(v) => *v,
                AxesIndex::Vec(v) => v.iter().fold(0_isize, |acc, &s| acc.wrapping_add(s)),
            };
            let out = roll_single_axis(tensor, shift_list, axis)?;
            Ok(out.into_dim())
        },
        AxesIndex::Vec(axes) => {
            // successive single-axis rolls (NumPy's own decomposition);
            // duplicate axes are allowed (each occurrence rolls once)
            let axes = normalize_axes_index(AxesIndex::Vec(axes), ndim, true, false)?;
            let shifts = match &shift {
                AxesIndex::None => vec![0_isize; axes.len()],
                AxesIndex::Val(v) => vec![*v; axes.len()],
                AxesIndex::Vec(v) => {
                    rstsr_assert_eq!(
                        v.len(),
                        axes.len(),
                        InvalidValue,
                        "roll: shift and axis must have the same length."
                    )?;
                    v.clone()
                },
            };
            let mut current: Option<Tensor<T, B, IxD>> = None;
            for (&axis, &shift) in axes.iter().zip(shifts.iter()) {
                let axis = axis as usize;
                current = Some(match &current {
                    None => roll_single_axis(tensor, shift, axis)?,
                    Some(prev) => roll_single_axis(prev, shift, axis)?,
                });
            }
            let result = match current {
                Some(t) => t,
                // empty axes tuple `()`: plain copy (valid input, no roll)
                None => {
                    rstsr_assert!(ndim > 0, InvalidValue, "roll requires ndim > 0.")?;
                    roll_single_axis(tensor, 0, 0)?
                },
            };
            Ok(result.into_dim())
        },
    }
}

/// One-axis roll into a fresh owned tensor; shift already normalized.
fn roll_single_axis<R, T, B, D>(tensor: &TensorAny<R, T, B, D>, shift: isize, axis: usize) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    let device = tensor.device().clone();
    let axis_size = tensor.shape()[axis];
    let layout: Layout<IxD> = tensor.layout().to_dim()?;
    let shape_ref: &[usize] = layout.shape().as_ref();
    let out_shape: Vec<usize> = shape_ref.to_vec();
    let layout_c = out_shape.new_contig(None, device.default_order());
    let (_, idx_max) = layout_c.bounds_index()?;
    let mut storage = device.uninit_impl(idx_max)?;

    if axis_size == 0 {
        // nothing to roll; output is empty as well
        // SAFETY: zero elements; `assume_init_impl` of an empty storage is the
        // crate-wide convention for empty tensors.
        let storage = unsafe { B::assume_init_impl(storage)? };
        return Tensor::new_f(storage, layout_c);
    }
    let s = shift.rem_euclid(axis_size as isize) as usize;
    if s == 0 {
        device.assign_arbitary_uninit(storage.raw_mut(), &layout_c, tensor.raw(), &layout)?;
    } else {
        // split source into the two wrap-around blocks along `axis`
        let split = (axis_size - s) as isize;
        let lo = layout.dim_narrow(axis as isize, slice!(0, split))?;
        let hi = layout.dim_narrow(axis as isize, slice!(split, axis_size as isize))?;
        let out_lo = layout_c.dim_narrow(axis as isize, slice!(s as isize, axis_size as isize))?;
        let out_hi = layout_c.dim_narrow(axis as isize, slice!(0, s as isize))?;
        device.assign_arbitary_uninit(storage.raw_mut(), &out_lo, tensor.raw(), &lo)?;
        device.assign_arbitary_uninit(storage.raw_mut(), &out_hi, tensor.raw(), &hi)?;
    }
    // SAFETY: the two (or one) assignments above partition the rolled axis of
    // `layout_c` exactly once.
    let storage = unsafe { B::assume_init_impl(storage)? };
    Tensor::new_f(storage, layout_c)
}

/// 1-D roll of an owned tensor (fresh output); shift pre-normalized.
fn roll_axis_1d<T, B>(tensor: &Tensor<T, B, IxD>, shift: usize) -> Result<Tensor<T, B, IxD>>
where
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    roll_single_axis(tensor, shift as isize, 0)
}

/// Roll array elements along a given axis.
///
/// Elements that roll beyond the last position are re-introduced at the first.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// Per-axis rolls behave identically under [`RowMajor`] and [`ColMajor`] device
/// default orders (the new tensor's memory arrangement follows the device
/// default order). The flattened form (`axis = None`) always visits elements in
/// row-major order, and its result is reinterpreted to the input shape in
/// row-major order — so on a [`ColMajor`] device, the flattened form returns a
/// row-major-strided tensor (values follow NumPy; the memory arrangement is
/// row-major regardless of the device default).
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - `shift`: TryInto [`AxesIndex<isize>`]
///
///   - Number of positions by which elements are shifted.
///   - A single integer, or a tuple/list matching the length of `axis` (a tuple shift on a single
///     axis is summed, NumPy-compatible).
///   - A single integer with a tuple of axes shifts every listed axis by the same amount
///     (NumPy-compatible).
///   - Shifts larger than the axis length wrap (modulo arithmetic); negative values shift in the
///     opposite direction.
///
/// - `axis`: TryInto [`AxesIndex<isize>`]
///
///   - The axis or axes along which elements are shifted.
///   - `None` (default): the tensor is flattened (row-major sequence), rolled, and restored to the
///     input shape.
///   - Duplicate axes are allowed (the roll is applied once per occurrence).
///   - Negative values count from the back.
///
/// # Returns
///
/// - [`Tensor<T, B, D>`][`Tensor`]
///
///   - A new owned tensor with the same shape as the input; the input is not modified.
///
/// # Examples
///
/// Rolling a 1-D tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device));
/// println!("{}", rt::roll((&a, 2, None)));
/// // [ 4 5 0 1 2 3]
/// # assert_eq!(format!("{}", rt::roll((&a, 2, None))), "[ 4 5 0 1 2 3]");
/// ```
///
/// Rolling along one axis, and along two axes at once:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((10, &device)).into_shape([2, 5]);
/// println!("{}", rt::roll((&a, 1, 0)));
/// // [[ 5 6 7 8 9]
/// //  [ 0 1 2 3 4]]
/// println!("{}", rt::roll((&a, (1, 1), (0, 1))));
/// // [[ 9 5 6 7 8]
/// //  [ 4 0 1 2 3]]
/// # let b = rt::roll((&a, (1, 1), (0, 1)));
/// # assert_eq!(format!("{b}"), "[[ 9 5 6 7 8]\n [ 4 0 1 2 3]]");
/// ```
///
/// Flattened roll (`axis = None`), shape preserved:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((10, &device)).into_shape([2, 5]);
/// println!("{}", rt::roll((&a, 1, None)));
/// // [[ 9 0 1 2 3]
/// //  [ 4 5 6 7 8]]
/// # let b = rt::roll((&a, 1, None));
/// # assert_eq!(format!("{b}"), "[[ 9 0 1 2 3]\n [ 4 5 6 7 8]]");
/// ```
///
/// # Overloads Table
///
/// Output is [`Tensor<T, B, D>`][`Tensor`] (same shape as the input).
///
/// - `roll((tensor, shift)) -> Tensor<T, B, D>` (implicit `axis = None`, flattened)
/// - `roll((tensor, shift, axis)) -> Tensor<T, B, D>` where `shift` and `axis` are any
///   `TryInto<AxesIndex<isize>>` forms (integer, tuple, list, `None`)
///
/// # Notes of API accordance
///
/// - Array-API: `roll(x, /, shift, *, axis=None)` ([`roll`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.roll.html))
/// - NumPy: `numpy.roll(a, shift, axis=None)` ([`numpy.roll`](https://numpy.org/doc/stable/reference/generated/numpy.roll.html))
/// - RSTSR: `rt::roll((tensor, shift, axis))`
///
/// RSTSR's behavior matches NumPy and Array-API, including the tuple-shift /
/// tuple-axis combinations, same-axis repeats, and the tuple-shift-on-single-axis
/// summing behavior.
///
/// # Panics
///
/// - Panics if any axis is out of range, if the lengths of a tuple `shift` and tuple `axis` differ,
///   or if the tensor is 0-dimensional and an axes tuple is given.
///
/// For a fallible version, use [`roll_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`repeat`]: repeat individual elements instead of the whole tensor.
/// - [`flip`]: reverse the order of elements along axes.
///
/// ## Variants of this function
///
/// - [`roll_f`]: fallible version.
/// - [`TensorAny::roll`]: associated method.
/// - [`TensorAny::roll_f`]: associated fallible method.
pub fn roll<Args, Inp>(args: Args) -> Args::Out
where
    Args: RollAPI<Inp>,
{
    Args::roll(args)
}

/// API trait backing [`roll`].
pub trait RollAPI<Inp> {
    type Out;

    fn roll_f(self) -> Result<Self::Out>;
    fn roll(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::roll_f(self).rstsr_unwrap()
    }
}

impl<'a, RA, T, B, D, SArg, AArg> RollAPI<()> for (&'a TensorAny<RA, T, B, D>, SArg, AArg)
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    SArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    AArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    T: Clone,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignArbitaryAPI<T, IxD, D>,
{
    type Out = Tensor<T, B, D>;

    fn roll_f(self) -> Result<Self::Out> {
        let (tensor, shift, axis) = self;
        roll_f(tensor, shift, axis)
    }
}

impl<'a, RA, T, B, D, SArg> RollAPI<()> for (&'a TensorAny<RA, T, B, D>, SArg)
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    SArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    T: Clone,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignArbitaryAPI<T, IxD, D>,
{
    type Out = Tensor<T, B, D>;

    fn roll_f(self) -> Result<Self::Out> {
        let (tensor, shift) = self;
        roll_f(tensor, shift, AxesIndex::<isize>::None)
    }
}

impl<'a, RA, T, B, D> TensorAny<RA, T, B, D>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    T: Clone,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignArbitaryAPI<T, IxD, D>,
{
    /// Roll array elements along a given axis.
    ///
    /// See also [`roll`].
    pub fn roll_f<SArg, AArg>(&'a self, shift: SArg, axis: AArg) -> Result<Tensor<T, B, D>>
    where
        SArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
        AArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        roll_f(self, shift, axis)
    }

    /// Roll array elements along a given axis.
    ///
    /// See also [`roll`].
    pub fn roll<SArg, AArg>(&'a self, shift: SArg, axis: AArg) -> Tensor<T, B, D>
    where
        SArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
        AArg: TryInto<AxesIndex<isize>, Error: Into<Error>>,
    {
        roll_f(self, shift, axis).rstsr_unwrap()
    }
}

/* #endregion */
