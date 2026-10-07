//! Tile a tensor by repeating it along axes: [`tile`], [`tile_f`].

use crate::prelude_dev::*;

/* #region tile */

/// Tile a tensor by repeating it along axes.
///
/// See also [`tile`].
pub fn tile_f<T, B, D>(
    tensor: impl TensorViewAPI<Type = T, Backend = B, Dim = D>,
    repetitions: impl TryInto<AxesIndex<usize>, Error: Into<Error>>,
) -> Result<Tensor<T, B, IxD>>
where
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let ndim = tensor.ndim();
    let repetitions = repetitions.try_into().map_err(Into::into)?;
    let mut reps: Vec<usize> = match repetitions {
        AxesIndex::None => vec![],
        AxesIndex::Val(v) => vec![v],
        AxesIndex::Vec(v) => v,
    };

    // rank promotion (NumPy `tile`): prepend 1s to reps when it is shorter
    if reps.len() < ndim {
        let mut padded = vec![1_usize; ndim - reps.len()];
        padded.extend_from_slice(&reps);
        reps = padded;
    }

    // tiled output shape (reps longer than ndim means leading singleton axes)
    let tiled_shape: Vec<usize> = if reps.len() > ndim {
        let n_insert = reps.len() - ndim;
        let mut shape = vec![1_usize; n_insert];
        shape.extend_from_slice(tensor.shape().as_ref());
        shape.iter().zip(reps.iter()).map(|(&s, &r)| s * r).collect()
    } else {
        tensor.shape().as_ref().iter().zip(reps.iter()).map(|(&s, &r)| s * r).collect()
    };
    let ndim_out = tiled_shape.len();

    // allocate output, contiguous in device default order
    let layout_c = tiled_shape.new_contig(None, device.default_order());
    let (_, idx_max) = layout_c.bounds_index()?;
    let mut storage = device.uninit_impl(idx_max)?;

    // assign the input once per index-tuple of the repetition grid (the full
    // reps vector); each block is an output sub-view of the input's own shape
    // (length 1 on the leading promoted axes)
    let layout: Layout<IxD> = tensor.layout().to_dim()?;
    let grid_shape: Vec<usize> = reps.clone();
    // the repetition grid is only a shape to enumerate: reuse the common
    // layout iterator for its row-major multi-index sequence
    let grid_layout = grid_shape.new_c_contig(None);
    for (multi, _) in IndexedIterLayout::new(&grid_layout, RowMajor)? {
        let mut start = vec![0_usize; ndim_out];
        let in_shape = tensor.shape().as_ref();
        let offset_idx = if reps.len() > ndim { reps.len() - ndim } else { 0 };
        for (i, &m) in multi.iter().enumerate() {
            let len = if i >= offset_idx { in_shape[i - offset_idx] } else { 1 };
            start[i] = m * len;
        }
        let mut layout_block = layout_c.clone();
        for i in 0..ndim_out {
            let len = if i >= offset_idx { in_shape[i - offset_idx] } else { 1 };
            layout_block = layout_block.dim_narrow(i as isize, slice!(start[i] as isize, (start[i] + len) as isize))?;
        }
        device.assign_arbitary_uninit(storage.raw_mut(), &layout_block, tensor.raw(), &layout)?;
    }

    // SAFETY: the grid of assignments above partitions `layout_c` exactly once
    // (blocks are disjoint and cover the tiled shape).
    let storage = unsafe { B::assume_init_impl(storage)? };
    Tensor::new_f(storage, layout_c)
}

/// Construct an array by tiling an input array.
///
/// The result has dimensionality `max(ndim(x), len(repetitions))`: if
/// `repetitions` is shorter than `ndim(x)`, ones are prepended to it; if it is
/// longer, singleton axes are prepended to `x`. This function behaves
/// identically under [`RowMajor`] and [`ColMajor`] device default orders. (Only
/// the memory arrangement of the new tensor follows the device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - `repetitions`: TryInto [`AxesIndex<usize>`]
///
///   - The number of repetitions of `x` along each axis.
///   - A single integer tiles along a new last axis (`repetitions = [n]`).
///   - A tuple/list gives the repetition count per axis, promoted as described above. Repetitions
///     may be zero (empty output axis).
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]
///
///   - A new owned tensor of shape `repetitions[i] * x.shape[i]` (after rank promotion); the input
///     is not modified. rstsr functions always return fresh data, so a fully-ones `repetitions`
///     also produces a copy (NumPy copies in that case too, since its gh4679 fix).
///
/// # Examples
///
/// Tiling a 1-D tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// println!("{}", rt::tile((&a, 2)));
/// // [ 0 1 2 0 1 2]
/// # assert_eq!(format!("{}", rt::tile((&a, 2))), "[ 0 1 2 0 1 2]");
/// ```
///
/// Tiling into a higher-rank result:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((3, &device));
/// println!("{}", rt::tile((&a, [2, 2])));
/// // [[ 0 1 2 0 1 2]
/// //  [ 0 1 2 0 1 2]]
/// # let b = rt::tile((&a, [2, 2]));
/// # assert_eq!(format!("{b}"), "[[ 0 1 2 0 1 2]\n [ 0 1 2 0 1 2]]");
/// ```
///
/// Tiling a 2-D tensor along the second axis only:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
/// println!("{}", rt::tile((&a, [1, 2])));
/// // [[ 1 2 1 2]
/// //  [ 3 4 3 4]]
/// # let b = rt::tile((&a, [1, 2]));
/// # assert_eq!(format!("{b}"), "[[ 1 2 1 2]\n [ 3 4 3 4]]");
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `tile(x, repetitions, /)` ([`tile`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.tile.html))
/// - NumPy: `numpy.tile(A, reps)` ([`numpy.tile`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html))
/// - RSTSR: `rt::tile((tensor, repetitions))`
///
/// # Panics
///
/// - Panics if the repetition counts cannot combine with the input shape (only possible through
///   internal errors; all shapes are valid inputs).
///
/// For a fallible version, use [`tile_f`].
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`repeat`]: repeat individual elements instead of the whole tensor.
/// - [`to_broadcast`]: stride-0 view without materializing.
///
/// ## Variants of this function
///
/// - [`tile_f`]: fallible version.
/// - [`TensorAny::tile`]: associated method.
/// - [`TensorAny::tile_f`]: associated fallible method.
pub fn tile<Args, Inp>(args: Args) -> Args::Out
where
    Args: TileAPI<Inp>,
{
    Args::tile(args)
}

/// API trait backing [`tile`].
pub trait TileAPI<Inp> {
    type Out;

    fn tile_f(self) -> Result<Self::Out>;
    fn tile(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::tile_f(self).rstsr_unwrap()
    }
}

impl<RA, T, B, D, RArg> TileAPI<()> for (&TensorAny<RA, T, B, D>, RArg)
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    RArg: TryInto<AxesIndex<usize>, Error: Into<Error>>,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn tile_f(self) -> Result<Self::Out> {
        let (tensor, repetitions) = self;
        tile_f(tensor, repetitions)
    }
}

impl<T, B, D, RArg> TileAPI<()> for (TensorView<'_, T, B, D>, RArg)
where
    D: DimAPI,
    RArg: TryInto<AxesIndex<usize>, Error: Into<Error>>,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    type Out = Tensor<T, B, IxD>;

    fn tile_f(self) -> Result<Self::Out> {
        let (tensor, repetitions) = self;
        tile_f(tensor, repetitions)
    }
}

impl<RA, T, B, D> TensorAny<RA, T, B, D>
where
    RA: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, IxD>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    /// Construct an array by tiling an input array.
    ///
    /// See also [`tile`].
    pub fn tile_f<RArg>(&self, repetitions: RArg) -> Result<Tensor<T, B, IxD>>
    where
        RArg: TryInto<AxesIndex<usize>, Error: Into<Error>>,
    {
        tile_f(self, repetitions)
    }

    /// Construct an array by tiling an input array.
    ///
    /// See also [`tile`].
    pub fn tile<RArg>(&self, repetitions: RArg) -> Tensor<T, B, IxD>
    where
        RArg: TryInto<AxesIndex<usize>, Error: Into<Error>>,
    {
        tile_f(self, repetitions).rstsr_unwrap()
    }
}

/* #endregion */
