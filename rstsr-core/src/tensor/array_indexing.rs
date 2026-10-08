//! Array indexing (fancy indexing) by integer arrays.
//!
//! Contrary to basic slicing ([`slice`](crate::tensor::indexing::slice())),
//! which only moves the layout, array indexing gathers elements and therefore
//! returns an owned tensor (or a view, when no array indexer is actually
//! involved; see [`array_index`]).

use crate::prelude_dev::*;

/// One parsed index entry, describing what one group of the index does.
enum Entry {
    /// A slice on `axis`.
    Slice { axis: usize, slice: SliceI },
    /// An integer selection on `axis`, dropping it.
    Select { axis: usize, index: usize },
    /// A new axis (size-1 dimension).
    Insert,
    /// An integer array indexing `src_axis`.
    Array { src_axis: usize, indices: Vec<usize>, layout: Layout<IxD> },
}

/// Indexes a tensor by integer arrays (array indexing, *fancy indexing*).
///
/// See also [`array_index`].
#[allow(clippy::type_complexity)]
pub fn array_index_f<'a, R, T, B, D, I>(
    tensor: &'a TensorAny<R, T, B, D>,
    indexer: I,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<isize, Raw = Vec<isize>>
        + DeviceAPI<bool, Raw = Vec<bool>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceArrayIndexAPI<T>,
    I: TryInto<ArrayIndexArgs<B>, Error: Into<Error>>,
{
    let device = tensor.device().clone();
    let la = tensor.layout().to_dim::<IxD>()?;
    let ndim = la.ndim();

    let indexers = indexer.try_into().map_err(Into::into)?.indexers;

    // lower host carriers; a zero-dimensional integer tensor is a scalar index
    enum Lowered {
        Basic(Indexer),
        Array {
            values: Vec<isize>,
            layout: Layout<IxD>,
        },
        Bool,
        /// A zero-width ellipsis: it selects nothing, but still separates the
        /// advanced indexers around it (NumPy's placement rule).
        Noop,
    }

    let mut lowered: Vec<Lowered> = Vec::with_capacity(indexers.len());
    for indexer in indexers {
        match indexer {
            ArrayIndexer::Basic(indexer) => lowered.push(Lowered::Basic(indexer)),
            ArrayIndexer::OneDimIndex(values) => {
                let layout = vec![values.len()].new_c_contig(None);
                lowered.push(Lowered::Array { values, layout });
            },
            ArrayIndexer::OneDimBool(_) | ArrayIndexer::ArrayBool(_) => lowered.push(Lowered::Bool),
            ArrayIndexer::ArrayIndex(index) => {
                rstsr_assert!(
                    device.same_device(index.device()),
                    DeviceMismatch,
                    "array_index requires the index arrays on the same device as the tensor."
                )?;
                if index.ndim() == 0 {
                    // a zero-dimensional integer array is an integer index
                    let offset = index.layout().index_uncheck(&[]) as usize;
                    lowered.push(Lowered::Basic(Indexer::Select(index.raw()[offset])));
                    continue;
                }
                let layout = index.layout().to_dim::<IxD>()?;
                let mut values = Vec::with_capacity(index.size());
                for (_, offset) in IndexedIterLayout::<IxD>::new(&layout, RowMajor)? {
                    values.push(index.raw()[offset]);
                }
                let layout = layout.shape().clone().new_c_contig(None);
                lowered.push(Lowered::Array { values, layout });
            },
        }
    }

    // expand ellipsis and validate the number of consumed axes
    let mut consumed = 0_usize;
    let mut n_ellipsis = 0_usize;
    for entry in &lowered {
        match entry {
            Lowered::Basic(Indexer::Insert) => {},
            Lowered::Basic(Indexer::Ellipsis) => n_ellipsis += 1,
            _ => consumed += 1,
        }
    }
    rstsr_assert!(n_ellipsis <= 1, InvalidValue, "Only one ellipsis indexer is allowed in array indexing.")?;
    rstsr_assert!(
        consumed <= ndim,
        IndexError,
        "Too many indices for the tensor: the index consumes {} axes, but the tensor has only {}.",
        consumed,
        ndim
    )?;
    let n_fill = ndim - consumed;

    // without any array indexer, array indexing degenerates to basic slicing,
    // which is a view
    if !lowered.iter().any(|e| matches!(e, Lowered::Array { .. })) {
        let mut basic: Vec<Indexer> = Vec::with_capacity(lowered.len());
        for entry in lowered {
            match entry {
                Lowered::Basic(indexer) => basic.push(indexer),
                Lowered::Bool => rstsr_raise!(
                    UnImplemented,
                    "boolean-array indexing is not supported by array_index; use mask_select or bool_select instead."
                )?,
                Lowered::Array { .. } => unreachable!(),
                Lowered::Noop => {},
            }
        }
        return Ok(into_slice_f(tensor.view(), AxesIndex::<Indexer>::Vec(basic))?.into_cow());
    }

    // if any boolean indexer is present, refuse (only a sole boolean array is
    // supported elsewhere, through `mask_select`)
    if lowered.iter().any(|e| matches!(e, Lowered::Bool)) {
        rstsr_raise!(
            UnImplemented,
            "boolean-array indexing is not supported by array_index; use mask_select or bool_select instead."
        )?;
    }

    let mut expanded: Vec<Lowered> = Vec::with_capacity(lowered.len() + n_fill);
    for entry in lowered {
        match entry {
            Lowered::Basic(Indexer::Ellipsis) => {
                if n_fill == 0 {
                    expanded.push(Lowered::Noop);
                } else {
                    for _ in 0..n_fill {
                        expanded.push(Lowered::Basic(Indexer::Slice(SliceI::new(None, None, None))));
                    }
                }
            },
            other => expanded.push(other),
        }
    }
    if n_ellipsis == 0 {
        for _ in 0..n_fill {
            expanded.push(Lowered::Basic(Indexer::Slice(SliceI::new(None, None, None))));
        }
    }

    // walk the index: resolve entries, compute the placement of the broadcast
    // index dimensions (NumPy's rule: in place when the advanced indexers are
    // consecutive, at the front otherwise)
    let mut entries: Vec<Entry> = Vec::with_capacity(expanded.len());
    let mut curr_axis = 0_usize;
    let mut result_dim = 0_usize;
    let mut consec = 0_usize;
    let mut consec_status = -1_i8;
    let mut fancy_ndim = 0_usize;
    for entry in expanded {
        let advanced = matches!(&entry, Lowered::Basic(Indexer::Select(_)) | Lowered::Array { .. });
        if advanced {
            if consec_status == -1 {
                consec = result_dim;
                consec_status = 0;
            } else if consec_status == 1 {
                consec_status = 2;
                consec = 0;
            }
        } else if consec_status == 0 {
            consec_status = 1;
        }
        match entry {
            Lowered::Basic(Indexer::Slice(slice)) => {
                entries.push(Entry::Slice { axis: curr_axis, slice });
                curr_axis += 1;
                result_dim += 1;
            },
            Lowered::Basic(Indexer::Select(index)) => {
                let axis_size = la.shape()[curr_axis];
                let index = if index < 0 { index + axis_size as isize } else { index };
                rstsr_pattern!(
                    index,
                    0..axis_size as isize,
                    IndexError,
                    "Integer index out of range along axis {}.",
                    curr_axis
                )?;
                entries.push(Entry::Select { axis: curr_axis, index: index as usize });
                curr_axis += 1;
            },
            Lowered::Basic(Indexer::Insert) => {
                entries.push(Entry::Insert);
                result_dim += 1;
            },
            // `Indexer` is `non_exhaustive`: every variant is handled above
            Lowered::Basic(_) => unreachable!(),
            Lowered::Bool => unreachable!(),
            Lowered::Noop => {},
            Lowered::Array { values, layout } => {
                let axis_size = la.shape()[curr_axis] as isize;
                let mut indices = Vec::with_capacity(values.len());
                for value in values {
                    let value = if value < 0 { value + axis_size } else { value };
                    rstsr_pattern!(
                        value,
                        0..axis_size,
                        IndexError,
                        "Array index out of range along axis {}.",
                        curr_axis
                    )?;
                    indices.push(value as usize);
                }
                fancy_ndim = fancy_ndim.max(layout.ndim());
                entries.push(Entry::Array { src_axis: curr_axis, indices, layout });
                curr_axis += 1;
            },
        }
    }
    rstsr_assert_eq!(curr_axis, ndim, Miscellaneous, "Internal error: array index did not consume all axes.")?;

    // layout of the non-array-indexed subspace (slices and new axes), with the
    // integer selections already folded into its offset
    let mut base_shape: Vec<usize> = Vec::with_capacity(entries.len());
    let mut base_stride: Vec<isize> = Vec::with_capacity(entries.len());
    let mut base_offset = la.offset() as isize;
    for entry in &entries {
        match entry {
            Entry::Slice { axis, slice } => {
                let narrowed = la.dim_narrow(*axis as isize, *slice)?;
                base_shape.push(narrowed.shape()[*axis]);
                base_stride.push(narrowed.stride()[*axis]);
                base_offset += narrowed.offset() as isize - la.offset() as isize;
            },
            Entry::Select { axis, index } => {
                base_offset += la.stride()[*axis] * *index as isize;
            },
            Entry::Insert => {
                base_shape.push(1);
                base_stride.push(0);
            },
            Entry::Array { .. } => {},
        }
    }
    rstsr_assert!(base_offset >= 0, InvalidLayout, "Array indexing produced a layout with negative offset.")?;
    let base_layout = Layout::<IxD>::new(base_shape, base_stride, base_offset as usize)?;

    // broadcast shape of the index arrays (trailing-aligned)
    let mut bulk_shape = vec![1_usize; fancy_ndim];
    for entry in &entries {
        if let Entry::Array { layout, .. } = entry {
            let ndim_idx = layout.ndim();
            for d in 0..ndim_idx {
                let dim = layout.shape()[d];
                if dim != 1 {
                    let broadcast = bulk_shape[fancy_ndim - ndim_idx + d];
                    if broadcast != 1 && broadcast != dim {
                        rstsr_raise!(
                            IndexError,
                            "shape mismatch: indexing arrays could not be broadcast together with shapes {} and {}.",
                            DebugShape(&bulk_shape),
                            DebugShape(&layout.shape()[..])
                        )?;
                    }
                    bulk_shape[fancy_ndim - ndim_idx + d] = dim;
                }
            }
        }
    }

    // output layout: the broadcast dimensions inserted at `consec`
    let mut out_shape: Vec<usize> = Vec::with_capacity(base_layout.ndim() + fancy_ndim);
    out_shape.extend_from_slice(&base_layout.shape()[..consec]);
    out_shape.extend_from_slice(&bulk_shape);
    out_shape.extend_from_slice(&base_layout.shape()[consec..]);
    let layout_c: Layout<IxD> = out_shape.new_contig(None, device.default_order());

    let (_, idx_max) = layout_c.bounds_index()?;
    let mut storage = device.uninit_impl(idx_max)?;
    let aux: Vec<ArrayAuxIndexer<'_>> = entries
        .iter()
        .filter_map(|entry| match entry {
            Entry::Array { src_axis, indices, layout } => {
                Some(ArrayAuxIndexer { src_axis: *src_axis, indices: indices.as_slice(), layout: layout.clone() })
            },
            _ => None,
        })
        .collect();
    device.array_index(storage.raw_mut(), &layout_c, tensor.raw(), &la, &base_layout, &aux, consec)?;
    // SAFETY: `device.array_index` above wrote every element of the fresh
    // storage exactly once (each output position is filled from one gathered
    // source element).
    let storage = unsafe { <B as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
    let out = Tensor::new_f(storage, layout_c)?;
    Ok(out.into_cow())
}

/// Helper for reporting index-array shapes in broadcast errors.
struct DebugShape<'a>(&'a [usize]);

impl Display for DebugShape<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, ")")
    }
}

/// Indexes a tensor by integer arrays (array indexing, *fancy indexing*).
///
/// Array indexing gathers the elements selected by index arrays, in the sense
/// of NumPy's *vectorized indexing*: each index array consumes one axis, the
/// index arrays broadcast against each other (aligning from the last axis),
/// and the gathered elements are the coordinates they describe together.
/// Contrary to basic slicing it is a copying operation; basic indexers may be
/// mixed freely with the index arrays in the same index ([`ArrayIndexer`]).
///
/// - With no index array at all, the index degenerates to basic slicing and the result is a
///   **view**.
/// - With one or more index arrays, the broadcast result of the index arrays forms the *advanced*
///   dimensions, which are placed following NumPy's rule: at the position of the advanced indexers
///   when those are consecutive, at the front otherwise. A plain integer index counts as an
///   "advanced" indexer for this grouping.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Overloads Table
///
/// ## Output `TensorCow<'a, T, B, IxD>`
///
/// - `array_index(tensor, indexer: impl Into<ArrayIndexer<B>>) -> TensorCow<'a, T, B, IxD>` — a
///   single per-axis indexer, applied to the first axis. A host list or vector is *one* index array
///   here, not a tuple of integer indexers.
/// - `array_index(tensor, indexers: (F1, .., F6)) -> TensorCow<'a, T, B, IxD>` — a tuple of up to
///   six per-axis indexers, one per indexed axis (the preferred form).
/// - `array_index(tensor, indexers: Vec<ArrayIndexer<B>>) -> TensorCow<'a, T, B, IxD>` — an
///   explicit list of indexers.
/// - `array_index(tensor, indexers: AxesIndex<ArrayIndexer<B>>) -> TensorCow<'a, T, B, IxD>` — the
///   `Val` / `Vec` forms; [`AxesIndex::None`] is rejected.
///
/// Also, per-axis indexer overloads (the elements of the forms above):
///
/// - integer (`isize` / `usize` / `i32` / `i64` / `u32` / `u64`): one axis dropped (`Select`);
/// - range (`1..4`) or [`slice!`] result: one axis narrowed (`Slice`);
/// - `None` / [`NewAxis`]: one new size-1 axis (`Insert`); [`Ellipsis`]: the ellipsis;
/// - host list (`Vec` / `&Vec` / `&[T]` / `[T; N]` / `&[T; N]`): a one-dimensional index array;
/// - integer tensor or tensor view of any rank: an index array.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`][TensorAny]: the tensor to index.
///
/// - `indexer`: Into [`ArrayIndexArgs<B>`]: the indexers, one per indexed axis.
///
///   - Overloads: a single indexer, a tuple of indexers, a host list/vector, a vector of
///     [`ArrayIndexer`], or an [`AxesIndex<ArrayIndexer<B>>`][AxesIndex].
///
/// # Returns
///
/// - [`TensorCow<'a, T, B, IxD>`][TensorCow]: a borrowed **view** when the index contains no index
///   array (basic slicing), an **owned** gathered tensor otherwise.
///
/// # Examples
///
/// A host list indexes the first axis by a one-dimensional index array:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// let result = rt::array_index(&a, [2, 0]);
/// println!("{result}");
/// // [[ 8 9 10 11]
/// //  [ 0 1 2 3]]
/// # assert_eq!(format!("{result}"), "[[ 8 9 10 11]\n [ 0 1 2 3]]");
/// ```
///
/// A tuple of index arrays zips them together (one index array per axis):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((12, &device)).into_shape([3, 4]);
/// let idx = rt::asarray((vec![0_isize, 2], &device));
/// let result = rt::array_index(&a, (&idx, [1, 3]));
/// println!("{result}");
/// // [ 1 11]
/// # assert_eq!(format!("{result}"), "[ 1 11]");
/// ```
///
/// # Elaborated examples
///
/// ## Mixing basic indexers with index arrays
///
/// Basic indexers may appear between the index arrays. Here a slice is taken
/// first, then two index arrays select along the remaining axes (they are
/// consecutive, so the broadcast dimension is inserted in place):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let b = rt::arange((36, &device)).into_shape([4, 3, 3]);
/// let result = rt::array_index(&b, (1..3, [0, 1, 2], [0, 2, 1]));
/// println!("{result}");
/// // [[ 9 14 16]
/// //  [18 23 25]]
/// # assert_eq!(format!("{result}"), "[[ 9 14 16]\n [ 18 23 25]]");
/// ```
///
/// ## Placement of the broadcast dimensions
///
/// When the advanced indexers are separated by a basic indexer, the broadcast
/// dimensions move to the front instead of staying in place:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::arange((2 * 3 * 4, &device)).into_shape([2, 3, 4]);
/// // the two index arrays are consecutive: the broadcast dimension stays at
/// // position 1
/// println!("{:?}", rt::array_index(&x, (.., [1, 0], 1)).shape());
/// // [2, 2]
/// // separated by a slice: it moves to the front
/// println!("{:?}", rt::array_index(&x, ([1, 0], .., 1)).shape());
/// // [2, 3]
/// # assert_eq!(rt::array_index(&x, (.., [1, 0], 1)).shape(), &[2, 2]);
/// # assert_eq!(rt::array_index(&x, ([1, 0], .., 1)).shape(), &[2, 3]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `x[k1, .., kN]` ([`indexing`](https://data-apis.org/array-api/2024.12/API_specification/indexing.html)):
///   the standard defines the *reduced* integer-array form (every indexer an integer or an integer
///   array, broadcast together, zipped). RSTSR accepts that form as a special case; mixing slices
///   with index arrays is left implementation-defined by the standard.
/// - NumPy: `x[k1, .., kN]` (`numpy.ndarray.__getitem__`): RSTSR implements NumPy's vectorized
///   indexing, including the placement rule for the broadcast dimensions, but not grouped
///   ("parenthesized") index tuples, and not boolean index arrays (use [`mask_select`] /
///   [`bool_select`], or a lone boolean mask through `x[mask]`).
/// - RSTSR: `rt::array_index(&tensor, indexers)`.
///
/// # Panics
///
/// - Panics if an axis index is out of range, if an index array entry (after resolving negative
///   values) is out of range on its axis, if the index arrays cannot be broadcast together, if the
///   index consumes more axes than the tensor has, if the index tensors live on a different device,
///   or if a boolean index array is used.
///
/// For a fallible version, use [`array_index_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: `x[k1, .., kN]` ([`numpy.ndarray.__getitem__`](https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing))
///   — `array_index` is the function form of NumPy's vectorized (advanced) indexing.
/// - Python Array API standard: [`indexing`](https://data-apis.org/array-api/2024.12/API_specification/indexing.html)
///   (`x[k1, .., kN]` with integer arrays).
///
/// ## Related functions in RSTSR
///
/// - [`slice`](crate::tensor::indexing::slice()): basic indexing, always a view.
/// - [`take`] / [`index_select`]: gather along one axis by a host integer list.
/// - [`take_along_axis`]: gather along one axis by an index tensor of the same rank.
/// - [`mask_select`]: gather by a boolean mask.
///
/// ## Variants of this function
///
/// - [`array_index_f`]: fallible version.
/// - Associated methods on [`TensorAny`]: [`TensorAny::array_index`] /
///   [`TensorAny::array_index_f`].
#[allow(clippy::type_complexity)]
pub fn array_index<'a, R, T, B, D, I>(tensor: &'a TensorAny<R, T, B, D>, indexer: I) -> TensorCow<'a, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<isize, Raw = Vec<isize>>
        + DeviceAPI<bool, Raw = Vec<bool>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceArrayIndexAPI<T>,
    I: TryInto<ArrayIndexArgs<B>, Error: Into<Error>>,
{
    array_index_f(tensor, indexer).rstsr_unwrap()
}

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceAPI<isize, Raw = Vec<isize>>
        + DeviceAPI<bool, Raw = Vec<bool>>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + DeviceArrayIndexAPI<T>,
{
    /// Indexes a tensor by integer arrays (array indexing, *fancy indexing*).
    ///
    /// See also [`array_index`].
    pub fn array_index_f<I>(&'a self, indexer: I) -> Result<TensorCow<'a, T, B, IxD>>
    where
        I: TryInto<ArrayIndexArgs<B>, Error: Into<Error>>,
    {
        array_index_f(self, indexer)
    }

    /// Indexes a tensor by integer arrays (array indexing, *fancy indexing*).
    ///
    /// See also [`array_index`].
    pub fn array_index<I>(&'a self, indexer: I) -> TensorCow<'a, T, B, IxD>
    where
        I: TryInto<ArrayIndexArgs<B>, Error: Into<Error>>,
    {
        array_index(self, indexer)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_array_index_workable() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((12, &device)).into_shape([3, 4]);

        // one host list: one index array on axis 0 (not a tuple of selects)
        let b = a.array_index([2, 0]);
        assert_eq!(b.shape(), &vec![2, 4]);
        assert_eq!(b.into_shape([-1]).to_vec(), vec![8, 9, 10, 11, 0, 1, 2, 3]);

        // tuple of two host lists: two index arrays, mutually broadcast
        let c = a.array_index(([0, 2], [1, 3]));
        assert_eq!(c.shape(), &vec![2]);
        assert_eq!(c.into_shape([-1]).to_vec(), vec![1, 11]);

        // no array indexer at all: basic slicing, hence a view
        let d = a.array_index((1..3, 0));
        assert_eq!(d.shape(), &vec![2]);
        assert!(!d.is_owned());
    }

    #[test]
    fn test_array_index_placement() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((36, &device)).into_shape([4, 3, 3]);

        // slice first, then two consecutive index arrays: the broadcast
        // dimensions are inserted in place (position 1)
        let b = a.array_index((1..3, [0, 1, 2], [0, 2, 1]));
        assert_eq!(b.shape(), &vec![2, 3]);
        assert_eq!(b.into_shape([-1]).to_vec(), vec![9, 14, 16, 18, 23, 25]);
    }

    #[test]
    fn test_array_index_placement_separated() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((24, &device)).into_shape([2, 3, 4]);

        // an array separated from an integer by a slice: the broadcast
        // dimensions move to the front
        let b = a.array_index(([1, 0], .., 1));
        assert_eq!(b.shape(), &vec![2, 3]);
        assert_eq!(b.into_shape([-1]).to_vec(), vec![13, 17, 21, 1, 5, 9]);

        // the same indexers without the separating slice: in place
        let c = a.array_index((.., [1, 0], 1));
        assert_eq!(c.shape(), &vec![2, 2]);
        assert_eq!(c.into_shape([-1]).to_vec(), vec![5, 1, 17, 13]);
    }

    #[test]
    fn test_array_index_broadcast() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((12, &device)).into_shape([3, 4]);
        let idx0 = asarray((vec![0_isize, 2], &device)).into_shape([2, 1]);
        let idx1 = asarray((vec![1_isize, 3, 0, 2], &device)).into_shape([1, 4]);

        let b = a.array_index((&idx0, &idx1));
        assert_eq!(b.shape(), &vec![2, 4]);
        assert_eq!(b.into_shape([-1]).to_vec(), vec![1, 3, 0, 2, 9, 11, 8, 10]);
    }

    #[test]
    fn test_array_index_scalar_and_empty() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((12, &device)).into_shape([3, 4]);

        // a zero-dimensional integer array is a scalar index (a view)
        let idx0d_owner = asarray((vec![2_isize], &device));
        let idx0d = idx0d_owner.i(0);
        let b = a.array_index((&idx0d,));
        assert_eq!(b.shape(), &vec![4]);
        assert!(!b.is_owned());
        assert_eq!(b.into_shape([-1]).to_vec(), vec![8, 9, 10, 11]);

        // an empty index array gives an empty result
        let empty = asarray((Vec::<isize>::new(), &device));
        let c = a.array_index((&empty,));
        assert_eq!(c.shape(), &vec![0, 4]);
    }

    #[test]
    fn test_array_index_errors() {
        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);
        let a = arange((12, &device)).into_shape([3, 4]);

        // out of range
        assert!(a.array_index_f([3]).is_err());
        assert!(a.array_index_f([-4]).is_err());
        // negatives count from the back
        let b = a.array_index([-1]);
        assert_eq!(b.into_shape([-1]).to_vec(), vec![8, 9, 10, 11]);
        // too many indexers
        assert!(a.array_index_f((1, 2, 3)).is_err());
        // index arrays that cannot broadcast
        assert!(a.array_index_f(([0, 1], [0, 1, 2])).is_err());
    }
}
