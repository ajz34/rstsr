//! Advanced indexing related device traits.
//!
//! Currently, full support of advanced indexing is not available. However, it
//! is still possible to index one axis by list.

use crate::prelude_dev::*;

pub trait DeviceIndexSelectAPI<T, D>
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Self: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>>,
{
    /// Index select on one axis.
    fn index_select(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: usize,
        indices: &[usize],
    ) -> Result<()>;
}

/// Gather along one axis with an index TENSOR (see
/// [`take_along_axis`]).
///
/// `idx` must have the same rank as `a`; every non-axis dimension must match
/// `a`'s shape; entries must be within `0..a.shape()[axis]` (all validated
/// at the tensor level). `layout_c` is a contiguous layout of the output
/// shape (input shape with the axis length replaced by `idx`'s).
pub trait DeviceTakeAlongAxisAPI<T, DA, DI>
where
    DA: DimAPI,
    DI: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<usize> + DeviceRawAPI<MaybeUninit<T>>,
{
    #[allow(clippy::too_many_arguments)]
    fn take_along_axis(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        layout_c: &Layout<IxD>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<DA>,
        idx: &<Self as DeviceRawAPI<usize>>::Raw,
        lidx: &Layout<DI>,
        axis: usize,
    ) -> Result<()>;
}

/// Whole-tensor boolean-mask gather / scatter (see [`mask_select`] /
/// [`mask_fill`]).
///
/// The mask has `dm <= da` axes matching `a`'s leading axes; a `true` selects
/// the trailing block `a[i_0, .., i_{dm-1}, ..]` of size
/// `B = prod(a.shape()[dm..])`. Mask entries are visited in the device default
/// order.
pub trait DeviceMaskIndexAPI<TA, DA, DM>
where
    DA: DimAPI,
    DM: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<bool> + DeviceRawAPI<MaybeUninit<TA>>,
{
    /// Gather the blocks of `a` selected by `mask` into `c`.
    ///
    /// `c` has capacity `count * B` (`count` = number of `true` entries in
    /// `mask`), written front-to-back in the mask visit order.
    fn mask_select(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TA>>>::Raw,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        mask: &<Self as DeviceRawAPI<bool>>::Raw,
        lm: &Layout<DM>,
    ) -> Result<()>;

    /// Write `value` into every element of `a` selected by `mask`.
    fn mask_fill(
        &self,
        a: &mut <Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        mask: &<Self as DeviceRawAPI<bool>>::Raw,
        lm: &Layout<DM>,
        value: TA,
    ) -> Result<()>;
}

/// One array indexer at the device boundary (see
/// [`DeviceArrayIndexAPI::array_index`]).
///
/// The tensor level resolves and validates everything, so a device only ever
/// sees non-negative in-bounds `usize` entries, and those entries live in
/// **device storage**. `(indices, layout)` represents the resolved index array,
/// but the representation is deliberately not canonical: an index array is not
/// required to be row-major (or contiguous, or in any particular order), so a
/// device reads the entries through `layout` — arbitrary strides and offset
/// included — instead of assuming a host slice or a C-contiguous buffer. Any
/// layout that represents the same index array yields the same result.
pub struct ArrayAuxIndexer<'a, B>
where
    B: DeviceRawAPI<usize>,
{
    /// Source axis consumed by this index array.
    pub src_axis: usize,
    /// Resolved index entries, in device storage.
    pub indices: &'a <B as DeviceRawAPI<usize>>::Raw,
    /// Layout of the index array's own shape.
    pub layout: Layout<IxD>,
}

/// Array indexing (fancy indexing) by integer arrays on one or more axes.
///
/// See [`array_index`]. The result
/// is fully described by `lc` (the output layout, with the broadcast index
/// dimensions placed at `consec`) and `base_layout` (the layout of the
/// non-array-indexed subspace, whose strides are those of `la` and whose offset
/// already carries the integer selections).
pub trait DeviceArrayIndexAPI<T>
where
    Self: DeviceAPI<T> + DeviceRawAPI<usize> + DeviceRawAPI<MaybeUninit<T>>,
{
    /// Gather `a` into `c` by integer arrays on selected axes.
    ///
    /// `order` is the device default order: the broadcast index dimensions are
    /// visited in that order. The gathered values do not depend on it (the
    /// output arrangement is carried by `lc`); the traversal, and with it the
    /// locality of the writes, follows the device.
    #[allow(clippy::too_many_arguments)]
    fn array_index(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        lc: &Layout<IxD>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<IxD>,
        base_layout: &Layout<IxD>,
        indexers: &[ArrayAuxIndexer<'_, Self>],
        consec: usize,
        order: FlagOrder,
    ) -> Result<()>;
}

/// Array-indexing assignment (scatter): write a broadcastable value into the
/// positions the index arrays select.
///
/// See [`array_index_assign`]. This is the inverse of
/// [`DeviceArrayIndexAPI::array_index`]: the destination `a` is addressed
/// through `la` / `base_layout` / `indexers` exactly as the gather addresses
/// its source, while the written values come from `value` read through
/// `lvalue`, a layout already broadcast to the gather output shape (its
/// `shape()[consec..consec + fancy_ndim]` is the broadcast block, as `lc` is
/// for the gather). The value dtype `TA` is cast to the destination dtype `TC`.
///
/// Duplicate index targets write the same destination position more than once;
/// the visit order — the device default order — decides the winner (the last
/// write wins), so this op is not parallelized.
pub trait DeviceArrayIndexAssignAPI<TC, TA = TC>
where
    Self: DeviceAPI<TC> + DeviceAPI<TA> + DeviceRawAPI<usize>,
{
    #[allow(clippy::too_many_arguments)]
    fn array_index_assign(
        &self,
        a: &mut <Self as DeviceRawAPI<TC>>::Raw,
        la: &Layout<IxD>,
        base_layout: &Layout<IxD>,
        indexers: &[ArrayAuxIndexer<'_, Self>],
        value: &<Self as DeviceRawAPI<TA>>::Raw,
        lvalue: &Layout<IxD>,
        consec: usize,
        order: FlagOrder,
    ) -> Result<()>;
}
