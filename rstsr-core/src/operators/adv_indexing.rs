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
/// sees non-negative in-bounds `usize` entries.
pub struct ArrayAuxIndexer<'a> {
    /// Source axis consumed by this index array.
    pub src_axis: usize,
    /// Resolved index values, in C (row-major) order over `layout`.
    pub indices: &'a [usize],
    /// Contiguous layout of the index array's own shape.
    pub layout: Layout<IxD>,
}

/// Array indexing (fancy indexing) by integer arrays on one or more axes.
///
/// See [`array_index`](crate::tensor::array_indexing::array_index). The result
/// is fully described by `lc` (the output layout, with the broadcast index
/// dimensions placed at `consec`) and `base_layout` (the layout of the
/// non-array-indexed subspace, whose strides are those of `la` and whose offset
/// already carries the integer selections).
pub trait DeviceArrayIndexAPI<T>
where
    Self: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>>,
{
    /// Gather `a` into `c` by integer arrays on selected axes.
    #[allow(clippy::too_many_arguments)]
    fn array_index(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        lc: &Layout<IxD>,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<IxD>,
        base_layout: &Layout<IxD>,
        indexers: &[ArrayAuxIndexer<'_>],
        consec: usize,
    ) -> Result<()>;
}
