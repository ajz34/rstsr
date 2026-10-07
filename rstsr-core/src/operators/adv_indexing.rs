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
