//! Sorting operator traits: [`OpSortAPI`], [`OpArgSortAPI`], and
//! [`OpSortCustomAPI`] (user comparator).

#![allow(clippy::type_complexity)]

use crate::prelude_dev::*;

/// Sort arguments: which axis, direction, and stability requirement.
///
/// Mirrors [`ReduceArgs`]-style conversion ergonomics: `()` gives the defaults
/// (`axis = -1`, ascending, stable), and tuples override individual fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortArgs {
    /// Axis to sort along (negative counts from the back); default `-1`.
    pub axis: isize,
    /// Sort descending; default `false`.
    pub descending: bool,
    /// Require a stable sort (ties keep input order); default `true`.
    pub stable: bool,
}

impl Default for SortArgs {
    fn default() -> Self {
        Self { axis: -1, descending: false, stable: true }
    }
}

impl From<()> for SortArgs {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl From<isize> for SortArgs {
    fn from(axis: isize) -> Self {
        Self { axis, ..Self::default() }
    }
}

impl From<bool> for SortArgs {
    fn from(descending: bool) -> Self {
        Self { descending, ..Self::default() }
    }
}

impl From<(isize, bool)> for SortArgs {
    fn from((axis, descending): (isize, bool)) -> Self {
        Self { axis, descending, ..Self::default() }
    }
}

impl From<(isize, bool, bool)> for SortArgs {
    fn from((axis, descending, stable): (isize, bool, bool)) -> Self {
        Self { axis, descending, stable }
    }
}

/// Built-in sort along one axis (default comparator, see [`ExtSortCmp`]).
pub trait OpSortAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<T>,
{
    /// Sort `a` along `axis` (already normalized, non-negative) into a fresh
    /// storage; output layout must be contiguous of the input's shape.
    fn sort_axes(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: usize,
        descending: bool,
        stable: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<T>>::Raw>, T, Self>, Layout<IxD>)>;
}

/// Argsort along one axis (default comparator, see [`ExtSortCmp`]); output
/// dtype is `usize` row-major flat indices into the sorted axis.
pub trait OpArgSortAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<usize>,
{
    /// Argsort `a` along `axis` (already normalized, non-negative) into a
    /// fresh storage of `usize` indices.
    fn argsort_axes(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: usize,
        descending: bool,
        stable: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)>;
}

/// Sort/argsort with a user-supplied comparator (see [`OpReduceCustomAPI`] for
/// the closure-carrying precedent); `F: Fn(&T, &T) -> core::cmp::Ordering`.
pub trait OpSortCustomAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<usize>,
{
    /// Sort with a custom comparator.
    fn sort_axes_custom<F>(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: usize,
        f: F,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<T>>::Raw>, T, Self>, Layout<IxD>)>
    where
        F: Fn(&T, &T) -> core::cmp::Ordering + Send + Sync;

    /// Argsort with a custom comparator; indices are row-major flat positions
    /// along `axis`.
    fn argsort_axes_custom<F>(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: usize,
        f: F,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)>
    where
        F: Fn(&T, &T) -> core::cmp::Ordering + Send + Sync;
}
