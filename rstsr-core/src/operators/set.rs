//! Set-operation operator traits: unique family and isin.

#![allow(clippy::type_complexity)]

use crate::prelude_dev::*;

/// Unique-family operations over the flattened row-major sequence.
///
/// Output storages are caller-allocated with the INPUT's element count `n`
/// (the unique count is data-dependent and returned); the tensor level
/// truncates to `u` elements. `unique_counts`/`unique_inverse` derive from
/// `unique_all` at the tensor level, so only the two kernels below exist.
pub trait OpUniqueAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceRawAPI<MaybeUninit<usize>>,
{
    /// Unique values only. Writes up to `n` values into `values`; returns
    /// the unique count.
    fn unique_values(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        values: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
    ) -> Result<usize>;

    /// Unique values + first-occurrence flat C-order indices (length `u`),
    /// inverse indices (length `n`, one unique-entry slot per input element,
    /// row-major), and counts (length `u`). Returns the unique count.
    #[allow(clippy::type_complexity)]
    fn unique_all(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        values: &mut <Self as DeviceRawAPI<MaybeUninit<T>>>::Raw,
        indices: &mut <Self as DeviceRawAPI<MaybeUninit<usize>>>::Raw,
        inverse: &mut <Self as DeviceRawAPI<MaybeUninit<usize>>>::Raw,
        counts: &mut <Self as DeviceRawAPI<MaybeUninit<usize>>>::Raw,
    ) -> Result<usize>;
}

/// Isin: element membership of `x1` (any shape) in `x2` (any shape); output
/// is `bool` with `x1`'s shape. The device handles sorting/dedup of `x2`
/// internally (efficiency exception: an O(m) sorted copy of x2's values).
pub trait OpIsinAPI<T, D1>
where
    D1: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<bool>,
{
    fn isin(
        &self,
        x1: &<Self as DeviceRawAPI<T>>::Raw,
        l1: &Layout<D1>,
        x2: &<Self as DeviceRawAPI<T>>::Raw,
        l2: &Layout<IxD>,
        invert: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<bool>>::Raw>, bool, Self>, Layout<IxD>)>;
}
