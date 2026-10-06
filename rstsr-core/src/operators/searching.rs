//! Searchsorted operator trait and argument types.

#![allow(clippy::type_complexity)]

use crate::prelude_dev::*;

/// Insertion side for [`searchsorted`](crate::tensor::searching::searchsorted):
/// `'left'` gives the first insertion point (`x1[i-1] < v <= x1[i]`), `'right'`
/// gives the last (`x1[i-1] <= v < x1[i]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSide {
    /// First insertion point.
    Left,
    /// Last insertion point.
    Right,
}

impl From<SearchSide> for bool {
    fn from(value: SearchSide) -> Self {
        matches!(value, SearchSide::Right)
    }
}

/// Searchsorted arguments: side and optional sorter permutation.
///
/// `()` gives the defaults (`side = Left`, no sorter); a [`SearchSide`], a
/// string (`"left"` / `"right"`), or a sorter sequence overrides individual
/// fields (tuple forms combine them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchSortedArgs {
    /// Insertion side; default [`SearchSide::Left`].
    pub side: SearchSide,
    /// Permutation of `x1` that sorts it ascending (NumPy-parity `sorter`).
    pub sorter: Option<Vec<usize>>,
}

impl Default for SearchSortedArgs {
    fn default() -> Self {
        Self { side: SearchSide::Left, sorter: None }
    }
}

impl From<()> for SearchSortedArgs {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl From<SearchSide> for SearchSortedArgs {
    fn from(side: SearchSide) -> Self {
        Self { side, ..Self::default() }
    }
}

impl TryFrom<&str> for SearchSortedArgs {
    type Error = Error;

    fn try_from(side: &str) -> Result<Self> {
        Ok(Self { side: side.try_into()?, ..Self::default() })
    }
}

impl TryFrom<String> for SearchSortedArgs {
    type Error = Error;

    fn try_from(side: String) -> Result<Self> {
        Self::try_from(side.as_str())
    }
}

impl TryFrom<&String> for SearchSortedArgs {
    type Error = Error;

    fn try_from(side: &String) -> Result<Self> {
        Self::try_from(side.as_str())
    }
}

impl<T: Into<usize>> TryFrom<Vec<T>> for SearchSortedArgs {
    type Error = Error;

    fn try_from(sorter: Vec<T>) -> Result<Self> {
        Ok(Self { sorter: Some(sorter.into_iter().map(|v| v.into()).collect()), ..Self::default() })
    }
}

impl<T: Into<usize> + Clone> TryFrom<&Vec<T>> for SearchSortedArgs {
    type Error = Error;

    fn try_from(sorter: &Vec<T>) -> Result<Self> {
        Vec::clone(sorter).try_into()
    }
}

impl TryFrom<(SearchSide, Vec<usize>)> for SearchSortedArgs {
    type Error = Error;

    fn try_from((side, sorter): (SearchSide, Vec<usize>)) -> Result<Self> {
        Ok(Self { side, sorter: Some(sorter) })
    }
}

impl TryFrom<(&str, Vec<usize>)> for SearchSortedArgs {
    type Error = Error;

    fn try_from((side, sorter): (&str, Vec<usize>)) -> Result<Self> {
        Ok(Self { side: side.try_into()?, sorter: Some(sorter) })
    }
}

impl TryFrom<&str> for SearchSide {
    type Error = Error;

    fn try_from(side: &str) -> Result<Self> {
        match side {
            "left" => Ok(SearchSide::Left),
            "right" => Ok(SearchSide::Right),
            other => rstsr_raise!(InvalidValue, "searchsorted side must be 'left' or 'right', got {:?}.", other),
        }
    }
}

impl From<bool> for SearchSide {
    /// Rust-side ergonomics: `false` = 'left', `true` = 'right'
    /// (mirrors the C-level convention of some libraries; NumPy itself does
    /// not accept booleans for `side`).
    fn from(value: bool) -> Self {
        if value {
            SearchSide::Right
        } else {
            SearchSide::Left
        }
    }
}

/// Binary search of sorted 1-D `x1` for each value of `x2` (any shape).
///
/// Output is `usize` positions with the same shape as `x2`. With `sorter`,
/// `x1` is searched through the permutation (i.e. `x1_permuted[j] =
/// x1[sorter[j]]` is the sorted sequence); positions are returned in terms of
/// the permuted sequence, matching NumPy.
pub trait OpSearchSortedAPI<T, V, D2>
where
    D2: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<V> + DeviceAPI<usize>,
{
    fn searchsorted(
        &self,
        x1: &<Self as DeviceRawAPI<T>>::Raw,
        l1: &Layout<IxD>,
        x2: &<Self as DeviceRawAPI<V>>::Raw,
        l2: &Layout<D2>,
        side_right: bool,
        sorter: Option<&[usize]>,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)>;
}

/// Nonzero: two-pass count + flat-index fill over the row-major visit
/// order. The tensor level splits the flat indices into per-dimension
/// coordinate tensors (host layout math).
pub trait OpNonzeroAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<usize> + DeviceRawAPI<MaybeUninit<usize>>,
{
    /// Count the nonzero elements (`is_nonzero` decides; bool: true,
    /// complex: either component nonzero).
    fn nonzero_count(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        is_nonzero: &dyn Fn(&T) -> bool,
    ) -> Result<usize>;

    /// Fill the flat C-order indices of every nonzero element into `out`
    /// (capacity = the count from [`Self::nonzero_count`]); returns the same
    /// count.
    fn nonzero_fill(
        &self,
        out: &mut <Self as DeviceRawAPI<MaybeUninit<usize>>>::Raw,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        is_nonzero: &dyn Fn(&T) -> bool,
    ) -> Result<usize>;
}
