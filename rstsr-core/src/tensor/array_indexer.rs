//! Indexer and arguments of array indexing (fancy indexing).
//!
//! [`ArrayIndexer`] is the per-axis indexer of array indexing; it extends the
//! basic [`Indexer`] with integer-array indexers (the "array indexing" of
//! rstsr, i.e. fancy indexing) and boolean-array indexers, plus host-slice
//! carriers (`OneDim*`) that exist to widen the conversion surface and are
//! lowered to the array variants while parsing.
//!
//! [`ArrayIndexArgs`] is the argument group of
//! [`array_index`](crate::tensor::array_indexing::array_index): a list of
//! indexers, one per indexed axis, supplied as a single indexer, a tuple, a
//! host slice/vector (a one-dimensional index array), or an
//! [`AxesIndex<ArrayIndexer>`][AxesIndex].

use crate::prelude_dev::*;

/// Indexer of array indexing.
///
/// The indexers of [`array_index`](crate::tensor::array_indexing::array_index)
/// mirror the per-axis indexers of NumPy's `x[...]`:
///
/// - [`ArrayIndexer::Basic`]: basic indexer, see [`Indexer`].
/// - [`ArrayIndexer::ArrayIndex`]: an integer tensor indexing one axis.
/// - [`ArrayIndexer::ArrayBool`]: a boolean tensor indexing axes.
/// - [`ArrayIndexer::OneDimIndex`] / [`ArrayIndexer::OneDimBool`]: host slices
///   carried through the `From` conversions; lowering turns them into the
///   tensor variants.
pub enum ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    /// Basic indexer (slice / integer select / new axis / ellipsis).
    Basic(Indexer),
    /// Integer-array indexer: the tensor indexes one axis by its entries.
    ArrayIndex(Tensor<isize, B, IxD>),
    /// Boolean-array indexer: the tensor indexes axes by its true entries.
    ///
    /// Not implemented yet; use [`mask_select`][crate::tensor::adv_indexing::mask_select]
    /// or [`bool_select`][crate::tensor::adv_indexing::bool_select].
    ArrayBool(Tensor<bool, B, IxD>),
    /// One-dimensional host integer list, lowered to
    /// [`ArrayIndexer::ArrayIndex`].
    OneDimIndex(Vec<isize>),
    /// One-dimensional host boolean list, lowered to
    /// [`ArrayIndexer::ArrayBool`].
    OneDimBool(Vec<bool>),
}

impl<B> ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    /// Returns `true` if this indexer is a basic indexer.
    pub fn is_basic(&self) -> bool {
        matches!(self, Self::Basic(_))
    }
}

/// Arguments of array indexing: the per-axis indexers.
///
/// One entry per indexed axis, exactly like NumPy's `x[i, j, k]`; axes not
/// mentioned are taken in full. The argument accepts:
///
/// - a single indexer ([`ArrayIndexer`], an integer, a range/slice, a host
///   integer/boolean list, or an integer/boolean tensor) — the one-entry form;
/// - a tuple of indexers (arity up to 6) — the multi-axis form, e.g.
///   `(1..3, [0, 1, 2], [0, 2, 1])`;
/// - host lists, as in the single-entry form, but written directly
///   (`array_index(&a, [2, 0])`);
/// - an [`AxesIndex<ArrayIndexer<B>>`][AxesIndex] (`AxesIndex::None` is
///   rejected: array indexing requires an explicit index).
///
/// Note that a list *is* one indexer (`[2, 0]` indexes one axis by two
/// entries), while a tuple is a group of indexers (`(2, 0)` selects two axes).
pub struct ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    /// The indexers, one per indexed axis.
    pub indexers: Vec<ArrayIndexer<B>>,
}

impl<B> ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    /// Arguments from an explicit list of indexers.
    pub fn new(indexers: Vec<ArrayIndexer<B>>) -> Self {
        Self { indexers }
    }
}

/* #region indexer conversions */

macro_rules! impl_from_slice_to_array_indexer {
    ($($t:ty),*) => {
        $(
            impl<B> From<$t> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: $t) -> Self {
                    Self::Basic(Indexer::Slice(value.into()))
                }
            }

            impl<B> From<$t> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: $t) -> Self {
                    Self::new(vec![value.into()])
                }
            }
        )*
    };
}

impl_from_slice_to_array_indexer!(
    SliceI,
    core::ops::Range<isize>,
    core::ops::RangeFrom<isize>,
    core::ops::RangeTo<isize>,
    core::ops::Range<usize>,
    core::ops::RangeFrom<usize>,
    core::ops::RangeTo<usize>,
    core::ops::Range<i32>,
    core::ops::RangeFrom<i32>,
    core::ops::RangeTo<i32>,
    core::ops::RangeFull
);

impl<B> From<Indexer> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Indexer) -> Self {
        Self::Basic(value)
    }
}

impl<B> From<Indexer> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Indexer) -> Self {
        Self::new(vec![value.into()])
    }
}

impl<B> From<Option<usize>> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Option<usize>) -> Self {
        Self::Basic(value.into())
    }
}

impl<B> From<Option<usize>> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Option<usize>) -> Self {
        Self::new(vec![value.into()])
    }
}

macro_rules! impl_from_int_to_array_indexer {
    ($($t:ty),*) => {
        $(
            impl<B> From<$t> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: $t) -> Self {
                    Self::Basic(Indexer::Select(value as isize))
                }
            }

            impl<B> From<$t> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: $t) -> Self {
                    Self::new(vec![value.into()])
                }
            }
        )*
    };
}

impl_from_int_to_array_indexer!(usize, isize, u32, i32, u64, i64);

/* #endregion */

/* #region host one-dimensional list conversions */

macro_rules! impl_from_one_dim_to_array_indexer {
    ($($t:ty),*) => {
        $(
            impl<B> From<Vec<$t>> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: Vec<$t>) -> Self {
                    Self::OneDimIndex(value.into_iter().map(|v| v as isize).collect())
                }
            }

            impl<B> From<&Vec<$t>> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: &Vec<$t>) -> Self {
                    value.as_slice().into()
                }
            }

            impl<B> From<&[$t]> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: &[$t]) -> Self {
                    Self::OneDimIndex(value.iter().map(|&v| v as isize).collect())
                }
            }

            impl<B, const N: usize> From<[$t; N]> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: [$t; N]) -> Self {
                    Self::OneDimIndex(value.into_iter().map(|v| v as isize).collect())
                }
            }

            impl<B, const N: usize> From<&[$t; N]> for ArrayIndexer<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: &[$t; N]) -> Self {
                    value.as_slice().into()
                }
            }

            impl<B> From<Vec<$t>> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: Vec<$t>) -> Self {
                    Self::new(vec![value.into()])
                }
            }

            impl<B> From<&Vec<$t>> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: &Vec<$t>) -> Self {
                    Self::new(vec![value.into()])
                }
            }

            impl<B> From<&[$t]> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: &[$t]) -> Self {
                    Self::new(vec![value.into()])
                }
            }

            impl<B, const N: usize> From<[$t; N]> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: [$t; N]) -> Self {
                    Self::new(vec![value.into()])
                }
            }

            impl<B, const N: usize> From<&[$t; N]> for ArrayIndexArgs<B>
            where
                B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            {
                fn from(value: &[$t; N]) -> Self {
                    Self::new(vec![value.into()])
                }
            }
        )*
    };
}

impl_from_one_dim_to_array_indexer!(isize, usize, u32, i32, u64, i64);

impl<B> From<Vec<bool>> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Vec<bool>) -> Self {
        Self::OneDimBool(value)
    }
}

impl<B> From<Vec<bool>> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Vec<bool>) -> Self {
        Self::new(vec![value.into()])
    }
}

impl<B> From<&Vec<bool>> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: &Vec<bool>) -> Self {
        Self::OneDimBool(value.clone())
    }
}

impl<B> From<&Vec<bool>> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: &Vec<bool>) -> Self {
        Self::new(vec![value.into()])
    }
}

impl<B> From<&[bool]> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: &[bool]) -> Self {
        Self::OneDimBool(value.to_vec())
    }
}

impl<B> From<&[bool]> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: &[bool]) -> Self {
        Self::new(vec![value.into()])
    }
}

impl<B, const N: usize> From<[bool; N]> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: [bool; N]) -> Self {
        Self::OneDimBool(value.to_vec())
    }
}

impl<B, const N: usize> From<[bool; N]> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: [bool; N]) -> Self {
        Self::new(vec![value.into()])
    }
}

impl<B, const N: usize> From<&[bool; N]> for ArrayIndexer<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: &[bool; N]) -> Self {
        Self::OneDimBool(value.to_vec())
    }
}

impl<B, const N: usize> From<&[bool; N]> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: &[bool; N]) -> Self {
        Self::new(vec![value.into()])
    }
}

/* #endregion */

/* #region tensor conversions */

/// An integer tensor used as an array indexer.
///
/// A zero-dimensional tensor is *not* an array indexer: it is equivalent to
/// [`Indexer::Select`] of its only entry, and is lowered accordingly while
/// parsing (see [`array_index`](crate::tensor::array_indexing::array_index)).
impl<R, B> From<TensorAny<R, isize, B, IxD>> for ArrayIndexer<B>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<isize>>::Raw>,
    R::Data: Clone,
    B: DeviceAPI<isize>
        + DeviceRawAPI<bool>
        + DeviceRawAPI<MaybeUninit<isize>>
        + DeviceCreationAnyAPI<isize>
        + OpAssignAPI<isize, IxD>,
    <B as DeviceRawAPI<isize>>::Raw: Clone,
{
    fn from(value: TensorAny<R, isize, B, IxD>) -> Self {
        Self::ArrayIndex(value.into_owned())
    }
}

impl<R, B> From<&TensorAny<R, isize, B, IxD>> for ArrayIndexer<B>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<isize>>::Raw>,
    R::Data: Clone,
    B: DeviceAPI<isize>
        + DeviceRawAPI<bool>
        + DeviceRawAPI<MaybeUninit<isize>>
        + DeviceCreationAnyAPI<isize>
        + OpAssignAPI<isize, IxD>,
    <B as DeviceRawAPI<isize>>::Raw: Clone,
{
    fn from(value: &TensorAny<R, isize, B, IxD>) -> Self {
        Self::ArrayIndex(value.to_owned())
    }
}

impl<R, B> From<TensorAny<R, bool, B, IxD>> for ArrayIndexer<B>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<bool>>::Raw>,
    R::Data: Clone,
    B: DeviceAPI<bool>
        + DeviceRawAPI<isize>
        + DeviceRawAPI<MaybeUninit<bool>>
        + DeviceCreationAnyAPI<bool>
        + OpAssignAPI<bool, IxD>,
    <B as DeviceRawAPI<bool>>::Raw: Clone,
{
    fn from(value: TensorAny<R, bool, B, IxD>) -> Self {
        Self::ArrayBool(value.into_owned())
    }
}

impl<R, B> From<&TensorAny<R, bool, B, IxD>> for ArrayIndexer<B>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<bool>>::Raw>,
    R::Data: Clone,
    B: DeviceAPI<bool>
        + DeviceRawAPI<isize>
        + DeviceRawAPI<MaybeUninit<bool>>
        + DeviceCreationAnyAPI<bool>
        + OpAssignAPI<bool, IxD>,
    <B as DeviceRawAPI<bool>>::Raw: Clone,
{
    fn from(value: &TensorAny<R, bool, B, IxD>) -> Self {
        Self::ArrayBool(value.to_owned())
    }
}

impl<R, B> From<TensorAny<R, isize, B, IxD>> for ArrayIndexArgs<B>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<isize>>::Raw>,
    R::Data: Clone,
    B: DeviceAPI<isize>
        + DeviceRawAPI<bool>
        + DeviceRawAPI<MaybeUninit<isize>>
        + DeviceCreationAnyAPI<isize>
        + OpAssignAPI<isize, IxD>,
    <B as DeviceRawAPI<isize>>::Raw: Clone,
{
    fn from(value: TensorAny<R, isize, B, IxD>) -> Self {
        Self::new(vec![value.into()])
    }
}

impl<R, B> From<&TensorAny<R, isize, B, IxD>> for ArrayIndexArgs<B>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<isize>>::Raw>,
    R::Data: Clone,
    B: DeviceAPI<isize>
        + DeviceRawAPI<bool>
        + DeviceRawAPI<MaybeUninit<isize>>
        + DeviceCreationAnyAPI<isize>
        + OpAssignAPI<isize, IxD>,
    <B as DeviceRawAPI<isize>>::Raw: Clone,
{
    fn from(value: &TensorAny<R, isize, B, IxD>) -> Self {
        Self::new(vec![value.into()])
    }
}

/* #endregion */

/* #region arguments conversions */

impl<B> From<ArrayIndexer<B>> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: ArrayIndexer<B>) -> Self {
        Self::new(vec![value])
    }
}

impl<B> From<Vec<ArrayIndexer<B>>> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(value: Vec<ArrayIndexer<B>>) -> Self {
        Self::new(value)
    }
}

impl<B> From<()> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    fn from(_: ()) -> Self {
        Self::new(vec![])
    }
}

impl<B> TryFrom<AxesIndex<ArrayIndexer<B>>> for ArrayIndexArgs<B>
where
    B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
{
    type Error = Error;

    fn try_from(value: AxesIndex<ArrayIndexer<B>>) -> Result<Self> {
        match value {
            AxesIndex::Val(value) => Ok(Self::new(vec![value])),
            AxesIndex::Vec(value) => Ok(Self::new(value)),
            AxesIndex::None => {
                rstsr_raise!(InvalidValue, "array_index requires an explicit index; AxesIndex::None is not accepted.")?
            },
        }
    }
}

/// Tuple of indexers, one per indexed axis (like NumPy's `x[i, j, k]`).
///
/// Note the difference between a tuple (several indexers) and a host slice or
/// vector (a single one-dimensional index array): `(a, b)` indexes two axes by
/// two indexers, while `[a, b]` is a single index array indexing one axis.
macro_rules! impl_from_tuple_to_array_index_args {
    ($($f:ident),+) => {
        impl<B, $($f,)+> From<($($f,)+)> for ArrayIndexArgs<B>
        where
            B: DeviceRawAPI<isize> + DeviceRawAPI<bool>,
            $($f: Into<ArrayIndexer<B>>,)+
        {
            fn from(value: ($($f,)+)) -> Self {
                #[allow(non_snake_case)]
                let ($($f,)+) = value;
                Self::new(vec![$($f.into(),)+])
            }
        }
    };
}

impl_from_tuple_to_array_index_args!(F1);
impl_from_tuple_to_array_index_args!(F1, F2);
impl_from_tuple_to_array_index_args!(F1, F2, F3);
impl_from_tuple_to_array_index_args!(F1, F2, F3, F4);
impl_from_tuple_to_array_index_args!(F1, F2, F3, F4, F5);
impl_from_tuple_to_array_index_args!(F1, F2, F3, F4, F5, F6);

/* #endregion */
