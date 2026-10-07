use crate::prelude_dev::*;

/// Enum for Axes indexing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxesIndex<T> {
    None,
    Val(T),
    Vec(Vec<T>),
}

impl<T> AsRef<[T]> for AxesIndex<T> {
    fn as_ref(&self) -> &[T] {
        match self {
            AxesIndex::Val(v) => core::slice::from_ref(v),
            AxesIndex::Vec(v) => v.as_slice(),
            AxesIndex::None => panic!("AxesIndex::None cannot be converted to a slice. This is developer's error; if encountered, please report it to github issue."),
        }
    }
}

/* #region AxesIndex self-type from */

impl<T> From<T> for AxesIndex<T> {
    fn from(value: T) -> Self {
        AxesIndex::Val(value)
    }
}

impl<T> From<&T> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &T) -> Self {
        AxesIndex::Val(value.clone())
    }
}

impl<T> From<Vec<T>> for AxesIndex<T> {
    fn from(value: Vec<T>) -> Self {
        AxesIndex::Vec(value)
    }
}

impl<T, const N: usize> From<[T; N]> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: [T; N]) -> Self {
        AxesIndex::Vec(value.to_vec())
    }
}

impl<T> From<&Vec<T>> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &Vec<T>) -> Self {
        AxesIndex::Vec(value.clone())
    }
}

impl<T> From<&[T]> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &[T]) -> Self {
        AxesIndex::Vec(value.to_vec())
    }
}

impl<T, const N: usize> From<&[T; N]> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &[T; N]) -> Self {
        AxesIndex::Vec(value.to_vec())
    }
}

#[duplicate_item(T; [usize]; [isize])]
impl From<()> for AxesIndex<T> {
    fn from(_: ()) -> Self {
        AxesIndex::Vec(vec![])
    }
}

#[duplicate_item(T; [usize]; [isize])]
impl TryFrom<Option<T>> for AxesIndex<T> {
    type Error = Error;

    fn try_from(value: Option<T>) -> Result<Self> {
        match value {
            Some(v) => Ok(AxesIndex::Val(v)),
            None => Ok(AxesIndex::None),
        }
    }
}

/* #endregion AxesIndex self-type from */

/* #region AxesIndex other-type from */

macro_rules! impl_try_from_axes_index {
    ($t1:ty, $($t2:ty),*) => {
        $(
            impl TryFrom<$t2> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: $t2) -> Result<Self> {
                    Ok(AxesIndex::Val(value.try_into()?))
                }
            }

            impl TryFrom<&$t2> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &$t2) -> Result<Self> {
                    Ok(AxesIndex::Val((*value).try_into()?))
                }
            }

            impl TryFrom<Vec<$t2>> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: Vec<$t2>) -> Result<Self> {
                    let value = value
                        .into_iter()
                        .map(|v| v.try_into().map_err(|_| rstsr_error!(TryFromIntError)))
                        .collect::<Result<Vec<$t1>>>()?;
                    Ok(AxesIndex::Vec(value))
                }
            }

            impl<const N: usize> TryFrom<[$t2; N]> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: [$t2; N]) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }

            impl TryFrom<&Vec<$t2>> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &Vec<$t2>) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }

            impl TryFrom<&[$t2]> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &[$t2]) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }

            impl<const N: usize> TryFrom<&[$t2; N]> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &[$t2; N]) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }
        )*
    };
}

impl_try_from_axes_index!(usize, isize, u32, u64, i32, i64);
impl_try_from_axes_index!(isize, usize, u32, u64, i32, i64);

/* #endregion AxesIndex other-type from */

/* #region AxesIndex tuple-type from */

// it seems that this directly implementing arbitary AxesIndex<T> will cause
// conflicting implementation so make a macro for this task

#[macro_export]
macro_rules! impl_from_tuple_to_axes_index {
    ($t: ty) => {
        impl<F1> TryFrom<(F1,)> for AxesIndex<$t>
        where
            $t: TryFrom<F1>,
        {
            type Error = Error;

            fn try_from(value: (F1,)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![value.0.try_into().ok().unwrap()]))
            }
        }

        impl<F1, F2> TryFrom<(F1, F2)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![value.0.try_into().ok().unwrap(), value.1.try_into().ok().unwrap()]))
            }
        }

        impl<F1, F2, F3> TryFrom<(F1, F2, F3)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4> TryFrom<(F1, F2, F3, F4)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5> TryFrom<(F1, F2, F3, F4, F5)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6> TryFrom<(F1, F2, F3, F4, F5, F6)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5> + TryFrom<F6>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7> TryFrom<(F1, F2, F3, F4, F5, F6, F7)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5> + TryFrom<F6> + TryFrom<F7>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8> TryFrom<(F1, F2, F3, F4, F5, F6, F7, F8)> for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>
                + TryFrom<F8>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7, F8)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8, F9> TryFrom<(F1, F2, F3, F4, F5, F6, F7, F8, F9)> for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>
                + TryFrom<F8>
                + TryFrom<F9>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7, F8, F9)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                    value.8.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8, F9, F10> TryFrom<(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)>
            for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>
                + TryFrom<F8>
                + TryFrom<F9>
                + TryFrom<F10>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                    value.8.try_into().ok().unwrap(),
                    value.9.try_into().ok().unwrap(),
                ]))
            }
        }
    };
}

impl_from_tuple_to_axes_index!(isize);
impl_from_tuple_to_axes_index!(usize);

/* #endregion AxesIndex tuple-type from */

/* #region utilities for AxesIndex */

/// Normalize axes argument into a tuple of non-negative integer axes.
///
/// Though the returned vector will be of type `isize` for convenience, the values will be actually
/// non-negative (`usize`-compatible).
pub fn normalize_axes_index(
    axes: AxesIndex<isize>,
    ndim: usize,
    allow_duplicate: bool,
    sort: bool,
) -> Result<Vec<isize>> {
    // generate the normalized axes vector
    let vec = match axes {
        AxesIndex::None => rstsr_raise!(InvalidValue, "Axes argument cannot be None for this operation.")?,
        AxesIndex::Val(axis) => {
            let axis = rstsr_check_axis!(axis, ndim)?;
            vec![axis as isize]
        },
        AxesIndex::Vec(axes) => {
            let mut normalized_axes = Vec::with_capacity(axes.len());
            for &axis in axes.iter() {
                let norm_axis = rstsr_check_axis!(axis, ndim)?;
                normalized_axes.push(norm_axis as isize);
            }
            if sort {
                normalized_axes.sort();
            }
            normalized_axes
        },
    };
    if !allow_duplicate {
        let vec_sorted = if sort { vec.clone() } else { vec.iter().copied().sorted().collect() };
        // check for duplicates in sorted vector
        if vec_sorted.windows(2).any(|w| w[0] == w[1]) {
            rstsr_raise!(InvalidValue, "Duplicate axes are not allowed.")?;
        }
    }
    Ok(vec)
}

/* #endregion */
/* #region AxisIndex (single axis) */

/// Wrapper for at most one axis, mirroring [`AxesIndex`] for the single-axis
/// case; makes one-axis signatures distinct from none-or-multi axes.
///
/// [`AxisIndex::None`] carries no explicit axis and selects the last axis (the
/// `axis = -1` convention); [`AxisIndex::Val`] carries the explicit axis, with
/// negative values counting from the back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisIndex<T> {
    /// No explicit axis: the operation selects the last axis.
    None,
    /// Explicit (possibly negative) axis.
    Val(T),
}

impl<T> AxisIndex<T> {
    /// Unwrap the inner axis value; [`AxisIndex::None`] stays `None`.
    pub fn into_option(self) -> Option<T> {
        match self {
            AxisIndex::None => None,
            AxisIndex::Val(value) => Some(value),
        }
    }
}

impl AxisIndex<isize> {
    /// Normalize into a non-negative axis for a tensor of `ndim` dimensions.
    ///
    /// [`AxisIndex::None`] selects the last axis; negative values count from
    /// the back. Errors when the axis is out of range — in particular for a
    /// 0-dimensional tensor, which has no axis at all.
    pub fn into_normalized(self, ndim: usize) -> Result<usize> {
        let axis = match self {
            AxisIndex::None => -1_isize,
            AxisIndex::Val(value) => value,
        };
        rstsr_check_axis!(axis, ndim)
    }
}

impl<T> From<T> for AxisIndex<T> {
    fn from(value: T) -> Self {
        AxisIndex::Val(value)
    }
}

impl<T> From<&T> for AxisIndex<T>
where
    T: Clone,
{
    fn from(value: &T) -> Self {
        AxisIndex::Val(value.clone())
    }
}

#[duplicate_item(T; [usize]; [isize])]
impl From<()> for AxisIndex<T> {
    fn from(_: ()) -> Self {
        AxisIndex::None
    }
}

#[duplicate_item(T; [usize]; [isize])]
impl TryFrom<Option<T>> for AxisIndex<T> {
    type Error = Error;

    fn try_from(value: Option<T>) -> Result<Self> {
        match value {
            Some(v) => Ok(AxisIndex::Val(v)),
            None => Ok(AxisIndex::None),
        }
    }
}

macro_rules! impl_try_from_axis_index {
    ($t1:ty, $($t2:ty),*) => {
        $(
            impl TryFrom<$t2> for AxisIndex<$t1> {
                type Error = Error;

                fn try_from(value: $t2) -> Result<Self> {
                    Ok(AxisIndex::Val(value.try_into()?))
                }
            }

            impl TryFrom<&$t2> for AxisIndex<$t1> {
                type Error = Error;

                fn try_from(value: &$t2) -> Result<Self> {
                    Ok(AxisIndex::Val((*value).try_into()?))
                }
            }

            impl TryFrom<AxisIndex<$t2>> for AxisIndex<$t1> {
                type Error = Error;

                fn try_from(value: AxisIndex<$t2>) -> Result<Self> {
                    match value {
                        AxisIndex::None => Ok(AxisIndex::None),
                        AxisIndex::Val(value) => Ok(AxisIndex::Val(value.try_into()?)),
                    }
                }
            }
        )*
    };
}

impl_try_from_axis_index!(usize, isize, u32, u64, i32, i64);
impl_try_from_axis_index!(isize, usize, u32, u64, i32, i64);

/* #endregion AxisIndex (single axis) */

#[cfg(test)]
mod axis_index_tests {
    use crate::prelude_dev::*;

    #[test]
    fn test_axis_index_from() {
        let a: AxisIndex<isize> = 2.into();
        assert_eq!(a.into_option(), Some(2));
        let a: AxisIndex<isize> = (&-1_isize).into();
        assert_eq!(a.into_option(), Some(-1));
        let a = AxisIndex::from(3_isize);
        let v: AxisIndex<usize> = a.try_into().unwrap();
        assert_eq!(v.into_option(), Some(3_usize));
        // negative axis cannot convert to unsigned: must error
        let a = AxisIndex::from(-1_isize);
        assert!(AxisIndex::<usize>::try_from(a).is_err());
        // raw integer TryFrom
        let a = AxisIndex::<isize>::try_from(4_i32).unwrap();
        assert_eq!(a.into_option(), Some(4));
        let a = AxisIndex::<isize>::try_from(&4_i64).unwrap();
        assert_eq!(a.into_option(), Some(4));
        // None and () both mean "no explicit axis"
        let a: AxisIndex<isize> = Option::<isize>::None.try_into().unwrap();
        assert_eq!(a, AxisIndex::None);
        let a = AxisIndex::<isize>::from(());
        assert_eq!(a, AxisIndex::None);
        // None survives the cross-integer conversion
        let a = AxisIndex::<usize>::try_from(AxisIndex::<isize>::None).unwrap();
        assert_eq!(a, AxisIndex::None);
    }

    #[test]
    fn test_axis_index_normalize() {
        assert_eq!(AxisIndex::Val(1_isize).into_normalized(3).unwrap(), 1);
        assert_eq!(AxisIndex::Val(-1_isize).into_normalized(3).unwrap(), 2);
        assert_eq!(AxisIndex::None.into_normalized(3).unwrap(), 2);
        assert!(AxisIndex::Val(3_isize).into_normalized(3).is_err());
        assert!(AxisIndex::None.into_normalized(0).is_err());
    }
}
