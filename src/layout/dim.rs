use super::*;
use core::fmt::Debug;

/* #region basic definitions */

pub type Ix<const N: usize> = [usize; N];
pub type Ix0 = Ix<0>;
pub type Ix1 = Ix<1>;
pub type Ix2 = Ix<2>;
pub type Ix3 = Ix<3>;
pub type Ix4 = Ix<4>;
pub type Ix5 = Ix<5>;
pub type Ix6 = Ix<6>;
pub type Ix7 = Ix<7>;
pub type Ix8 = Ix<8>;
pub type Ix9 = Ix<9>;
pub type IxD = Vec<usize>;
pub type IxDyn = IxD;

pub trait DimBaseAPI: AsMut<[usize]> + AsRef<[usize]> + Debug + Clone {
    type Shape: AsMut<[usize]> + AsRef<[usize]> + Debug + Clone;
    type Stride: AsMut<[isize]> + AsRef<[isize]> + Debug + Clone;

    /// Number of dimension
    fn ndim(&self) -> usize;

    /// Dynamic or static dimension
    fn is_dynamic() -> bool;
}

impl<const N: usize> DimBaseAPI for Ix<N> {
    type Shape = [usize; N];
    type Stride = [isize; N];

    fn ndim(&self) -> usize {
        N
    }

    fn is_dynamic() -> bool {
        false
    }
}

impl DimBaseAPI for IxD {
    type Shape = Vec<usize>;
    type Stride = Vec<isize>;

    fn ndim(&self) -> usize {
        self.len()
    }

    fn is_dynamic() -> bool {
        true
    }
}

pub trait DimAPI:
    DimBaseAPI + DimShapeAPI + DimStrideAPI + DimLayoutAPI + DimLayoutContigAPI
{
}

// currently, we make specialization to Ix<N>, so we can only implement finite N
// for now.
macro_rules! impl_dim_api {
    ($($N:literal),*) => {
        $(
            impl DimAPI for Ix<$N> {}
        )*
    };
}
impl_dim_api!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl DimAPI for IxD {}

/* #endregion */

/* #region dimension relative eq */

// Trait for defining smaller dimension by one.
pub trait DimSmallerOneAPI: DimAPI {
    type SmallerOne: DimAPI;
}

// Trait for defining larger dimension by one.
pub trait DimLargerOneAPI: DimAPI {
    type LargerOne: DimAPI;
}

impl DimSmallerOneAPI for IxD {
    type SmallerOne = IxD;
}

impl DimLargerOneAPI for IxD {
    type LargerOne = IxD;
}

macro_rules! impl_dim_smaller_one {
    ($(($N:literal, $N1:literal)),*) => {
        $(
            impl DimSmallerOneAPI for Ix<$N> {
                type SmallerOne = Ix<$N1>;
            }
        )*
    };
}

impl_dim_smaller_one!((1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8));

macro_rules! impl_dim_larger_one {
    ($(($N:literal, $N1:literal)),*) => {
        $(
            impl DimLargerOneAPI for Ix<$N> {
                type LargerOne = Ix<$N1>;
            }
        )*
    };
}

impl_dim_larger_one!((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9));

/// Trait for comparing two dimensions.
///
/// This trait is used to broadcast two tensors.
pub trait DimMaxAPI<D2>
where
    D2: DimBaseAPI,
{
    type Max: DimBaseAPI;
}

/// Same type of dimensions will always be able to be broadcasted.
impl<D> DimMaxAPI<D> for D
where
    D: DimBaseAPI,
{
    type Max = D;
}

macro_rules! impl_dim_max {
    ($(($N:literal, $N1:literal)),*) => {
        $(
            impl DimMaxAPI<Ix<$N1>> for Ix<$N> {
                type Max = Ix<$N1>;
            }
        )*
    };
}

impl_dim_max!((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9));
impl_dim_max!((1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9));
impl_dim_max!((2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9));
impl_dim_max!((3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9));
impl_dim_max!((4, 5), (4, 6), (4, 7), (4, 8), (4, 9));
impl_dim_max!((5, 6), (5, 7), (5, 8), (5, 9));
impl_dim_max!((6, 7), (6, 8), (6, 9));
impl_dim_max!((7, 8), (7, 9));
impl_dim_max!((8, 9));

macro_rules! impl_dim_max_dyn {
    ($($N:literal),*) => {
        $(
            impl DimMaxAPI<IxD> for Ix<$N> {
                type Max = IxD;
            }
        )*
    };
}

impl_dim_max_dyn!(0, 1, 2, 3, 4, 5, 6, 7, 8);

/* #endregion */
