use super::*;
use core::fmt::Debug;

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
impl<const N: usize> DimAPI for Ix<N> {}
impl DimAPI for IxD {}
