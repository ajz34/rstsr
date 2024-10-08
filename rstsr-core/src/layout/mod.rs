pub mod broadcast;
pub mod dim;
pub mod indexer;
pub mod iterator_col_major;
pub mod iterator_row_major;
pub mod layoutbase;
pub mod matmul;
pub mod rearrangement;
pub mod shape;
pub mod slice;
pub mod stride;

pub use broadcast::*;
pub use dim::*;
pub use indexer::*;
pub use iterator_col_major::*;
pub use iterator_row_major::*;
pub use layoutbase::*;
pub use matmul::*;
pub use rearrangement::*;
pub use shape::*;
pub use slice::*;
pub use stride::*;

pub trait DimDevAPI: DimBaseAPI + DimShapeAPI + DimStrideAPI + DimLayoutContigAPI {}

impl<const N: usize> DimDevAPI for Ix<N> {}
impl DimDevAPI for IxD {}

pub trait DimAPI:
    DimDevAPI
    + DimConvertAPI<IxD>
    + DimConvertAPI<Ix0>
    + DimConvertAPI<Ix1>
    + DimConvertAPI<Ix2>
    + DimConvertAPI<Ix3>
    + DimConvertAPI<Ix4>
    + DimConvertAPI<Ix5>
    + DimConvertAPI<Ix6>
    + DimConvertAPI<Ix7>
    + DimConvertAPI<Ix8>
    + DimConvertAPI<Ix9>
    + DimConvertAPI<Self>
{
}

impl<const N: usize> DimAPI for Ix<N> {}
impl DimAPI for IxD {}
