pub mod broadcast;
pub mod dim;
pub mod indexer;
pub mod iterator;
pub mod layoutbase;
pub mod shape;
pub mod slice;
pub mod stride;

pub use broadcast::*;
pub use dim::*;
pub use indexer::*;
pub use iterator::*;
pub use layoutbase::*;
pub use shape::*;
pub use slice::*;
pub use stride::*;

pub trait DimDevAPI:
    DimBaseAPI + DimShapeAPI + DimStrideAPI + DimIndexUncheckAPI + DimLayoutContigAPI
{
}

impl<const N: usize> DimDevAPI for Ix<N> {}
impl DimDevAPI for IxD {}

pub trait DimAPI:
    DimDevAPI + DimIterLayoutAPI<IterLayoutC<Self>> + DimIterLayoutAPI<IterLayoutF<Self>>
{
}

impl<const N: usize> DimAPI for Ix<N> {}
impl DimAPI for IxD {}
