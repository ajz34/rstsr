use crate::data::TraitData;
use crate::layout::{Layout, TraitDimension};

pub struct TensorBase<S, D>
where
    S: TraitData,
    D: TraitDimension,
{
    data: S,
    layout: Layout<D>,
}
