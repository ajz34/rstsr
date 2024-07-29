use crate::base_data::TraitData;
use crate::base_layout::{Layout, TraitDimension};

pub struct TensorBase<S, D>
where
    S: TraitData,
    D: TraitDimension,
{
    data: S,
    layout: Layout<D>,
}
