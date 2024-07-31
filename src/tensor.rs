use crate::data::Data;
use crate::layout::{Dimension, Layout};

pub struct TensorBase<S, D>
where
    S: Data,
    D: Dimension,
{
    data: S,
    layout: Layout<D>,
}
