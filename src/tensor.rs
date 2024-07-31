use crate::layout::{Dimension, Layout};
use crate::storage::Data;

pub struct TensorBase<S, D>
where
    S: Data,
    D: Dimension,
{
    data: S,
    layout: Layout<D>,
}
