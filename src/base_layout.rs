pub trait TraitDimension {}

pub struct Layout<D> {
    dim: D,
    offset: usize,
    strides: [usize; 1],
    shape: [usize; 1],
}
