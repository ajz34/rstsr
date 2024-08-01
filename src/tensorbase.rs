use crate::cpu_backend::device::CpuDevice;
use crate::layout::{Dimension, Layout};
use crate::storage::{Data, DataOwned, Storage};

#[derive(Debug, Clone)]
pub struct TensorBase<S, D>
where
    S: Data,
    D: Dimension,
{
    data: S,
    layout: Layout<D>,
}

pub type Tensor<T, D, B = CpuDevice> = TensorBase<DataOwned<Storage<T, B>>, D>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        use crate::layout::*;
        let a = Tensor::<f64, Ix<2>> { data: Storage { rawvec: vec![1.12345, 2.0], device: CpuDevice }.into(), layout: [1, 2].new_c_contig(0) };
        println!("{a:6.3?}");
    }
}
