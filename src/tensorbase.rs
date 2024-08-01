use crate::cpu_backend::device::CpuDevice;
use crate::layout::{DimAPI, Layout};
use crate::storage::{DataAPI, DataOwned, Storage};

#[derive(Debug, Clone)]
pub struct TensorBase<S, D>
where
    S: DataAPI,
    D: DimAPI,
{
    data: S,
    layout: Layout<D>,
}

impl<S, D> TensorBase<S, D>
where
    S: DataAPI,
    D: DimAPI,
{
    pub unsafe fn new(data: S, layout: Layout<D>) -> Self {
        Self { data, layout }
    }

    pub fn data(&self) -> &S {
        &self.data
    }

    pub fn layout(&self) -> &Layout<D> {
        &self.layout
    }
}

pub type Tensor<T, D, B = CpuDevice> = TensorBase<DataOwned<Storage<T, B>>, D>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        use crate::layout::*;
        let a = Tensor::<f64, Ix<2>> {
            data: Storage { rawvec: vec![1.12345, 2.0], device: CpuDevice }.into(),
            layout: [1, 2].new_c_contig(0),
        };
        println!("{a:6.3?}");
    }
}
