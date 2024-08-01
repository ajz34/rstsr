use crate::{TensorBase, Tensor};
use crate::layout::{Layout, DimAPI, LayoutAPI};
use crate::storage::{DataOwned, DeviceAPI, Storage, DeviceToStorageAPI};

pub trait TensorCreationAPI<T, D, B>
where 
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    fn zeros(layout: Layout<D>, device: B) -> TensorBase<DataOwned<Storage<T, B>>, D>;
}

impl<T, D, B> TensorCreationAPI<T, D, B> for Tensor<T, D, B>
where 
    T: Clone,
    D: DimAPI,
    Layout<D>: LayoutAPI,
    B: DeviceAPI<T> + DeviceToStorageAPI<T, B>
{
    fn zeros(layout: Layout<D>, device: B) -> Tensor<T, D, B> {
        let data = device.zeros_impl(layout.size()).unwrap();
        Tensor::new(data.into(), layout)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        use crate::layout::*;
        use crate::cpu_backend::device::CpuDevice;
        let a = Tensor::<f64, _, _>::zeros([1, 2].new_c_contig(0), CpuDevice);
        println!("{a:6.3?}");
    }
}
