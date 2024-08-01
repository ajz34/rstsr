use crate::layout::{DimAPI, Layout, LayoutAPI};
use crate::storage::{DataOwned, DeviceAPI, DeviceWithStorageAPI, Storage};
use crate::{Tensor, TensorBase};

pub trait TensorCreationAPI<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceWithStorageAPI<T>,
{
    fn zeros(layout: Layout<D>, device: B) -> TensorBase<DataOwned<Storage<T, B>>, D>;
}

impl<T, D, B> TensorCreationAPI<T, D, B> for Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI,
    B: DeviceWithStorageAPI<T>,
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
        use crate::cpu_backend::device::CpuDevice;
        use crate::layout::*;
        let a = Tensor::<f64, _, _>::zeros([1, 2].new_c_contig(0), CpuDevice);
        println!("{a:6.3?}");
    }
}
