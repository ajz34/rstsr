use crate::layout::{DimAPI, Layout, LayoutAPI};
use crate::storage::{DataOwned, DeviceAPI, DeviceToStorageAPI, Storage};
use crate::{Tensor, TensorBase};

pub trait TensorCreationAPI<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    fn zeros(layout: impl Into<Layout<D>>, device: B) -> TensorBase<DataOwned<Storage<T, B>>, D>;
}

impl<T, D, B> TensorCreationAPI<T, D, B> for Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToStorageAPI<T>,
    Layout<D>: LayoutAPI,
{
    fn zeros(layout: impl Into<Layout<D>>, device: B) -> Tensor<T, D, B> {
        let layout = layout.into();
        let data = device.zeros_impl(layout.size()).unwrap();
        unsafe { Tensor::new(data.into(), layout) }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        use crate::cpu_backend::device::CpuDevice;
        use crate::layout::*;
        let a = Tensor::<f64, _, _>::zeros([2, 2], CpuDevice);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _, _>::zeros([2, 2].new_f_contig(0), CpuDevice);
        println!("{a:6.3?}");
    }
}
