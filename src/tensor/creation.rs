use num::Num;

use crate::cpu_backend::device::CpuDevice;
use crate::layout::{DimAPI, Layout, LayoutAPI, Shape, ShapeAPI, Stride};
use crate::storage::{DataOwned, DeviceAPI, DeviceToStorageAPI, Storage, StorageAPI};
use crate::{Error, Result};
use crate::{Tensor, TensorBase};

pub trait TensorCreationWithDeviceAPI<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    fn zeros_with_device(layout: impl Into<Layout<D>>, device: B) -> Result<Tensor<T, D, B>>;
    fn from_shape_with_device(
        layout: impl Into<Layout<D>>,
        vec: &[T],
        device: B,
    ) -> Result<Tensor<T, D, B>>;
}

impl<T, D, B> TensorCreationWithDeviceAPI<T, D, B> for Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToStorageAPI<T>,
    Storage<T, B>: StorageAPI,
    Layout<D>: LayoutAPI,
{
    fn zeros_with_device(layout: impl Into<Layout<D>>, device: B) -> Result<Tensor<T, D, B>> {
        let layout = layout.into();
        let data = device.zeros_impl(layout.size()).unwrap();
        Tensor::new(data.into(), layout)
    }

    fn from_shape_with_device(
        layout: impl Into<Layout<D>>,
        vec: &[T],
        device: B,
    ) -> Result<Tensor<T, D, B>> {
        let layout = layout.into();
        let data = device.from_cpu_vec_owned(vec.to_vec()).unwrap();
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
        let a = Tensor::<f64, _, _>::zeros_with_device([2, 2], CpuDevice);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _, _>::zeros_with_device([2, 2].new_f_contig(0), CpuDevice);
        println!("{a:6.3?}");
    }
}
