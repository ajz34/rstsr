use crate::cpu_backend::device::CpuDevice;
use crate::layout::{DimAPI, Layout};
use crate::storage::{DeviceAPI, Storage, StorageAPI, StorageFromDeviceAPI};
use crate::{Result, Tensor};
use num::Num;

pub trait TensorCreationWithDeviceAPI: Sized {
    type DType;
    type Dim: DimAPI;
    type Device;

    fn zeros_with_device(
        layout: impl Into<Layout<Self::Dim>>,
        device: Self::Device,
    ) -> Result<Self>;

    fn from_shape_with_device(
        layout: impl Into<Layout<Self::Dim>>,
        vec: &[Self::DType],
        device: Self::Device,
    ) -> Result<Self>;
}

impl<T, D, B> TensorCreationWithDeviceAPI for Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI,
    Storage<T, B>: StorageAPI<DType = T, Device = B> + StorageFromDeviceAPI,
{
    type DType = T;
    type Dim = D;
    type Device = B;

    fn zeros_with_device(layout: impl Into<Layout<D>>, device: B) -> Result<Tensor<T, D, B>> {
        let layout = layout.into();
        let (_, idx_max) = layout.bounds_index()?;
        let data: Storage<T, B> = StorageFromDeviceAPI::zeros_impl(&device, idx_max).unwrap();
        Tensor::new(data.into(), layout)
    }

    fn from_shape_with_device(
        layout: impl Into<Layout<D>>,
        vec: &[T],
        device: B,
    ) -> Result<Tensor<T, D, B>> {
        let layout = layout.into();
        let data: Storage<T, B> =
            StorageFromDeviceAPI::outof_cpu_vec(&device, vec.to_vec()).unwrap();
        Tensor::new(data.into(), layout)
    }
}

pub trait TensorCreationCpuAPI: Sized {
    type DType;
    type Dim: DimAPI;

    fn zeros(layout: impl Into<Layout<Self::Dim>>) -> Result<Self>;
    fn from_shape(layout: impl Into<Layout<Self::Dim>>, vec: &[Self::DType]) -> Result<Self>;
}

impl<T, D> TensorCreationCpuAPI for Tensor<T, D, CpuDevice>
where
    T: Clone + Num,
    D: DimAPI,
{
    type DType = T;
    type Dim = D;

    fn zeros(layout: impl Into<Layout<D>>) -> Result<Tensor<T, D, CpuDevice>> {
        Tensor::<T, D, CpuDevice>::zeros_with_device(layout, CpuDevice)
    }

    fn from_shape(layout: impl Into<Layout<D>>, vec: &[T]) -> Result<Tensor<T, D, CpuDevice>> {
        Tensor::<T, D, CpuDevice>::from_shape_with_device(layout, vec, CpuDevice)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::layout::*;

    #[test]
    fn playground() {
        use crate::cpu_backend::device::CpuDevice;
        let a = Tensor::<f64, _, _>::zeros_with_device([2, 2], CpuDevice);
        println!("{a:6.3?}");
        let a = Tensor::<f64, _, _>::zeros_with_device([2, 2].f(), CpuDevice);
        println!("{a:6.3?}");
    }
}
