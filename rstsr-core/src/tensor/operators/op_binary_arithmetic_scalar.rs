use crate::prelude_dev::*;

use super::TensorAddAPI;

use core::ops::*;

impl<R, D, B> TensorAddAPI<&TensorBase<R, D>> for i32
where
    R: DataAPI<Data = Storage<f64, B>>,
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceCreationAnyAPI<f64>,
    B: DeviceAddAPI<f64, f64, f64, D>,
{
    type Output = Tensor<f64, D, B>;
    fn add(a: Self, b: &TensorBase<R, D>) -> Result<Self::Output> {
        let a = f64::from(a);
        let device = b.device();
        let lb = b.layout();
        let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        let storage_b = b.storage();
        device.op_mutc_numa_refb(&mut storage_c, &lc, a, storage_b, lb)?;
        Tensor::new(DataOwned::from(storage_c), lc)
    }
}

impl<R, D, B> Add<&TensorBase<R, D>> for i32
where
    R: DataAPI<Data = Storage<f64, B>>,
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceCreationAnyAPI<f64>,
    B: DeviceAddAPI<f64, f64, f64, D>,
{
    type Output = Tensor<f64, D, B>;
    fn add(self, rhs: &TensorBase<R, D>) -> Self::Output {
        TensorAddAPI::add(self, rhs).unwrap()
    }
}

impl<'l, D, B> TensorAddAPI<TensorView<'l, f64, D, B>> for i32
where
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceCreationAnyAPI<f64>,
    B: DeviceAddAPI<f64, f64, f64, D>,
{
    type Output = Tensor<f64, D, B>;
    fn add(a: Self, b: TensorView<'l, f64, D, B>) -> Result<Self::Output> {
        let a = f64::from(a);
        let device = b.device();
        let lb = b.layout();
        let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        let storage_b = b.storage();
        device.op_mutc_numa_refb(&mut storage_c, &lc, a, storage_b, lb)?;
        Tensor::new(DataOwned::from(storage_c), lc)
    }
}

impl<'l, D, B> Add<TensorView<'l, f64, D, B>> for i32
where
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceCreationAnyAPI<f64>,
    B: DeviceAddAPI<f64, f64, f64, D>,
{
    type Output = Tensor<f64, D, B>;
    fn add(self, rhs: TensorView<'l, f64, D, B>) -> Self::Output {
        TensorAddAPI::add(self, rhs).unwrap()
    }
}

impl<D, B> TensorAddAPI<Tensor<f64, D, B>> for i32
where
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceCreationAnyAPI<f64>,
    B: DeviceRConsumeAddAPI<f64, f64, D>,
{
    type Output = Tensor<f64, D, B>;
    fn add(a: Self, mut b: Tensor<f64, D, B>) -> Result<Self::Output> {
        let a = f64::from(a);
        let device = b.device().clone();
        let lb = b.layout().clone();
        let storage_b = b.data_mut().storage_mut();
        device.op_muta_numb(storage_b, &lb, a)?;
        return Ok(b);
    }
}

impl<D, B> Add<Tensor<f64, D, B>> for i32
where
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceCreationAnyAPI<f64>,
    B: DeviceRConsumeAddAPI<f64, f64, D>,
{
    type Output = Tensor<f64, D, B>;
    fn add(self, rhs: Tensor<f64, D, B>) -> Self::Output {
        TensorAddAPI::add(self, rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = 1;
        let c = b + &a;
        println!("{:?}", c);
    }
}
