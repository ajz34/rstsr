//! Implementation of function `asarray`.

use core::mem::ManuallyDrop;

use crate::prelude_dev::*;

pub trait AsArrayAPI<Input, B>: Sized {
    fn asarray(input: Input, device: Option<&B>) -> Result<Self>;
}

impl<T, B> AsArrayAPI<Vec<T>, B> for Tensor<T, Ix1, B>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray(input: Vec<T>, device: Option<&B>) -> Result<Self> {
        let layout = [input.len()].c();
        let device_binding = B::default();
        let device = device.unwrap_or(&device_binding);
        let storage = device.outof_cpu_vec(input)?;
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<T, B, const N: usize> AsArrayAPI<[T; N], B> for Tensor<T, Ix1, B>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    fn asarray(input: [T; N], device: Option<&B>) -> Result<Self> {
        let layout = [input.len()].c();
        let device_binding = B::default();
        let device = device.unwrap_or(&device_binding);
        let storage = device.outof_cpu_vec(input.into())?;
        let data = DataOwned::from(storage);
        let tensor = unsafe { Tensor::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

impl<'a, T> AsArrayAPI<&'a [T], CpuDevice> for TensorView<'a, T, Ix1, CpuDevice>
where
    T: Clone,
{
    fn asarray(input: &'a [T], _device: Option<&CpuDevice>) -> Result<Self> {
        let layout = [input.len()].c();
        let device = CpuDevice {};

        let ptr = input.as_ptr();
        let len = input.len();
        let rawvec: Vec<T> = unsafe {
            let ptr = ptr as *mut T;
            Vec::from_raw_parts(ptr, len, len)
        };
        let storage = ManuallyDrop::new(Storage::new(rawvec, device));
        let data = DataRef::from_manually_drop(storage);
        let tensor = unsafe { TensorView::new_unchecked(data, layout) };
        return Ok(tensor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asarray() {
        let input = vec![1, 2, 3];
        let tensor = Tensor::<i32, Ix1, CpuDevice>::asarray(input, None).unwrap();
        println!("{:?}", tensor);

        let input = vec![1, 2, 3];
        println!("{:?}", input.as_ptr());
        let tensor = TensorView::<i32, Ix1, CpuDevice>::asarray(&input, None).unwrap();
        println!("{:?}", tensor.data().storage().rawvec().as_ptr());
        println!("{:?}", tensor);
    }
}
