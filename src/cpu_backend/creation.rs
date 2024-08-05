use super::*;
use crate::storage::*;
use crate::Error;
use crate::Result;
use core::fmt::Debug;
use num::complex::ComplexFloat;
use num::Float;
use num::Num;

impl<T> StorageCreationAPI for Storage<T, CpuDevice>
where
    T: Clone + Num + Debug,
{
    fn zeros_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::zero(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn ones_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::one(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    unsafe fn empty_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec: Vec<T> = Vec::with_capacity(len);
        unsafe {
            rawvec.set_len(len);
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn full_impl(device: &Self::Device, len: usize, fill: Self::DType) -> Result<Self> {
        let rawvec = vec![fill; len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn arange_int_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec = vec![];
        let mut v = T::zero();
        for _ in 0..len {
            rawvec.push(v.clone());
            v = v + T::one();
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn from_cpu_vec(device: &CpuDevice, vec: &Vec<T>) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec.clone();
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn outof_cpu_vec(device: &CpuDevice, vec: Vec<T>) -> Result<Storage<T, CpuDevice>> {
        Ok(Storage::<T, CpuDevice> { rawvec: vec, device: device.clone() })
    }
}

impl<T> StorageCreationComplexFloatAPI for Storage<T, CpuDevice>
where
    T: Clone + Debug + ComplexFloat,
{
    fn linspace_impl(
        device: &Self::Device,
        start: Self::DType,
        end: Self::DType,
        n: usize,
    ) -> Result<Self> {
        if n <= 1 {
            return Err(Error::InvalidInteger {
                value: n as isize,
                msg: "linspace requires at least two values.".to_string(),
            });
        }

        let mut rawvec = vec![];
        let step = (end - start) / T::from(n - 1).unwrap();
        let mut v = start.clone();
        for _ in 0..n {
            rawvec.push(v.clone());
            v = v + step.clone();
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }
}

impl<T> StorageCreationFloatAPI for Storage<T, CpuDevice>
where
    T: Clone + Debug + Float,
{
    fn arange_float_impl(
        device: &CpuDevice,
        start: T,
        end: T,
        step: T,
    ) -> Result<Storage<T, CpuDevice>> {
        if step == T::zero() {
            return Err(Error::InvalidValue { msg: "step must be non-zero.".to_string() });
        }
        if end < start {
            return Err(Error::InvalidValue {
                msg: format!("end {:?} must be greater than start {:?}", end, start),
            });
        }
        let n = ((end - start) / step).ceil().to_usize().unwrap();

        let rawvec = (0..n).map(|i| start + step * T::from(i).unwrap()).collect();
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_creation() {
        use super::*;
        use num::Complex;

        let device = CpuDevice {};
        let storage = Storage::<f32, CpuDevice>::zeros_impl(&device, 10).unwrap();
        println!("{:?}", storage);
        let storage = Storage::<f32, CpuDevice>::ones_impl(&device, 10).unwrap();
        println!("{:?}", storage);
        let storage = Storage::<f32, CpuDevice>::arange_int_impl(&device, 10).unwrap();
        println!("{:?}", storage);
        let storage = unsafe { Storage::<f32, CpuDevice>::empty_impl(&device, 10).unwrap() };
        println!("{:?}", storage);
        let storage = Storage::<f32, CpuDevice>::from_cpu_vec(&device, &vec![1.0; 10]).unwrap();
        println!("{:?}", storage);
        let storage = Storage::<f32, CpuDevice>::outof_cpu_vec(&device, vec![1.0; 10]).unwrap();
        println!("{:?}", storage);
        let storage = Storage::<f32, CpuDevice>::linspace_impl(&device, 0.0, 1.0, 10).unwrap();
        println!("{:?}", storage);
        let storage = Storage::<Complex<f32>, CpuDevice>::linspace_impl(
            &device,
            Complex::new(1.0, 2.0),
            Complex::new(3.5, 4.7),
            10,
        )
        .unwrap();
        println!("{:?}", storage);
        let storage = Storage::<f32, CpuDevice>::arange_float_impl(&device, 0.0, 1.0, 0.1).unwrap();
        println!("{:?}", storage);
    }
}
