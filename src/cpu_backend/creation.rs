use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Float, Num};

#[test]
fn test() {
    let t = Storage::<f64, CpuDevice> { rawvec: vec![], device: CpuDevice {} };
    println!("{:?}", t);
}

impl<T> DeviceCreationAnyAPI<T> for CpuDevice
where
    T: Clone + Debug,
    CpuDevice: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    #[allow(clippy::uninit_vec)]
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec: Vec<T> = Vec::with_capacity(len);
        unsafe {
            rawvec.set_len(len);
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![fill; len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<T, CpuDevice>> {
        Ok(Storage::<T, CpuDevice> { rawvec: vec, device: self.clone() })
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec.to_vec();
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }
}

impl<T> DeviceCreationNumAPI<T> for CpuDevice
where
    T: Num + Clone + Debug,
    CpuDevice: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::zero(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::one(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn arange_int_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec = vec![];
        let mut v = T::zero();
        for _ in 0..len {
            rawvec.push(v.clone());
            v = v + T::one();
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for CpuDevice
where
    T: ComplexFloat + Clone + Debug,
    CpuDevice: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn linspace_impl(&self, start: T, end: T, n: usize) -> Result<Storage<T, CpuDevice>>
    where
        T: ComplexFloat,
    {
        // handle special cases
        if n == 0 {
            return Ok(Storage::<T, CpuDevice> { rawvec: vec![], device: self.clone() });
        } else if n == 1 {
            return Ok(Storage::<T, CpuDevice> { rawvec: vec![start], device: self.clone() });
        }

        let mut rawvec = vec![];
        let step = (end - start) / T::from(n - 1).unwrap();
        let mut v = start;
        for _ in 0..n {
            rawvec.push(v);
            v = v + step;
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }
}

impl<T> DeviceCreationFloatAPI<T> for CpuDevice
where
    T: Float + Clone + Debug,
    CpuDevice: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<T, CpuDevice>> {
        rstsr_assert!(step != T::zero(), InvalidValue)?;
        let n = ((end - start) / step).ceil();
        rstsr_pattern!(n, T::zero().., ValueOutOfRange)?;
        let n = n.to_usize().unwrap();

        let mut rawvec: Vec<T> = (0..n).map(|i| start + step * T::from(i).unwrap()).collect();
        if rawvec.last().is_some_and(|&v| v == end) {
            rawvec.pop();
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_creation() {
        use super::*;
        use num::Complex;

        let device = CpuDevice {};
        let storage: Storage<f64> = device.zeros_impl(10).unwrap();
        println!("{:?}", storage);
        let storage: Storage<f64> = device.ones_impl(10).unwrap();
        println!("{:?}", storage);
        let storage: Storage<f64> = device.arange_int_impl(10).unwrap();
        println!("{:?}", storage);
        let storage: Storage<f64> = unsafe { device.empty_impl(10).unwrap() };
        println!("{:?}", storage);
        let storage = device.from_cpu_vec(&[1.0; 10]).unwrap();
        println!("{:?}", storage);
        let storage = device.outof_cpu_vec(vec![1.0; 10]).unwrap();
        println!("{:?}", storage);
        let storage = device.linspace_impl(0.0, 1.0, 10).unwrap();
        println!("{:?}", storage);
        let storage =
            device.linspace_impl(Complex::new(1.0, 2.0), Complex::new(3.5, 4.7), 10).unwrap();
        println!("{:?}", storage);
        let storage = device.arange_impl(0.0, 1.0, 0.1).unwrap();
        println!("{:?}", storage);
    }
}
