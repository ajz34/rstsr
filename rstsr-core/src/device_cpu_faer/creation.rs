use super::device::DeviceCpuFaer;
use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Float, Num};
use rayon::prelude::*;

impl<T> DeviceCreationAnyAPI<T> for DeviceCpuFaer
where
    T: Clone + Debug,
    DeviceCpuFaer: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    #[allow(clippy::uninit_vec)]
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, DeviceCpuFaer>> {
        let mut rawvec: Vec<T> = Vec::with_capacity(len);
        unsafe {
            rawvec.set_len(len);
        }
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }

    fn full_impl(&self, len: usize, fill: T) -> Result<Storage<T, DeviceCpuFaer>> {
        let rawvec = vec![fill; len];
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }

    fn outof_cpu_vec(&self, vec: Vec<T>) -> Result<Storage<T, DeviceCpuFaer>> {
        Ok(Storage::<T, DeviceCpuFaer> { rawvec: vec, device: self.clone() })
    }

    fn from_cpu_vec(&self, vec: &[T]) -> Result<Storage<T, DeviceCpuFaer>> {
        let rawvec = vec.to_vec();
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }
}

impl<T> DeviceCreationNumAPI<T> for DeviceCpuFaer
where
    T: Num + Clone + Debug,
    DeviceCpuFaer: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, DeviceCpuFaer>> {
        let rawvec = vec![T::zero(); len];
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<T, DeviceCpuFaer>> {
        let rawvec = vec![T::one(); len];
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }

    fn arange_int_impl(&self, len: usize) -> Result<Storage<T, DeviceCpuFaer>> {
        let mut rawvec = Vec::with_capacity(len);
        let mut v = T::zero();
        for _ in 0..len {
            rawvec.push(v.clone());
            v = v + T::one();
        }
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }
}

impl<T> DeviceCreationComplexFloatAPI<T> for DeviceCpuFaer
where
    T: ComplexFloat + Clone + Debug + Send + Sync,
    DeviceCpuFaer: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn linspace_impl(
        &self,
        start: T,
        end: T,
        n: usize,
        endpoint: bool,
    ) -> Result<Storage<T, DeviceCpuFaer>> {
        // handle special cases
        if n == 0 {
            return Ok(Storage::<T, DeviceCpuFaer> { rawvec: vec![], device: self.clone() });
        } else if n == 1 {
            return Ok(Storage::<T, DeviceCpuFaer> { rawvec: vec![start], device: self.clone() });
        }

        let step = match endpoint {
            true => (end - start) / T::from(n - 1).unwrap(),
            false => (end - start) / T::from(n).unwrap(),
        };

        let mut rawvec: Vec<T> = Vec::with_capacity(n);
        let pool = self.get_pool(0).unwrap();
        pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|idx| start + step * T::from(idx).unwrap())
                .collect_into_vec(&mut rawvec)
        });
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }
}

impl<T> DeviceCreationFloatAPI<T> for DeviceCpuFaer
where
    T: Float + Clone + Debug + Send + Sync,
    DeviceCpuFaer: DeviceRawVecAPI<T, RawVec = Vec<T>>,
{
    fn arange_impl(&self, start: T, end: T, step: T) -> Result<Storage<T, Self>> {
        rstsr_assert!(step != T::zero(), InvalidValue)?;
        let n = ((end - start) / step).ceil();
        rstsr_pattern!(n, T::zero().., ValueOutOfRange)?;
        let n = n.to_usize().unwrap();

        let mut rawvec: Vec<T> = Vec::with_capacity(n);
        let pool = self.get_pool(0).unwrap();
        pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|idx| start + step * T::from(idx).unwrap())
                .collect_into_vec(&mut rawvec)
        });
        if rawvec.last().is_some_and(|&v| v == end) {
            rawvec.pop();
        }
        Ok(Storage::<T, DeviceCpuFaer> { rawvec, device: self.clone() })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_linspace() {
        let device = DeviceCpuFaer::default();
        let a = Tensor::linspace(1.0, 5.0, 5, &device);
        assert_eq!(a.data().storage().rawvec(), &vec![1., 2., 3., 4., 5.]);
    }
}
