use crate::device::{TraitDevice, TraitDeviceToStorage, TraitStorage};
use crate::Result;
use core::fmt::Debug;
use num_traits::Num;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl TraitDevice for CpuDevice {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

pub struct CpuStorage<T> {
    rawvec: Vec<T>,
    device: CpuDevice,
}

impl<T> TraitStorage for CpuStorage<T>
where
    T: Clone,
{
    type Device = CpuDevice;
    type DType = T;
    type VType = Vec<T>;

    fn device(&self) -> CpuDevice {
        self.device.clone()
    }

    fn to_rawvec(&self) -> Vec<T> {
        self.rawvec.clone()
    }

    fn into_rawvec(self) -> Vec<T> {
        self.rawvec
    }

    fn new(vector: Self::VType, device: Self::Device) -> Self {
        Self { rawvec: vector, device }
    }
}

impl<T> TraitDeviceToStorage<CpuStorage<T>> for CpuDevice
where
    T: Clone + Num,
{
    fn zeros_impl(&self, len: usize) -> Result<CpuStorage<T>> {
        let vec = vec![T::zero(); len];
        Ok(CpuStorage::new(vec, self.clone()))
    }

    fn from_cpu_vec(&self, vec: &Vec<T>) -> Result<CpuStorage<T>> {
        let vec = vec.clone();
        Ok(CpuStorage::new(vec, self.clone()))
    }

    fn from_cpu_vec_owned(&self, vec: Vec<T>) -> Result<CpuStorage<T>> {
        Ok(CpuStorage::new(vec, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_same_device() {
        let device1 = CpuDevice;
        let device2 = CpuDevice;
        assert!(device1.same_device(&device2));
    }

    #[test]
    fn test_cpu_storage_to_vec() {
        let storage = CpuStorage::new(vec![1, 2, 3], CpuDevice);
        let vec = storage.to_rawvec();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_cpu_storage_into_vec() {
        let storage = CpuStorage::new(vec![1, 2, 3], CpuDevice);
        let vec = storage.into_rawvec();
        assert_eq!(vec, vec![1, 2, 3]);
    }
}
