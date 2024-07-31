use crate::device::{Device, DeviceWithDType, Storage, TraitDeviceToStorage, TraitStorage};
use crate::Result;
use core::fmt::Debug;
use num::Num;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> DeviceWithDType<T> for CpuDevice
where
    T: Clone,
{
    type RawVec = Vec<T>;
}

impl<T> TraitStorage<T, CpuDevice> for Storage<T, CpuDevice>
where
    T: Clone,
{
    fn device(&self) -> CpuDevice {
        self.device.clone()
    }

    fn to_rawvec(&self) -> Vec<T> {
        self.rawvec.clone()
    }

    fn into_rawvec(self) -> Vec<T> {
        self.rawvec
    }

    fn new(vector: Vec<T>, device: CpuDevice) -> Self {
        Self { rawvec: vector, device }
    }
}

impl<T> TraitDeviceToStorage<T, CpuDevice> for CpuDevice
where
    T: Clone + Num,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::zero(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn from_cpu_vec(&self, vec: &Vec<T>) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec.clone();
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn from_cpu_vec_owned(&self, vec: Vec<T>) -> Result<Storage<T, CpuDevice>> {
        Ok(Storage::<T, CpuDevice> { rawvec: vec, device: self.clone() })
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
        let storage = Storage { rawvec: vec![1, 2, 3], device: CpuDevice };
        let vec = storage.to_rawvec();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_cpu_storage_into_vec() {
        let storage = Storage { rawvec: vec![1, 2, 3], device: CpuDevice };
        let vec = storage.into_rawvec();
        assert_eq!(vec, vec![1, 2, 3]);
    }
}
