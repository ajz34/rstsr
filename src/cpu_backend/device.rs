use crate::storage::{DeviceAPI, Storage, StorageAPI, StorageFromDeviceAPI, StorageToCpuAPI};
use crate::Result;
use core::fmt::Debug;
use num::Num;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl DeviceAPI for CpuDevice {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> StorageAPI for Storage<T, CpuDevice>
where
    T: Clone,
{
    type DType = T;
    type Device = CpuDevice;
    type RawVec = Vec<T>;

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

    fn len(&self) -> usize {
        self.rawvec.len()
    }
}

impl<T> StorageFromDeviceAPI for Storage<T, CpuDevice>
where
    T: Clone + Num,
{
    fn zeros_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::zero(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn ones_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::one(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    fn arange_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec = vec![];
        let mut v = T::zero();
        for _ in 0..len {
            rawvec.push(v.clone());
            v = v + T::one();
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: device.clone() })
    }

    unsafe fn empty_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec: Vec<T> = Vec::with_capacity(len);
        unsafe {
            rawvec.set_len(len);
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

impl<T> StorageToCpuAPI for Storage<T, CpuDevice>
where
    T: Clone,
{
    fn to_cpu_vec(&self) -> Result<Vec<T>> {
        Ok(self.rawvec.clone())
    }

    fn into_cpu_vec(self) -> Result<Vec<T>> {
        Ok(self.rawvec)
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
