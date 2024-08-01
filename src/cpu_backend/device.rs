use crate::storage::{DeviceBasicAPI, DeviceWithDTypeAPI, DeviceToStorageAPI, DeviceAPI, Storage, StorageAPI};
use crate::Result;
use core::fmt::Debug;
use num::Num;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl DeviceBasicAPI for CpuDevice {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> DeviceWithDTypeAPI<T> for CpuDevice
where
    T: Clone,
{
    type RawVec = Vec<T>;
}

impl<T> StorageAPI<T, CpuDevice> for Storage<T, CpuDevice>
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

impl<T> DeviceToStorageAPI<T> for CpuDevice
where
    T: Clone + Num,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::zero(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn ones_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let rawvec = vec![T::one(); len];
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    fn arange_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec = vec![];
        let mut v = T::zero();
        for _ in 0..len {
            rawvec.push(v.clone());
            v = v + T::one();
        }
        Ok(Storage::<T, CpuDevice> { rawvec, device: self.clone() })
    }

    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, CpuDevice>> {
        let mut rawvec: Vec<T> = Vec::with_capacity(len);
        unsafe {
            rawvec.set_len(len);
        }
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

impl<T> DeviceAPI<T> for CpuDevice
where
    T: Clone + Num,
{
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
