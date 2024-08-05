use crate::storage::*;
use crate::Result;
use core::fmt::Debug;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl DeviceAPI for CpuDevice {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> StorageAPI for Storage<T, CpuDevice>
where
    T: Clone + Debug,
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

impl<T> StorageToCpuAPI for Storage<T, CpuDevice>
where
    T: Clone + Debug,
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
