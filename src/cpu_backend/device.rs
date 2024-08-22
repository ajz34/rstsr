use crate::prelude_dev::*;

#[derive(Clone, Debug)]
pub struct CpuDevice;

impl DeviceBaseAPI for CpuDevice {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> DeviceRawVecAPI<T> for CpuDevice {
    type RawVec = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for CpuDevice
where
    T: Clone,
{
    fn device(storage: &Storage<T, CpuDevice>) -> CpuDevice {
        storage.device.clone()
    }

    fn to_rawvec(storage: &Storage<T, CpuDevice>) -> Vec<T> {
        storage.rawvec.clone()
    }

    fn into_rawvec(storage: Storage<T, CpuDevice>) -> Vec<T> {
        storage.rawvec
    }

    fn new(vector: Vec<T>, device: CpuDevice) -> Storage<T, CpuDevice> {
        Storage::<T, CpuDevice> { rawvec: vector, device }
    }

    fn len(storage: &Storage<T, CpuDevice>) -> usize {
        storage.rawvec.len()
    }

    fn to_cpu_vec(storage: &Storage<T, CpuDevice>) -> Result<Vec<T>> {
        Ok(storage.rawvec.clone())
    }

    fn into_cpu_vec(storage: Storage<T, CpuDevice>) -> Result<Vec<T>> {
        Ok(storage.rawvec)
    }
}

impl<T> DeviceAPI<T> for CpuDevice where T: Clone {}

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
