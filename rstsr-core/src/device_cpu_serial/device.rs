use crate::prelude_dev::*;

#[derive(Clone, Debug)]
pub struct DeviceCpuSerial;

impl Default for DeviceCpuSerial {
    fn default() -> Self {
        DeviceCpuSerial
    }
}

impl DeviceBaseAPI for DeviceCpuSerial {
    fn same_device(&self, _other: &Self) -> bool {
        true
    }
}

impl<T> DeviceRawVecAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    type RawVec = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    fn new(vector: Vec<T>, device: DeviceCpuSerial) -> Storage<T, DeviceCpuSerial> {
        Storage::<T, DeviceCpuSerial> { rawvec: vector, device }
    }

    fn len(storage: &Storage<T, DeviceCpuSerial>) -> usize {
        storage.rawvec.len()
    }

    fn to_cpu_vec(storage: &Storage<T, DeviceCpuSerial>) -> Result<Vec<T>> {
        Ok(storage.rawvec.clone())
    }

    fn into_cpu_vec(storage: Storage<T, DeviceCpuSerial>) -> Result<Vec<T>> {
        Ok(storage.rawvec)
    }

    #[inline]
    fn get_index(storage: &Storage<T, DeviceCpuSerial>, index: usize) -> T {
        storage.rawvec[index].clone()
    }

    #[inline]
    fn get_index_ptr(storage: &Storage<T, DeviceCpuSerial>, index: usize) -> *const T {
        &storage.rawvec[index] as *const T
    }

    #[inline]
    fn get_index_mut_ptr(storage: &mut Storage<T, DeviceCpuSerial>, index: usize) -> *mut T {
        &mut storage.rawvec[index] as *mut T
    }

    #[inline]
    fn set_index(storage: &mut Storage<T, Self>, index: usize, value: T) {
        storage.rawvec[index] = value;
    }
}

impl<T> DeviceAPI<T> for DeviceCpuSerial where T: Clone {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_same_device() {
        let device1 = DeviceCpuSerial;
        let device2 = DeviceCpuSerial;
        assert!(device1.same_device(&device2));
    }

    #[test]
    fn test_cpu_storage_to_vec() {
        let storage = Storage { rawvec: vec![1, 2, 3], device: DeviceCpuSerial };
        let vec = storage.to_rawvec();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_cpu_storage_into_vec() {
        let storage = Storage { rawvec: vec![1, 2, 3], device: DeviceCpuSerial };
        let vec = storage.into_rawvec();
        assert_eq!(vec, vec![1, 2, 3]);
    }
}
