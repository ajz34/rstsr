use super::*;
use crate::storage::*;
use crate::Result;
use num::Num;

impl<T> StorageCreationAPI for Storage<T, CpuDevice>
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

    fn arange_int_impl(device: &CpuDevice, len: usize) -> Result<Storage<T, CpuDevice>> {
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
