//! This implementation uses code from `candle`.
use super::error::CudaDriverError;
use crate::base_device::{DeviceId, TraitDevice, TraitDeviceToStorage, TraitStorage};
use crate::Result;
use cudarc;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct CudaDevice {
    id: DeviceId,
    device: Arc<cudarc::driver::CudaDevice>,
}

/// Implementation of getters
impl CudaDevice {
    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn device(&self) -> Arc<cudarc::driver::CudaDevice> {
        self.device.clone()
    }
}

/// Implementation of other utilities
impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal).map_err(CudaDriverError)?;
        Ok(Self { id: DeviceId::new(), device })
    }

    pub fn ordinal(&self) -> usize {
        self.device.ordinal()
    }
}

impl TraitDevice for CudaDevice {
    fn same_device(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

/// Inheritance to `cudarc::driver::CudaDevice`
///
/// **note** Uses implementation of `candle`.
impl std::ops::Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaDevice>;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

pub struct CudaStorage<T> {
    vector: CudaSlice<T>,
    device: CudaDevice,
}

impl<T> TraitStorage for CudaStorage<T>
where
    T: DeviceRepr,
{
    type Device = CudaDevice;
    type DType = T;
    type VType = CudaSlice<T>;

    fn device(&self) -> CudaDevice {
        self.device.clone()
    }

    fn to_rawvec(&self) -> CudaSlice<T> {
        self.vector.clone()
    }

    fn into_rawvec(self) -> CudaSlice<T> {
        self.vector
    }

    fn new(rawvec: Self::VType, device: Self::Device) -> Self {
        Self { vector: rawvec, device }
    }
}

impl<T> TraitDeviceToStorage<CudaStorage<T>> for CudaDevice
where
    T: Unpin + DeviceRepr + ValidAsZeroBits,
{
    fn zeros_impl(&self, len: usize) -> Result<CudaStorage<T>> {
        let vec = self.alloc_zeros::<T>(len).map_err(CudaDriverError)?;
        Ok(CudaStorage::new(vec, self.clone()))
    }

    fn from_cpu_vec(&self, vec: &Vec<T>) -> Result<CudaStorage<T>> {
        let vec = self.htod_sync_copy(vec).map_err(CudaDriverError)?;
        Ok(CudaStorage::new(vec, self.clone()))
    }

    fn from_cpu_vec_owned(&self, vec: Vec<T>) -> Result<CudaStorage<T>> {
        let vec = self.htod_copy(vec).map_err(CudaDriverError)?;
        Ok(CudaStorage::new(vec, self.clone()))
    }
}
