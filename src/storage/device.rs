use crate::cpu_backend::device::CpuDevice;
use crate::Result;
use core::fmt::Debug;

pub trait DeviceAPI: Clone + Debug {
    fn same_device(&self, other: &Self) -> bool;
}

pub struct Storage<T, B = CpuDevice>
where
    Self: StorageAPI<DType = T, Device = B>,
{
    pub(crate) rawvec: <Self as StorageAPI>::RawVec,
    pub(crate) device: <Self as StorageAPI>::Device,
}

pub trait StorageAPI: Sized {
    type DType;
    type Device;
    type RawVec;
    fn device(&self) -> Self::Device;
    fn to_rawvec(&self) -> Self::RawVec;
    fn into_rawvec(self) -> Self::RawVec;
    fn new(vector: Self::RawVec, device: Self::Device) -> Self;
    fn len(&self) -> usize;
}

pub trait StorageToCpuAPI: StorageAPI {
    fn to_cpu_vec(&self) -> Result<Vec<Self::DType>>;
    fn into_cpu_vec(self) -> Result<Vec<Self::DType>>;
}

/// Unique identifier for cuda devices.
///
/// This code is from `candle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    #[allow(unused)]
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}
