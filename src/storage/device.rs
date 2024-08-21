use crate::cpu_backend::device::CpuDevice;
use crate::prelude_dev::*;

pub trait DeviceAPI: Clone + Debug {
    fn same_device(&self, other: &Self) -> bool;
}

#[derive(Debug, Clone)]
pub struct Storage<T, B = CpuDevice>
where
    Self: StorageBaseAPI,
{
    pub(crate) rawvec: <Self as StorageBaseAPI>::RawVec,
    pub(crate) device: <Self as StorageBaseAPI>::Device,
}

pub trait StorageBaseAPI: Sized {
    type DType;
    type Device: Clone + Debug;
    type RawVec: Clone + Debug;
    fn device(&self) -> Self::Device;
    fn to_rawvec(&self) -> Self::RawVec;
    fn into_rawvec(self) -> Self::RawVec;
    fn new(vector: Self::RawVec, device: Self::Device) -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait StorageToCpuAPI: StorageBaseAPI {
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
        use core::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}
