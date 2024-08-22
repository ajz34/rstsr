use crate::cpu_backend::device::CpuDevice;
use crate::prelude_dev::*;

pub trait DeviceBaseAPI: Clone + Debug {
    fn same_device(&self, other: &Self) -> bool;
}

pub trait DeviceRawVecAPI<T>: DeviceBaseAPI {
    type RawVec;
}

#[derive(Debug, Clone)]
pub struct Storage<T, B = CpuDevice>
where
    B: DeviceRawVecAPI<T>,
{
    pub(crate) rawvec: B::RawVec,
    pub(crate) device: B,
}

pub trait DeviceStorageAPI<T>: DeviceRawVecAPI<T> {
    fn device(storage: &Storage<T, Self>) -> Self;
    fn to_rawvec(storage: &Storage<T, Self>) -> Self::RawVec;
    fn into_rawvec(storage: Storage<T, Self>) -> Self::RawVec;
    fn new(vector: Self::RawVec, device: Self) -> Storage<T, Self>;
    fn len(storage: &Storage<T, Self>) -> usize;
    fn is_empty(storage: &Storage<T, Self>) -> bool {
        storage.len() == 0
    }
    fn to_cpu_vec(storage: &Storage<T, Self>) -> Result<Vec<T>>;
    fn into_cpu_vec(storage: Storage<T, Self>) -> Result<Vec<T>>;
}

impl<T, B> Storage<T, B>
where
    B: DeviceStorageAPI<T>,
{
    pub fn device(&self) -> B {
        B::device(self)
    }

    pub fn to_rawvec(&self) -> B::RawVec {
        B::to_rawvec(self)
    }

    pub fn into_rawvec(self) -> B::RawVec {
        B::into_rawvec(self)
    }

    pub fn new(vector: B::RawVec, device: B) -> Self {
        Self { rawvec: vector, device }
    }

    pub fn len(&self) -> usize {
        B::len(self)
    }

    pub fn is_empty(&self) -> bool {
        B::is_empty(self)
    }

    pub fn to_cpu_vec(&self) -> Result<Vec<T>> {
        B::to_cpu_vec(self)
    }

    pub fn into_cpu_vec(self) -> Result<Vec<T>> {
        B::into_cpu_vec(self)
    }
}

pub trait DeviceAPI<T>: DeviceBaseAPI + DeviceRawVecAPI<T> + DeviceStorageAPI<T> {}

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
