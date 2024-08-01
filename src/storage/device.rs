use crate::cpu_backend::device::CpuDevice;
use crate::Result;

pub trait DeviceAPI: Clone {
    fn same_device(&self, other: &Self) -> bool;
}

pub trait DeviceWithDTypeAPI<T>
where
    T: Clone,
{
    type RawVec;
}

#[derive(Debug, Clone)]
pub struct Storage<T, B = CpuDevice>
where
    T: Clone,
    B: DeviceWithDTypeAPI<T>,
{
    pub(crate) rawvec: B::RawVec,
    pub(crate) device: B,
}

pub trait StorageAPI<T, B>
where
    T: Clone,
    B: DeviceWithDTypeAPI<T>,
    Self: Sized,
{
    fn device(&self) -> B;
    fn to_rawvec(&self) -> B::RawVec;
    fn into_rawvec(self) -> B::RawVec;
    fn new(vector: B::RawVec, device: B) -> Self;
}

pub trait DeviceToStorageAPI<T>
where
    T: Clone,
    Self: DeviceAPI + DeviceWithDTypeAPI<T>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    fn ones_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    fn arange_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, Self>>;
    fn from_cpu_vec_owned(&self, vec: Vec<T>) -> Result<Storage<T, Self>>;
    fn from_cpu_vec(&self, vec: &Vec<T>) -> Result<Storage<T, Self>>;
}

pub trait DeviceWithStorageAPI<T>
where
    T: Clone,
    Self: DeviceAPI + DeviceWithDTypeAPI<T> + DeviceToStorageAPI<T>,
{
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
