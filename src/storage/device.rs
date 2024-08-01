use crate::cpu_backend::device::CpuDevice;
use crate::Result;

pub trait Device: Clone {
    fn same_device(&self, other: &Self) -> bool;
}

pub trait DeviceWithDType<T>
where
    T: Clone,
{
    type RawVec;
}

#[derive(Debug, Clone)]
pub struct Storage<T, D = CpuDevice>
where
    T: Clone,
    D: Device + DeviceWithDType<T>,
{
    pub(crate) rawvec: D::RawVec,
    pub(crate) device: D,
}

pub trait TraitStorage<T, D>
where
    T: Clone,
    D: Device + DeviceWithDType<T>,
{
    fn device(&self) -> D;
    fn to_rawvec(&self) -> D::RawVec;
    fn into_rawvec(self) -> D::RawVec;
    fn new(vector: D::RawVec, device: D) -> Self;
}

pub trait TraitDeviceToStorage<T, D>
where
    T: Clone,
    D: Device + DeviceWithDType<T>,
{
    fn zeros_impl(&self, len: usize) -> Result<Storage<T, D>>;
    fn ones_impl(&self, len: usize) -> Result<Storage<T, D>>;
    fn arange_impl(&self, len: usize) -> Result<Storage<T, D>>;
    unsafe fn empty_impl(&self, len: usize) -> Result<Storage<T, D>>;
    fn from_cpu_vec_owned(&self, vec: Vec<T>) -> Result<Storage<T, D>>;
    fn from_cpu_vec(&self, vec: &Vec<T>) -> Result<Storage<T, D>>;
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
