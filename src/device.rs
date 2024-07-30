use crate::Result;

pub trait TraitDevice: Clone {
    fn same_device(&self, other: &Self) -> bool;
}

pub trait TraitStorage {
    type Device: TraitDevice;
    type DType;
    type VType;

    fn device(&self) -> Self::Device;
    fn to_rawvec(&self) -> Self::VType;
    fn into_rawvec(self) -> Self::VType;
    fn new(vector: Self::VType, device: Self::Device) -> Self;
}

pub trait TraitDeviceToStorage<S>
where
    S: TraitStorage,
{
    fn zeros_impl(&self, len: usize) -> Result<S>;
    fn from_cpu_vec_owned(&self, vec: Vec<S::DType>) -> Result<S>;
    fn from_cpu_vec(&self, vec: &Vec<S::DType>) -> Result<S>;
}

/// Unique identifier for cuda devices.
///
/// This code is from `candle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}
