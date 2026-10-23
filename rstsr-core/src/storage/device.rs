use crate::prelude_dev::*;

pub trait DeviceBaseAPI {
    fn same_device(&self, other: &Self) -> bool;
    fn default_order(&self) -> FlagOrder;
    fn set_default_order(&mut self, order: FlagOrder);
}

pub trait DeviceRawAPI<T>: DeviceBaseAPI + Clone {
    type Raw;
}

#[derive(Debug)]
pub struct Storage<R, T, B>
where
    B: DeviceRawAPI<T>,
{
    pub(crate) data: R,
    pub(crate) device: B,
    _phantom: PhantomData<T>,
}

pub trait DeviceStorageAPI<T>: DeviceRawAPI<T> {
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>;
    fn is_empty<R>(storage: &Storage<R, T, Self>) -> bool
    where
        R: DataAPI<Data = Self::Raw>,
    {
        Self::len::<R>(storage) == 0
    }
    /// Copy the buffer to a host `Vec<T>`.
    ///
    /// Requires the device to implement [`DeviceToCpuAPI`].
    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self: DeviceToCpuAPI<T>,
        R: DataAPI<Data = Self::Raw>,
    {
        <Self as DeviceToCpuAPI<T>>::to_cpu_vec(storage)
    }
    /// Convert the storage into a host `Vec<T>`.
    ///
    /// Requires the device to implement [`DeviceToCpuAPI`].
    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        Self: DeviceToCpuAPI<T>,
        R: DataCloneAPI<Data = Self::Raw>,
    {
        <Self as DeviceToCpuAPI<T>>::into_cpu_vec(storage)
    }
    fn get_index<R>(storage: &Storage<R, T, Self>, index: usize) -> T
    where
        T: Clone,
        R: DataAPI<Data = Self::Raw>;
    /// Address of element `index`.
    ///
    /// The pointer is meaningful only to the backend that owns the storage:
    /// host devices return a host pointer, while device-memory backends
    /// return a device address that must not be dereferenced on the host.
    fn get_index_ptr<R>(storage: &Storage<R, T, Self>, index: usize) -> *const T
    where
        R: DataAPI<Data = Self::Raw>;
    /// Mutable form of [`DeviceStorageAPI::get_index_ptr`]; same pointer
    /// contract applies.
    fn get_index_mut_ptr<R>(storage: &mut Storage<R, T, Self>, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = Self::Raw>;
    fn set_index<R>(storage: &mut Storage<R, T, Self>, index: usize, value: T)
    where
        R: DataMutAPI<Data = Self::Raw>;
}

/// Devices that can copy their buffer contents to a host `Vec<T>`.
///
/// Implementations decide the cost and requirements themselves; e.g. the CPU
/// devices require `T: Clone` because their host copy is a `Vec<T>` clone.
pub trait DeviceToCpuAPI<T>: DeviceStorageAPI<T> {
    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>;
    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataCloneAPI<Data = Self::Raw>;
}

impl<R, T, B> Storage<R, T, B>
where
    B: DeviceStorageAPI<T>,
    R: DataAPI<Data = B::Raw>,
{
    pub fn device(&self) -> &B {
        &self.device
    }

    pub fn device_mut(&mut self) -> &mut B {
        &mut self.device
    }

    pub fn data(&self) -> &R {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut R {
        &mut self.data
    }

    pub fn into_raw_parts(self) -> (R, B) {
        (self.data, self.device)
    }

    pub fn new(data: R, device: B) -> Self {
        Self { data, device, _phantom: PhantomData }
    }

    pub fn len(&self) -> usize {
        B::len(self)
    }

    pub fn is_empty(&self) -> bool {
        B::is_empty(self)
    }

    pub fn to_cpu_vec(&self) -> Result<Vec<T>>
    where
        B: DeviceToCpuAPI<T>,
    {
        <B as DeviceToCpuAPI<T>>::to_cpu_vec(self)
    }

    pub fn into_cpu_vec(self) -> Result<Vec<T>>
    where
        B: DeviceToCpuAPI<T>,
        R: DataCloneAPI<Data = B::Raw>,
    {
        <B as DeviceToCpuAPI<T>>::into_cpu_vec(self)
    }

    #[inline]
    pub fn get_index(&self, index: usize) -> T
    where
        T: Clone,
    {
        B::get_index(self, index)
    }

    #[inline]
    pub fn get_index_ptr(&self, index: usize) -> *const T {
        B::get_index_ptr(self, index)
    }

    #[inline]
    pub fn get_index_mut_ptr(&mut self, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = B::Raw>,
    {
        B::get_index_mut_ptr(self, index)
    }

    #[inline]
    pub fn set_index(&mut self, index: usize, value: T)
    where
        R: DataMutAPI<Data = B::Raw>,
    {
        B::set_index(self, index, value)
    }
}

impl<R, T, B> Storage<R, T, B>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceRawAPI<T>,
{
    pub fn raw(&self) -> &B::Raw {
        self.data.raw()
    }
}

impl<R, T, B> Storage<R, T, B>
where
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceRawAPI<T>,
{
    pub fn raw_mut(&mut self) -> &mut B::Raw {
        self.data.raw_mut()
    }
}

pub trait DeviceAPI<T>: DeviceBaseAPI + DeviceRawAPI<T> + DeviceStorageAPI<T> + Clone + Default {}
