//! The pinned-host device type ([`DeviceCudaPinned`]) and its storage traits.

use crate::pinned::buffer::{copy_vec_into, fill_host, PinnedBuffer};
use crate::prelude_dev::*;
use crate::registry::{ordinal_obj, CudaOrdinalObj};
use core::mem::MaybeUninit;
use cudarc::driver::{CudaContext, ValidAsZeroBits};
use std::sync::Arc;

/// Device of page-locked host tensors (`Raw = PinnedBuffer<T>`).
///
/// A device value carries the CUDA ordinal whose context pins the pages and a
/// `DeviceCpuSerial` base (default order). It has no stream and no runtime
/// flags: transfers run on the `DeviceCuda` value passed to `to_device`, and
/// host access goes through `host_view`/`host_view_mut`, which hand out
/// ordinary `DeviceCpuSerial` views over the same bytes.
///
/// No operation traits are implemented for this device; it is a transfer
/// endpoint and a host-compute source through views only. `same_device`
/// compares the ordinal and the base device.
#[derive(Clone, Debug)]
pub struct DeviceCudaPinned {
    ordinal: Arc<CudaOrdinalObj>,
    base: DeviceCpuSerial,
}

impl DeviceCudaPinned {
    /// Build (or fetch from the registry) the per-ordinal object.
    pub fn new(ordinal: usize) -> Self {
        Self::new_f(ordinal).rstsr_unwrap()
    }

    /// Fallible form of [`DeviceCudaPinned::new`].
    pub fn new_f(ordinal: usize) -> Result<Self> {
        Ok(Self { ordinal: ordinal_obj(ordinal)?, base: DeviceCpuSerial::default() })
    }

    /// The CUDA ordinal whose context pins the pages.
    pub fn ordinal(&self) -> usize {
        self.ordinal.ordinal()
    }

    /// The shared per-ordinal object (context, default stream).
    pub fn ordinal_obj(&self) -> &Arc<CudaOrdinalObj> {
        &self.ordinal
    }

    /// The context the pinned pages belong to.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.ordinal.context()
    }

    /// The CPU base device the host views are created on.
    pub fn base(&self) -> &DeviceCpuSerial {
        &self.base
    }

    /// Allocate `len` uninitialized pinned elements (`cudaHostAllocDefault`).
    pub fn alloc_f<T: Copy>(&self, len: usize) -> Result<PinnedBuffer<T>> {
        PinnedBuffer::new_f(self.context(), len)
    }

    /// Allocate `len` zeroed pinned elements (host memset).
    pub fn zeros_f<T: Copy + ValidAsZeroBits>(&self, len: usize) -> Result<PinnedBuffer<T>> {
        let mut buf = self.alloc_f::<T>(len)?;
        buf.uninit_bytes_mut().fill(MaybeUninit::new(0));
        Ok(buf)
    }

    /// Allocate `len` pinned elements set to `value` (host fill).
    pub fn full_f<T: Copy>(&self, len: usize, value: T) -> Result<PinnedBuffer<T>> {
        let mut buf = self.alloc_f::<T>(len)?;
        fill_host(&mut buf, &value);
        Ok(buf)
    }

    /// Copy a host `Vec` into a new pinned buffer (host memcpy).
    pub fn from_vec_f<T: Copy>(&self, vec: Vec<T>) -> Result<PinnedBuffer<T>> {
        let mut buf = self.alloc_f::<T>(vec.len())?;
        copy_vec_into(&mut buf, &vec)?;
        Ok(buf)
    }

    /// Wrap a pinned buffer with the row/column-major layout of `shape`.
    pub fn tensor_f<T, D>(&self, buf: PinnedBuffer<T>, shape: D) -> Result<Tensor<T, Self, IxD>>
    where
        T: Copy + Send + Sync + 'static,
        D: DimAPI,
    {
        let layout = match self.default_order() {
            RowMajor => shape.c(),
            ColMajor => shape.f(),
        };
        let storage = Storage::new(DataOwned::from(buf), self.clone());
        let tensor = TensorAny::new_f(storage, layout.into_dim()?)?;
        Ok(tensor)
    }
}

impl Default for DeviceCudaPinned {
    /// Ordinal 0. Panics through the error mapping on CUDA-less machines.
    fn default() -> Self {
        DeviceCudaPinned::new(0)
    }
}

impl DeviceBaseAPI for DeviceCudaPinned {
    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ordinal, &other.ordinal) && self.base.same_device(&other.base)
    }

    fn default_order(&self) -> FlagOrder {
        self.base.default_order()
    }

    fn set_default_order(&mut self, order: FlagOrder) {
        self.base.set_default_order(order);
    }
}

impl<T> DeviceRawAPI<T> for DeviceCudaPinned {
    type Raw = PinnedBuffer<T>;
}

// `T: Copy`: the bytes are reinterpreted as `T` on the host.
impl<T: Copy> DeviceStorageAPI<T> for DeviceCudaPinned {
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().len()
    }

    fn get_index<R>(storage: &Storage<R, T, Self>, index: usize) -> T
    where
        T: Clone,
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().as_slice()[index]
    }

    fn get_index_ptr<R>(storage: &Storage<R, T, Self>, index: usize) -> *const T
    where
        R: DataAPI<Data = Self::Raw>,
    {
        // Host pointer, valid after the sync inside `as_slice`.
        let raw = storage.raw();
        assert!(index < raw.len(), "get_index_ptr out of bounds");
        // SAFETY: `index < len` was asserted.
        unsafe { raw.as_slice().as_ptr().add(index) }
    }

    fn get_index_mut_ptr<R>(storage: &mut Storage<R, T, Self>, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = Self::Raw>,
    {
        let raw = storage.raw_mut();
        assert!(index < raw.len(), "get_index_mut_ptr out of bounds");
        // SAFETY: `index < len` was asserted.
        unsafe { raw.as_mut_slice().as_mut_ptr().add(index) }
    }

    fn set_index<R>(storage: &mut Storage<R, T, Self>, index: usize, value: T)
    where
        R: DataMutAPI<Data = Self::Raw>,
    {
        let raw = storage.raw_mut();
        assert!(index < raw.len(), "set_index out of bounds");
        raw.as_mut_slice()[index] = value;
    }
}

impl<T: Copy> DeviceToCpuAPI<T> for DeviceCudaPinned {
    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>,
    {
        storage.raw().to_vec_f()
    }

    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataCloneAPI<Data = Self::Raw>,
    {
        storage.raw().to_vec_f()
    }
}

impl<T: Copy> DeviceAPI<T> for DeviceCudaPinned {}

/// Display prints `cuda-pinned:<ordinal>`; `Debug` is the detailed form.
impl core::fmt::Display for DeviceCudaPinned {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cuda-pinned:{}", self.ordinal.ordinal())
    }
}
