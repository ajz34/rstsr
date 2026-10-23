//! The CUDA device type.

use crate::buffer::CudaBuffer;
use crate::flags::{self, ComputeType32F, CudaRuntimeFlags};
use crate::registry::{ordinal_obj, CudaOrdinalObj};
use crate::stream::CudaStreamObj;
use rstsr_core::prelude_dev::*;
use std::sync::Arc;

/// The CUDA device type.
///
/// A device value carries:
/// - `ordinal`: the registry-canonical per-GPU identity (context, default stream, later BLAS
///   handles / kernel cache / allocator config);
/// - `stream`: the execution stream this device value runs ops on;
/// - `flags`: an optional runtime-flags override; `None` tracks the process-global default;
/// - `default_order`: the row/column-major default snapshot.
///
/// Cloning is cheap: heavy fields are Arc handles, the rest are small `Copy`
/// values. Every tensor's `Storage` carries its own device value (framework
/// convention).
///
/// `same_device` compares:
/// - ordinal identity (Arc ptr-eq);
/// - the execution stream;
/// - flags **as stored** (`None` vs `Some(x)` differ even when the global is `x`);
/// - `default_order` (the CPU-device precedent).
///
/// Flags are compared as stored, never resolved.
#[derive(Clone, Debug)]
pub struct DeviceCuda {
    pub(crate) ordinal: Arc<CudaOrdinalObj>,
    pub(crate) stream: Arc<CudaStreamObj>,
    pub(crate) flags: Option<CudaRuntimeFlags>,
    pub(crate) default_order: FlagOrder,
}

impl DeviceCuda {
    /// Build (or fetch from the registry) the canonical per-ordinal object
    /// and return a device on its default (non-blocking) stream.
    pub fn new(ordinal: usize) -> Self {
        Self::new_f(ordinal).rstsr_unwrap()
    }

    /// Fallible form of [`DeviceCuda::new`].
    pub fn new_f(ordinal: usize) -> Result<Self> {
        let ordinal_obj = ordinal_obj(ordinal)?;
        let stream = ordinal_obj.default_stream.clone();
        Ok(Self { ordinal: ordinal_obj, stream, flags: None, default_order: FlagOrder::default() })
    }

    /// The CUDA ordinal of this device.
    pub fn ordinal(&self) -> usize {
        self.ordinal.ordinal()
    }

    /// The shared per-ordinal object (context, default stream, later BLAS
    /// handles / kernel cache / allocator config).
    pub fn ordinal_obj(&self) -> &Arc<CudaOrdinalObj> {
        &self.ordinal
    }

    /// The execution stream this device value runs ops on.
    pub fn stream(&self) -> &Arc<CudaStreamObj> {
        &self.stream
    }

    /// Switch the execution stream (mutating setter, like
    /// `set_default_order`). Cross-stream producer/consumer stays explicit:
    /// clone the device, `set_stream`, then re-tag entry tensors via
    /// `to_device`.
    ///
    /// # Panics
    ///
    /// Panics if `stream` belongs to a different ordinal. Use
    /// [`DeviceCuda::set_stream_f`] for the fallible form.
    pub fn set_stream(&mut self, stream: Arc<CudaStreamObj>) {
        self.set_stream_f(stream).rstsr_unwrap()
    }

    /// Fallible form of [`DeviceCuda::set_stream`]: same conditions, error
    /// instead of panic.
    pub fn set_stream_f(&mut self, stream: Arc<CudaStreamObj>) -> Result<()> {
        rstsr_assert_eq!(stream.ordinal(), self.ordinal(), DeviceMismatch, "stream belongs to a different ordinal")?;
        self.stream = stream;
        Ok(())
    }

    /// Resolve flags at launch time: override if pinned, else the global.
    pub fn flags(&self) -> CudaRuntimeFlags {
        self.flags.unwrap_or_else(flags::default_flags)
    }

    /// The stored flags override (comparison uses this, not [`Self::flags`]).
    pub fn flags_override(&self) -> Option<CudaRuntimeFlags> {
        self.flags
    }

    /// Pin (`Some`) or unpin (`None`) the flags of this device value.
    pub fn set_flags_override(&mut self, flags: Option<CudaRuntimeFlags>) {
        self.flags = flags;
    }

    /// Shortcut accessor for the common flag.
    pub fn compute_type_32f(&self) -> ComputeType32F {
        self.flags().compute_type_32f
    }

    /// The ordinal's default stream.
    pub fn default_stream(&self) -> Arc<CudaStreamObj> {
        self.ordinal.default_stream.clone()
    }

    /// Block the host until this device's execution stream completes.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Compute capability `(major, minor)` of this device.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        self.ordinal.compute_capability()
    }

    /// `(free, total)` device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.ordinal.memory_info()
    }
}

impl Default for DeviceCuda {
    /// Ordinal 0. Panics through the error mapping on CUDA-less machines —
    /// a device the program cannot use should fail loudly.
    fn default() -> Self {
        DeviceCuda::new(0)
    }
}

impl DeviceBaseAPI for DeviceCuda {
    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ordinal, &other.ordinal)
            && self.stream == other.stream
            && self.flags == other.flags
            && self.default_order == other.default_order
    }

    fn default_order(&self) -> FlagOrder {
        self.default_order
    }

    fn set_default_order(&mut self, order: FlagOrder) {
        self.default_order = order;
    }
}

impl<T> DeviceRawAPI<T> for DeviceCuda {
    type Raw = CudaBuffer<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceCuda {
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
        // Single-element D2H on the allocation stream + sync; the trait is
        // infallible, so errors panic through the mapping.
        // DEFER: batched/fallible variant; per-element sync is debug-grade.
        let raw = storage.raw();
        assert!(index < raw.len(), "get_index out of bounds");
        crate::transfer::d2h_scalar(raw, index)
    }

    fn get_index_ptr<R>(storage: &Storage<R, T, Self>, index: usize) -> *const T
    where
        R: DataAPI<Data = Self::Raw>,
    {
        // Device address; never host-dereferenceable.
        let base = storage.raw().device_ptr();
        assert!(index < storage.raw().len(), "get_index_ptr out of bounds");
        unsafe { base.add(index) }
    }

    fn get_index_mut_ptr<R>(storage: &mut Storage<R, T, Self>, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = Self::Raw>,
    {
        let base = storage.raw_mut().device_ptr_mut();
        assert!(index < storage.raw().len(), "get_index_mut_ptr out of bounds");
        unsafe { base.add(index) }
    }

    fn set_index<R>(storage: &mut Storage<R, T, Self>, index: usize, value: T)
    where
        R: DataMutAPI<Data = Self::Raw>,
    {
        // DEFER: batched variant; per-element sync is debug-grade.
        assert!(index < storage.raw().len(), "set_index out of bounds");
        crate::transfer::h2d_scalar(storage.raw_mut(), index, value);
    }
}

impl<T> DeviceAPI<T> for DeviceCuda where T: Clone + Send + Sync + 'static {}

/// Display prints `cuda:<ordinal>`; `Debug` is the detailed form.
impl core::fmt::Display for DeviceCuda {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cuda:{}", self.ordinal.ordinal())
    }
}
