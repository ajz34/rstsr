//! The storage representation shared by import and zero-copy export.

use core::mem::ManuallyDrop;
use std::sync::Arc;

use dlpark::ffi::DLPackVersion;
use dlpark::AllocationDeleter;
use rstsr_common::error::Result;
use rstsr_core::prelude::*;
use rstsr_core::storage::exports::{DataArc, DataForceMutAPI, DataMut, DataOwned, Storage};

/// A tensor buffer **borrowed** through a keep-alive owner.
///
/// `span` is a fabricated `Vec` covering the buffer; it must never be dropped,
/// resized, or reached mutably (the pointer belongs to `owner`, not to the
/// Rust allocator). Implementing `DataAPI`/`DataCloneAPI` only keeps the
/// representation read-only; cloning shares the owner rather than the data.
pub struct DataDlpack<C, O> {
    span: ManuallyDrop<C>,
    owner: O,
}

impl<C, O> DataDlpack<C, O> {
    /// # Safety
    ///
    /// - `span` must be created with `Vec::from_raw_parts` over memory kept alive by `owner` (or by
    ///   whoever keeps `owner` alive), and must never be dropped or resized.
    /// - `span` must cover every index any layout used with this representation can address.
    pub(crate) unsafe fn from_parts(span: ManuallyDrop<C>, owner: O) -> Self {
        Self { span, owner }
    }

    /// The keep-alive owner of the span.
    pub fn owner(&self) -> &O {
        &self.owner
    }
}

impl<C, O> DataAPI for DataDlpack<C, O> {
    type Data = C;

    fn raw(&self) -> &Self::Data {
        &self.span
    }
}

impl<C, O> DataCloneAPI for DataDlpack<C, O>
where
    C: Clone,
{
    /// Deep copy into owned storage (the span is cloned, never re-homed).
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned::from(ManuallyDrop::into_inner(self.span.clone()))
    }

    /// Deep copy into a fresh `Arc`-backed buffer.
    fn into_shared(self) -> DataArc<Self::Data> {
        DataArc::from(ManuallyDrop::into_inner(self.span.clone()))
    }
}

impl<T, O> Clone for DataDlpack<Vec<T>, O>
where
    O: Clone,
{
    /// Clones the owner and rebuilds the span over the same memory (no data copy).
    fn clone(&self) -> Self {
        let ptr = self.span.as_ptr() as *mut T;
        let len = self.span.len();
        // SAFETY: same fabricated-span contract as `from_parts`; the cloned owner keeps it alive.
        let span = unsafe { ManuallyDrop::new(Vec::from_raw_parts(ptr, len, len)) };
        Self { span, owner: self.owner.clone() }
    }
}

// A `DataDlpack` only ever hands out shared access, so it may cross threads
// exactly when the pointee may be shared and the owner may. The foreign owner
// is `Send + Sync` (dlpark's deleter contract makes it callable from any
// thread), so imported tensors are `Send + Sync` too.
unsafe impl<T, O> Send for DataDlpack<Vec<T>, O>
where
    T: Send + Sync,
    O: Send,
{
}
unsafe impl<T, O> Sync for DataDlpack<Vec<T>, O>
where
    T: Send + Sync,
    O: Sync,
{
}

/// Owns a foreign DLPack managed tensor; calls its producer deleter exactly
/// once, when the last Rust handle of the imported tensor dies.
///
/// Ownership is delegated to dlpark's [`AllocationDeleter`]
/// ([`Managed::into_deleter`](dlpark::Managed::into_deleter)); the imported
/// tensor's version and flags are captured up front, so the type-erased deleter
/// is all that survives. A `NULL` producer deleter is a no-op, and the deleter
/// runs on drop from any thread (DLPack's contract).
pub struct DlpackForeignOwner {
    deleter: AllocationDeleter,
    is_versioned: bool,
    version: Option<DLPackVersion>,
    flags: Option<u64>,
}

impl DlpackForeignOwner {
    pub(crate) fn new(
        deleter: AllocationDeleter,
        is_versioned: bool,
        version: Option<DLPackVersion>,
        flags: Option<u64>,
    ) -> Self {
        Self { deleter, is_versioned, version, flags }
    }

    /// The exactly-once producer deleter.
    pub fn deleter(&self) -> &AllocationDeleter {
        &self.deleter
    }

    /// Whether the imported tensor came from a versioned (DLPack >= 1.0) producer.
    pub fn is_versioned(&self) -> bool {
        self.is_versioned
    }

    /// DLPack flags of the imported tensor; `None` for legacy producers.
    pub fn flags(&self) -> Option<u64> {
        self.flags
    }

    /// DLPack version of the imported tensor; `None` for legacy producers.
    pub fn version(&self) -> Option<DLPackVersion> {
        self.version
    }
}

/// Read-only tensor over a foreign DLPack buffer: zero-copy import result.
///
/// Dropping it (the last handle) calls the producer's deleter.
pub type TensorDlpack<T, B = DeviceCpu, D = IxD> = TensorBase<Storage<DataDlpack<Vec<T>, DlpackForeignOwner>, T, B>, D>;

/// Read-only tensor over a shared (`Arc`-owned) buffer: the zero-copy export
/// handle. It can be cloned cheaply and exported repeatedly.
pub type TensorDlpackShared<T, B = DeviceCpu, D = IxD> = TensorBase<Storage<DataDlpack<Vec<T>, Arc<Vec<T>>>, T, B>, D>;

/// Move an owned tensor into the shareable representation (no data copy).
///
/// The buffer is moved into an `Arc`; the returned tensor can be cloned cheaply
/// and exported repeatedly, and the original values stay owned by the caller.
pub fn into_shared_dlpack_f<T, B, D>(tensor: Tensor<T, B, D>) -> Result<TensorDlpackShared<T, B, D>>
where
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();
    let (data, device) = storage.into_raw_parts();
    let buffer = data.into_raw();
    let len = buffer.len();
    let arc = Arc::new(buffer);
    // SAFETY: the `Arc` keeps the buffer alive for as long as any clone of it
    // exists; the span covers the whole buffer, so every valid layout fits.
    let span = unsafe { ManuallyDrop::new(Vec::from_raw_parts(arc.as_ptr() as *mut T, len, len)) };
    let repr = unsafe { DataDlpack::from_parts(span, Arc::clone(&arc)) };
    let storage = Storage::new(repr, device);
    TensorDlpackShared::<T, B, D>::new_f(storage, layout)
}

/// Enables [`TensorAny::force_mut`] on an imported tensor: an `unsafe` escape hatch that
/// writes through to the producer's buffer (the caller guarantees unique access).
impl<T, O> DataForceMutAPI<Vec<T>> for DataDlpack<Vec<T>, O> {
    /// # Safety
    ///
    /// The caller must guarantee unique access to the buffer for the returned handle's
    /// lifetime (see [`DataForceMutAPI::force_mut`]). For an imported tensor the buffer is
    /// the producer's, so a write here mutates the producer's memory in place, zero-copy.
    unsafe fn force_mut(&self) -> DataMut<'_, Vec<T>> {
        let ptr = self.span.as_ptr() as *mut T;
        let len = self.span.len();
        // SAFETY: same fabricated-span contract as `from_parts`: the span covers the owner's
        // buffer, and the `ManuallyDrop` wrapper keeps it from being freed, resized, or grown.
        let vec = unsafe { Vec::from_raw_parts(ptr, len, len) };
        DataMut::ManuallyDropOwned(ManuallyDrop::new(vec))
    }
}
