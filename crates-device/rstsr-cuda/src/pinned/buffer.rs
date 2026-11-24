//! Pinned (page-locked) host buffer; `DeviceRawAPI::Raw` for `DeviceCudaPinned`.

use crate::error::{map_driver_error, map_driver_result, map_driver_result_alloc};
use crate::event::CudaEvent;
use crate::prelude_dev::*;
use core::mem::MaybeUninit;
use cudarc::driver::{result as dr, sys, CudaContext, CudaStream};
use std::sync::Arc;

/// `cuMemHostAlloc` flags accepted by [`PinnedBuffer::with_flags_f`].
pub mod pinned_flags {
    /// `cudaHostAllocDefault`: cached, pageable-locked pages (the default).
    pub const DEFAULT: u32 = 0;
    /// `cudaHostAllocPortable`: usable by every context of the process.
    pub const PORTABLE: u32 = cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE;
    /// `cudaHostAllocMapped`: also mapped into the device address space.
    pub const MAPPED: u32 = cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP;
    /// `cudaHostAllocWriteCombined`: fast device writes, uncached host reads.
    pub const WRITE_COMBINED: u32 = cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
}

/// Owning page-locked host buffer of `len` elements of `T`.
///
/// - Allocation is byte-based (`cuMemHostAlloc`); the `Copy` restriction is enforced where bytes
///   are read or written as `T`, not here. Layout is identical for `PinnedBuffer<MaybeUninit<T>>`,
///   so the tensor layer's `Raw<T> <-> Raw<MaybeUninit<T>>` transmutes are sound.
/// - **One `CudaEvent` per buffer** orders every asynchronous copy against the host: a copy makes
///   its stream [`wait`](PinnedBuffer::wait) for the event first and
///   [`record`](PinnedBuffer::record) it afterwards; every host access synchronizes the event
///   before touching the bytes. Sequential use of one buffer across several streams is therefore
///   safe; *simultaneous* use by two streams is not (it is aliasing anyway) and is not detected.
/// - `Drop` synchronizes the event, then frees (`cuMemFreeHost`), so a buffer is never released
///   while a DMA still targets it.
/// - `Clone` is a deep host memcpy (value semantics, as `CudaBuffer`/`Vec`).
pub struct PinnedBuffer<T> {
    ptr: *mut u8,
    len: usize,
    byte_len: usize,
    event: CudaEvent,
    _m: PhantomData<*const T>,
}

impl<T> PinnedBuffer<T> {
    /// Allocate `len` uninitialized elements with `cudaHostAllocDefault`.
    pub fn new_f(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        Self::with_flags_f(ctx, len, pinned_flags::DEFAULT)
    }

    /// Allocate `len` uninitialized elements with explicit `cuMemHostAlloc` flags (see
    /// [`pinned_flags`]).
    ///
    /// # Safety
    ///
    /// The caller must initialize every element before it is read (as a `T` or by a copy).
    pub fn with_flags_f(ctx: &Arc<CudaContext>, len: usize, flags: u32) -> Result<Self> {
        let byte_len = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| rstsr_error!(MemoryError, "pinned buffer length overflow: {len} elements"))?;
        ctx.bind_to_thread().map_err(map_driver_error)?;
        // BLOCKING_SYNC: a waiting host thread sleeps instead of spinning; the wake-up latency is
        // negligible against a chunk-sized copy.
        let event = ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC)).map_err(map_driver_error)?;
        // Empty buffers keep a T-aligned dangling pointer (a null pointer is not a valid slice
        // base).
        let ptr = if byte_len == 0 {
            std::ptr::null_mut::<u8>().wrapping_add(core::mem::align_of::<T>().max(1))
        } else {
            // SAFETY: `byte_len` bytes are requested; the buffer is uninitialized by contract.
            map_driver_result_alloc(unsafe { dr::malloc_host(byte_len, flags) }, byte_len)? as *mut u8
        };
        Ok(Self { ptr, len, byte_len, event: CudaEvent::new(event), _m: PhantomData })
    }

    /// Logical element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical length is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// The context the pages were pinned in.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.event.context()
    }

    /// The ordering event of this buffer (see the type docs).
    pub fn event(&self) -> &CudaEvent {
        &self.event
    }

    /// Make `stream` wait for every copy previously recorded on this buffer.
    pub fn wait(&self, stream: &CudaStream) -> Result<()> {
        map_driver_result(stream.wait(self.event.raw()))
    }

    /// Record this buffer's event on `stream` (after a copy was enqueued there).
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        map_driver_result(self.event.raw().record(stream))
    }

    /// Wait until every copy recorded on this buffer has completed.
    pub fn sync_f(&self) -> Result<()> {
        self.event.synchronize_f()
    }

    /// Panicking form of [`PinnedBuffer::sync_f`].
    pub fn sync(&self) {
        self.sync_f().rstsr_unwrap()
    }

    /// Base host pointer; **not synchronized** (internal transfer path).
    pub(crate) fn ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Synchronize, then borrow the initialized elements as a slice.
    pub fn as_slice_f(&self) -> Result<&[T]>
    where
        T: Copy,
    {
        self.sync_f()?;
        // SAFETY: `len` initialized `T`s (POD) start at `ptr`; no DMA is in flight after the sync.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr(), self.len) })
    }

    /// Synchronize, then borrow the initialized elements mutably.
    pub fn as_mut_slice_f(&mut self) -> Result<&mut [T]>
    where
        T: Copy,
    {
        self.sync_f()?;
        // SAFETY: as `as_slice_f`, with the exclusive borrow providing write access.
        Ok(unsafe { core::slice::from_raw_parts_mut(self.ptr(), self.len) })
    }

    /// Panicking form of [`PinnedBuffer::as_slice_f`].
    pub fn as_slice(&self) -> &[T]
    where
        T: Copy,
    {
        self.as_slice_f().rstsr_unwrap()
    }

    /// Panicking form of [`PinnedBuffer::as_mut_slice_f`].
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        T: Copy,
    {
        self.as_mut_slice_f().rstsr_unwrap()
    }

    /// Copy the (synchronized) contents into a `Vec`.
    pub fn to_vec_f(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        Ok(self.as_slice_f()?.to_vec())
    }

    /// Copy the (synchronized) contents into a `Vec`, then free the pages.
    pub fn into_vec_f(self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.to_vec_f()
    }

    /// The storage as uninitialized bytes (copy source or destination). No
    /// reference to initialized data is formed, so this is valid for a fresh
    /// `alloc` as well.
    pub(crate) fn uninit_bytes(&self) -> &[MaybeUninit<u8>] {
        // SAFETY: `byte_len` bytes of pinned storage; `MaybeUninit<u8>` imposes no validity.
        unsafe { core::slice::from_raw_parts(self.ptr as *const MaybeUninit<u8>, self.byte_len) }
    }

    /// Mutable uninitialized bytes (copy destination).
    pub(crate) fn uninit_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: as `uninit_bytes`, with the exclusive borrow providing write access.
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut MaybeUninit<u8>, self.byte_len) }
    }
}

impl<T: Copy> Clone for PinnedBuffer<T> {
    /// Deep host memcpy (value semantics). Panics through the error mapping on failure.
    fn clone(&self) -> Self {
        self.clone_f().rstsr_unwrap()
    }
}

impl<T: Copy> PinnedBuffer<T> {
    /// Fallible deep copy.
    pub fn clone_f(&self) -> Result<Self> {
        let out = Self::new_f(self.context(), self.len)?;
        let src = self.as_slice_f()?;
        // SAFETY: `src` and the destination are non-overlapping; the destination is written
        // before it is read.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), out.ptr(), self.len) };
        Ok(out)
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // A pending DMA must finish before the pages go back to the driver.
        self.event.context().record_err(self.event.raw().synchronize());
        if self.byte_len > 0 {
            self.event.context().record_err(unsafe { dr::free_host(self.ptr as _) });
        }
    }
}

impl<T> core::fmt::Debug for PinnedBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedBuffer").field("len", &self.len).field("ptr", &(self.ptr as usize)).finish()
    }
}

// # Safety
//
// The bytes are host memory governed by ordinary borrows; the event makes every
// DMA complete before a host access or a free. `T: Send`/`Sync` keeps the usual
// element rules (no `&T` across threads for non-`Sync` elements).
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

/// Fill every element of `dst` with `value` (raw writes; valid on a fresh buffer).
pub(crate) fn fill_host<T: Copy>(dst: &mut PinnedBuffer<T>, value: &T) {
    let ptr = dst.ptr();
    for i in 0..dst.len() {
        // SAFETY: element `i` is inside the allocation; `write` does not read the old value.
        unsafe { ptr.add(i).write(*value) };
    }
}

/// Copy `src` (host `Vec`) into `dst` element-for-element (host memcpy).
pub(crate) fn copy_vec_into<T: Copy>(dst: &mut PinnedBuffer<T>, src: &[T]) -> Result<()> {
    rstsr_assert_eq!(dst.len(), src.len(), InvalidValue, "pinned buffer length mismatch")?;
    if !src.is_empty() {
        // SAFETY: non-overlapping; `dst` is exclusively borrowed.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst.ptr(), src.len()) };
    }
    Ok(())
}
