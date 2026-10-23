//! Host <-> device transfers.
//!
//! Copies use the raw driver API on byte views, keeping the storage layer
//! dtype-agnostic. All transfers synchronize before host data is touched;
//! the async API surface supersedes that.

use crate::buffer::CudaBuffer;
use crate::error::{map_driver_result, map_driver_result_alloc};
use crate::prelude_dev::*;
use cudarc::driver::result;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// Base device address of `raw` as a raw `CUdeviceptr` (byte view).
pub(crate) fn base_ptr<T>(raw: &CudaBuffer<T>) -> CUdeviceptr {
    let (ptr, _record) = DevicePtr::<u8>::device_ptr(&raw.slice, raw.slice.stream());
    ptr
}

/// Mutable base device address of `raw` as a raw `CUdeviceptr` (byte view).
pub(crate) fn base_ptr_mut<T>(raw: &mut CudaBuffer<T>) -> CUdeviceptr {
    let stream = raw.slice.stream().clone();
    let (ptr, _record) = DevicePtrMut::<u8>::device_ptr_mut(&mut raw.slice, &stream);
    ptr
}

/// D2H of one element (used by `get_index` / `to_scalar`). Panics through
/// the error mapping on driver failure; infallible in the trait.
// DEFER: batched/fallible variant; per-element sync is debug-grade.
pub(crate) fn d2h_scalar<T: Copy>(raw: &CudaBuffer<T>, index: usize) -> T {
    let esz = core::mem::size_of::<T>();
    let stream = raw.slice.stream();
    let mut host = vec![0u8; esz];
    unsafe {
        map_driver_result(result::memcpy_dtoh_async(
            &mut host,
            base_ptr(raw) + (index * esz) as u64,
            stream.cu_stream(),
        ))
        .rstsr_unwrap();
        map_driver_result(stream.synchronize()).rstsr_unwrap();
    }
    rstsr_assert_eq!(host.len(), esz, InvalidValue).rstsr_unwrap();
    unsafe { core::ptr::read(host.as_ptr() as *const T) }
}

/// H2D of one element (used by `set_index`).
// DEFER: batched variant; per-element sync is debug-grade.
pub(crate) fn h2d_scalar<T: Copy>(raw: &mut CudaBuffer<T>, index: usize, value: T) {
    let esz = core::mem::size_of::<T>();
    // stage the value's bytes on the host (type-erased copy)
    let mut host = vec![0u8; esz];
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(value) as *const u8, host.as_mut_ptr(), esz);
    }
    let stream = raw.slice.stream().clone();
    let ptr = base_ptr_mut(raw);
    unsafe {
        map_driver_result(result::memcpy_htod_async(ptr + (index * esz) as u64, &host, stream.cu_stream()))
            .rstsr_unwrap();
        map_driver_result(stream.synchronize()).rstsr_unwrap();
    }
}

/// Full-buffer D2H into a host `Vec` (blocking; `to_cpu_vec` path).
pub(crate) fn d2h_full<T: Copy>(raw: &CudaBuffer<T>) -> Result<Vec<T>> {
    let esz = core::mem::size_of::<T>();
    let byte_len = raw.len * esz;
    let mut bytes = vec![0u8; byte_len];
    if byte_len > 0 {
        let stream = raw.slice.stream();
        unsafe {
            map_driver_result(result::memcpy_dtoh_async(&mut bytes, base_ptr(raw), stream.cu_stream()))?;
            map_driver_result(stream.synchronize())?;
        }
    }
    // reinterpret the staged bytes as `T` (POD dtypes only reach storage)
    let mut vec: Vec<T> = Vec::with_capacity(raw.len);
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, vec.as_mut_ptr(), raw.len);
        vec.set_len(raw.len);
    }
    Ok(vec)
}

/// Full-buffer H2D from a host `Vec` (blocking; `outof_cpu_vec` path).
pub(crate) fn h2d_full<T: Copy>(device: &DeviceCuda, vec: Vec<T>) -> Result<CudaBuffer<T>> {
    let len = vec.len();
    let esz = core::mem::size_of::<T>();
    let byte_len = len * esz;
    let mut buf = CudaBuffer::<T>::new(device.stream(), len)?;
    if byte_len > 0 {
        let bytes = unsafe { core::slice::from_raw_parts(vec.as_ptr() as *const u8, byte_len) };
        let stream = buf.slice.stream().clone();
        let ptr = base_ptr_mut(&mut buf);
        unsafe {
            map_driver_result_alloc(result::memcpy_htod_async(ptr, bytes, stream.cu_stream()), byte_len)?;
            map_driver_result(stream.synchronize())?;
        }
    }
    Ok(buf)
}

/// Device-to-device full-buffer copy into an already-allocated destination.
pub(crate) fn dtod_full<T>(src: &CudaBuffer<T>, dst: &mut CudaBuffer<T>) -> Result<()> {
    rstsr_assert_eq!(src.len, dst.len, InvalidValue)?;
    let byte_len = src.len * core::mem::size_of::<T>();
    if byte_len == 0 {
        return Ok(());
    }
    let sptr = base_ptr(src);
    let dptr = base_ptr_mut(dst);
    let stream = dst.slice.stream();
    // the copy runs on the destination's stream; order the source's pending
    // work into it
    let _guard = crate::kernel::order_input_raw(src, stream)?;
    unsafe {
        map_driver_result(result::memcpy_dtod_async(dptr, sptr, byte_len, stream.cu_stream()))?;
        map_driver_result(stream.synchronize())?;
    }
    Ok(())
}

/// Zero-fill the whole buffer on its allocation stream (`zeros` fast path).
pub(crate) fn memset_zero<T>(raw: &mut CudaBuffer<T>) -> Result<()> {
    let byte_len = raw.len * core::mem::size_of::<T>();
    if byte_len == 0 {
        return Ok(());
    }
    let ptr = base_ptr_mut(raw);
    let stream = raw.slice.stream();
    unsafe {
        map_driver_result(result::memset_d8_async(ptr, 0, byte_len, stream.cu_stream()))?;
    }
    Ok(())
}

/// Fill the whole buffer by repeating a host value. Upload one element, then
/// geometric dtod doubling.
///
/// DEFER: device fill kernel.
pub(crate) fn fill_naive<T: Copy>(raw: &mut CudaBuffer<T>, fill: &T) -> Result<()> {
    let n = raw.len;
    if n == 0 {
        return Ok(());
    }
    h2d_scalar(raw, 0, *fill);
    let esz = core::mem::size_of::<T>();
    let ptr = base_ptr_mut(raw);
    let stream = raw.slice.stream();
    let mut done = 1usize;
    while done < n {
        let count = done.min(n - done);
        unsafe {
            map_driver_result(result::memcpy_dtod_async(
                ptr + (done * esz) as u64,
                ptr,
                count * esz,
                stream.cu_stream(),
            ))?;
        }
        done += count;
    }
    map_driver_result(stream.synchronize())
}
