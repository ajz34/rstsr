//! Non-blocking transfers between pinned host buffers and device buffers.

use crate::buffer::CudaBuffer;
use crate::error::{map_driver_result, map_driver_result_alloc};
use crate::pinned::PinnedBuffer;
use crate::prelude_dev::*;
use crate::transfer::base_ptr_mut;
use cudarc::driver::{result, DevicePtr};
/// H2D from a pinned buffer, **non-blocking**: enqueued on `device`'s
/// execution stream; the buffer's event orders it against every other use.
/// The returned device buffer is ready when the stream reaches the copy.
pub fn h2d_pinned_async<T: Copy>(device: &DeviceCuda, src: &PinnedBuffer<T>) -> Result<CudaBuffer<T>> {
    let byte_len = src.byte_len();
    let mut buf = CudaBuffer::<T>::new_f(device.stream(), src.len())?;
    if byte_len > 0 {
        let stream = device.stream().inner();
        src.wait(stream)?;
        let ptr = base_ptr_mut(&mut buf);
        // SAFETY: `ptr` owns `byte_len` device bytes; `src` is an initialized pinned buffer.
        unsafe {
            map_driver_result_alloc(result::memcpy_htod_async(ptr, src.uninit_bytes(), stream.cu_stream()), byte_len)?;
        }
        src.record(stream)?;
    }
    Ok(buf)
}

/// D2H into a pinned buffer, **non-blocking**: enqueued on `device`'s
/// execution stream; the destination's event orders it against every later
/// host access. The source is ordered alloc-stream -> exec (pending
/// allocation/kernels), and the DMA is recorded as a *read* of the source
/// so that the source's eventual free (on its allocation stream) waits for
/// it — at drop time, not now: making the allocation stream wait here would
/// stall every later kernel on it (measured: it serializes compute behind
/// D2H).
pub fn d2h_pinned_async<T: Copy>(device: &DeviceCuda, src: &CudaBuffer<T>, dst: &mut PinnedBuffer<T>) -> Result<()> {
    rstsr_assert_eq!(src.len(), dst.len(), InvalidValue, "pinned destination length mismatch")?;
    let byte_len = dst.byte_len();
    if byte_len > 0 {
        let stream = device.stream().inner();
        dst.wait(stream)?;
        let _input = crate::kernel::order_input_raw(src, stream)?;
        // cudarc records a read event on `stream` when `_read` drops; `CudaSlice::drop`
        // makes the allocation stream wait for it before the stream-ordered free.
        let (sptr, _read) = DevicePtr::<u8>::device_ptr(src.as_bytes(), stream);
        // SAFETY: `dst` owns `byte_len` writable pinned bytes.
        unsafe {
            map_driver_result(result::memcpy_dtoh_async(dst.uninit_bytes_mut(), sptr, stream.cu_stream()))?;
        }
        drop(_read);
        drop(_input);
        dst.record(stream)?;
    }
    Ok(())
}
