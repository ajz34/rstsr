//! Non-blocking transfers between pinned host buffers and device buffers.

use crate::buffer::CudaBuffer;
use crate::error::{map_driver_result, map_driver_result_alloc};
use crate::pinned::PinnedBuffer;
use crate::prelude_dev::*;
use crate::transfer::{base_ptr, base_ptr_mut};
use cudarc::driver::result;
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
/// host access. The source's allocation stream is ordered both ways (a
/// foreign-stream temporary is not reused while the DMA reads it).
pub fn d2h_pinned_async<T: Copy>(device: &DeviceCuda, src: &CudaBuffer<T>, dst: &mut PinnedBuffer<T>) -> Result<()> {
    rstsr_assert_eq!(src.len(), dst.len(), InvalidValue, "pinned destination length mismatch")?;
    let byte_len = dst.byte_len();
    if byte_len > 0 {
        let stream = device.stream().inner();
        dst.wait(stream)?;
        let _input = crate::kernel::order_input_raw(src, stream)?;
        // SAFETY: `dst` owns `byte_len` writable pinned bytes.
        unsafe {
            map_driver_result(result::memcpy_dtoh_async(dst.uninit_bytes_mut(), base_ptr(src), stream.cu_stream()))?;
        }
        drop(_input);
        let _output = crate::kernel::order_output_raw(src, stream)?;
        drop(_output);
        dst.record(stream)?;
    }
    Ok(())
}
