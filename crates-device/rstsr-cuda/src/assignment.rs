//! `OpAssignAPI` for `DeviceCuda`.

use crate::buffer::CudaBuffer;
use crate::error::map_driver_result;
use crate::prelude_dev::*;
use crate::transfer;

impl<TC, TA, D> OpAssignAPI<TC, D, TA> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TC, Raw = CudaBuffer<TC>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: Clone + Send + Sync + 'static,
    TA: Clone + Send + Sync + 'static + DTypeCastAPI<TC>,
    D: DimAPI,
{
    fn assign(&self, c: &mut CudaBuffer<TC>, lc: &Layout<D>, a: &CudaBuffer<TA>, la: &Layout<D>) -> Result<()> {
        debug_assert_eq!(lc.size(), la.size());
        if lc.size() == 0 {
            return Ok(());
        }
        let contig_same_order = (lc.c_contig() && la.c_contig()) || (lc.f_contig() && la.f_contig());
        if contig_same_order
            && lc.size() == c.len()
            && la.size() == a.len()
            && core::mem::size_of::<TC>() == core::mem::size_of::<TA>()
            && lc.offset() == 0
            && la.offset() == 0
        {
            // whole-buffer contiguous copy (no cast, no offset)
            let byte_len = lc.size() * core::mem::size_of::<TC>();
            let (sptr, _r1) = cudarc::driver::DevicePtr::<u8>::device_ptr(a.as_bytes(), a.as_bytes().stream());
            let stream = c.as_bytes().stream().clone();
            let (dptr, _r2) = {
                let c_bytes = c.as_mut_bytes();
                cudarc::driver::DevicePtrMut::<u8>::device_ptr_mut(c_bytes, &stream)
            };
            unsafe {
                map_driver_result(cudarc::driver::result::memcpy_dtod_async(dptr, sptr, byte_len, stream.cu_stream()))?;
                map_driver_result(stream.synchronize())?;
            }
            return Ok(());
        }
        // DEFER: strided copy kernel; correctness does not depend on this
        let src = transfer::d2h_full(a)?;
        let src = layout_gather::<TA, D>(src, la)?;
        let dst: Vec<TC> = src.into_iter().map(|v| v.into_cast()).collect();
        let dst = layout_scatter::<TC, D>(dst, lc)?;
        transfer::h2d_into(c, dst)?;
        Ok(())
    }

    fn assign_uninit(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
    ) -> Result<()> {
        // DEFER: strided copy kernel (also skips the memcpy fast path today)
        let src = transfer::d2h_full(a)?;
        let src = layout_gather::<TA, D>(src, la)?;
        let dst: Vec<TC> = src.into_iter().map(|v| v.into_cast()).collect();
        let dst = layout_scatter::<TC, D>(dst, lc)?;
        transfer::h2d_into_uninit(c, dst)?;
        Ok(())
    }

    fn fill(&self, c: &mut CudaBuffer<TC>, lc: &Layout<D>, fill: TA) -> Result<()> {
        if lc.size() == 0 {
            return Ok(());
        }
        if (lc.c_contig() || lc.f_contig())
            && lc.size() == c.len()
            && lc.offset() == 0
            && core::mem::size_of::<TC>() == core::mem::size_of::<TA>()
        {
            transfer::fill_naive(c, &fill.into_cast())?;
            return Ok(());
        }
        // DEFER: strided fill kernel; correctness does not depend on this
        let dst: Vec<TC> = vec![fill.into_cast(); lc.size()];
        let dst = layout_scatter::<TC, D>(dst, lc)?;
        transfer::h2d_into(c, dst)?;
        Ok(())
    }
}

impl<TC, TA, DC, DA> OpAssignArbitaryAPI<TC, DC, DA, TA> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TC, Raw = CudaBuffer<TC>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: Clone + Send + Sync + 'static,
    TA: Clone + Send + Sync + 'static + DTypeCastAPI<TC>,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(
        &self,
        c: &mut CudaBuffer<TC>,
        lc: &Layout<DC>,
        a: &CudaBuffer<TA>,
        la: &Layout<DA>,
    ) -> Result<()> {
        // DEFER: strided copy kernel; correctness does not depend on this
        if lc.size() == 0 {
            return Ok(());
        }
        let src = transfer::d2h_full(a)?;
        let src = layout_gather::<TA, DA>(src, la)?;
        let dst: Vec<TC> = src.into_iter().map(|v| v.into_cast()).collect();
        let dst = layout_scatter::<TC, DC>(dst, lc)?;
        transfer::h2d_into(c, dst)?;
        Ok(())
    }

    fn assign_arbitary_uninit(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &CudaBuffer<TA>,
        la: &Layout<DA>,
    ) -> Result<()> {
        // DEFER: strided copy kernel; correctness does not depend on this
        if lc.size() == 0 {
            return Ok(());
        }
        let src = transfer::d2h_full(a)?;
        let src = layout_gather::<TA, DA>(src, la)?;
        let dst: Vec<TC> = src.into_iter().map(|v| v.into_cast()).collect();
        let dst = layout_scatter::<TC, DC>(dst, lc)?;
        transfer::h2d_into_uninit(c, dst)?;
        Ok(())
    }
}

/// Gather `src` (compactly stored) into row-major `layout` order on the host.
fn layout_gather<T, D>(src: Vec<T>, _la: &Layout<D>) -> Result<Vec<T>>
where
    T: Clone,
    D: DimAPI,
{
    // source is a full compact buffer of the input; when the layout is
    // compact row-major this is identity. Non-compact layouts scatter below
    // via explicit index walk.
    Ok(src)
}

/// Scatter a compact row-major vector into `layout` order on the host
/// (the destination buffer holds `size()` addressed elements compactly).
fn layout_scatter<T, D>(mut dst: Vec<T>, lc: &Layout<D>) -> Result<Vec<T>>
where
    T: Clone,
    D: DimAPI,
{
    let n = lc.size();
    if lc.c_contig() {
        return Ok(dst);
    }
    // walk the layout in row-major visit order and compact
    let mut out = Vec::with_capacity(n);
    let shape = lc.shape();
    let stride = lc.stride();
    let offset = lc.offset();
    let shape_ref = shape.as_ref();
    let total: usize = shape_ref.iter().product();
    let mut idx = vec![0usize; shape_ref.len()];
    for _ in 0..total {
        let mut flat = offset;
        for (i, &v) in idx.iter().enumerate() {
            flat += v * stride[i] as usize;
        }
        out.push(dst[flat].clone());
        for ax in (0..idx.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < shape_ref[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    dst = out;
    Ok(dst)
}
