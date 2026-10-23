//! `OpAssignAPI` / `OpAssignArbitaryAPI` for `DeviceCuda` — kernel-backed.
//!
//! Same-dtype assigns use the dtype-free byte-copy kernel; cross-dtype
//! assigns generate a cast kernel per (TA, TC) pair (D6). Contiguous
//! whole-buffer pairs still take the `memcpy_dtod` fast path (async).

use crate::buffer::CudaBuffer;
use crate::dtype::CudaDType;
use crate::error::map_driver_result;
use crate::kernels_ops::{copy_cast, copy_same_dtype, fill_dispatch};
use crate::prelude_dev::*;
use cudarc::driver::DevicePtrMut;

/// Whole-buffer contiguous same-size copy (async; no per-op synchronize —
/// ordering comes from stream semantics, host reads sync).
fn memcpy_fast<TC, TA>(c: &mut CudaBuffer<TC>, a: &CudaBuffer<TA>) -> Result<()> {
    let byte_len = c.len() * core::mem::size_of::<TC>();
    let stream = c.as_bytes().stream().clone();
    // order a foreign-stream source before the copy (the raw accessor below
    // bypasses cudarc's own cross-stream event)
    let _guard = crate::kernel::order_input_raw(a, &stream)?;
    let sptr = a.device_ptr() as u64;
    let dptr = {
        let c_bytes = c.as_mut_bytes();
        let (ptr, _record) = DevicePtrMut::<u8>::device_ptr_mut(c_bytes, &stream);
        ptr
    };
    unsafe {
        map_driver_result(cudarc::driver::result::memcpy_dtod_async(dptr, sptr, byte_len, stream.cu_stream()))?;
    }
    Ok(())
}

/// Shared assign body: `c[lc] = a[la]` elementwise (with cast).
fn assign_impl<TC, TA, DC, DA>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<DC>,
    a: &CudaBuffer<TA>,
    la: &Layout<DA>,
) -> Result<()>
where
    TC: CudaDType,
    TA: CudaDType,
    DC: DimAPI,
    DA: DimAPI,
{
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
        return memcpy_fast(c, a);
    }
    if core::any::TypeId::of::<TC>() == core::any::TypeId::of::<TA>() {
        // same dtype: dtype-free byte-copy kernel (T1 partially-strided aware)
        let a_typed: &CudaBuffer<TC> = unsafe { transmute_ref_cross::<TA, TC>(a) };
        return copy_same_dtype::<TC, DC, DA>(device, c, lc, a_typed, la);
    }
    copy_cast::<TA, TC, DA, DC>(device, c, lc, a, la)
}

impl<TC, TA, D> OpAssignAPI<TC, D, TA> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TC, Raw = CudaBuffer<TC>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: CudaDType,
    TA: CudaDType + DTypeCastAPI<TC>,
    D: DimAPI,
{
    fn assign(&self, c: &mut CudaBuffer<TC>, lc: &Layout<D>, a: &CudaBuffer<TA>, la: &Layout<D>) -> Result<()> {
        assign_impl::<TC, TA, D, D>(self, c, lc, a, la)
    }

    fn assign_uninit(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
    ) -> Result<()> {
        // MaybeUninit<TC> is layout-identical to TC (crate invariant).
        let c_typed: &mut CudaBuffer<TC> = unsafe { transmute_mut(c) };
        assign_impl::<TC, TA, D, D>(self, c_typed, lc, a, la)
    }

    fn fill(&self, c: &mut CudaBuffer<TC>, lc: &Layout<D>, fill: TA) -> Result<()> {
        if lc.size() == 0 {
            return Ok(());
        }
        fill_dispatch(self, c, lc, fill)
    }
}

impl<TC, TA, DC, DA> OpAssignArbitaryAPI<TC, DC, DA, TA> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TC, Raw = CudaBuffer<TC>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: CudaDType,
    TA: CudaDType + DTypeCastAPI<TC>,
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
        assign_impl::<TC, TA, DC, DA>(self, c, lc, a, la)
    }

    fn assign_arbitary_uninit(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &CudaBuffer<TA>,
        la: &Layout<DA>,
    ) -> Result<()> {
        let c_typed: &mut CudaBuffer<TC> = unsafe { transmute_mut(c) };
        assign_impl::<TC, TA, DC, DA>(self, c_typed, lc, a, la)
    }
}

/// `&CudaBuffer<MaybeUninit<T>> -> &CudaBuffer<T>` (layout-identical types;
/// the crate's `assume_init_impl` relies on the same equivalence).
///
/// # Safety
/// `MaybeUninit<T>` and `T` have identical layout (standard library
/// guarantee); the buffer wrapper adds no layout of its own.
unsafe fn transmute_mut<T>(b: &mut CudaBuffer<MaybeUninit<T>>) -> &mut CudaBuffer<T> {
    &mut *(b as *mut CudaBuffer<MaybeUninit<T>> as *mut CudaBuffer<T>)
}

/// `&CudaBuffer<TA> -> &CudaBuffer<TC>` after a TypeId equality check.
///
/// # Safety
/// Caller must prove `TypeId::of::<TA>() == TypeId::of::<TC>()` (the
/// buffers are then layout-identical, as in the CPU TypeId fast paths).
unsafe fn transmute_ref_cross<TA, TC>(b: &CudaBuffer<TA>) -> &CudaBuffer<TC> {
    &*(b as *const CudaBuffer<TA> as *const CudaBuffer<TC>)
}
