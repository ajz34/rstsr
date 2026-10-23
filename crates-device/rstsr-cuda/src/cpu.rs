//! Host-read capability traits for `DeviceCuda`: `DeviceToCpuAPI`
//! (stream-ordered D2H + sync; no `Raw: Clone` requirement) and formatting
//! (stage the visible elements, print from them; never a full D2H).
//!
//! Coherence note: `DeviceFormatAPI` must be implemented per concrete dtype
//! (per-dtype impls below). A generic-`T` impl conflicts with rstsr-core's blanket
//! impl for `Raw = Vec<T>` devices — with generic `T`, rustc cannot decide
//! `CudaBuffer<T> != Vec<T>`; with concrete `T` it can.

use crate::buffer::CudaBuffer;
use crate::prelude_dev::*;
use crate::transfer;
use core::fmt::Formatter;
use duplicate::duplicate_item;
use num::Complex;

impl<T> DeviceToCpuAPI<T> for DeviceCuda
where
    T: Clone + Send + Sync + 'static,
{
    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>,
    {
        transfer::d2h_full(storage.raw())
    }

    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataCloneAPI<Data = Self::Raw>,
    {
        transfer::d2h_full(storage.raw())
    }
}

impl DeviceCuda {
    /// Stage the visible elements one `d2h_scalar` at a time, then print
    /// through the shared CPU walker on the original layout.
    /// // DEFER: single gather kernel; correctness does not depend on this.
    pub(crate) fn format_with<T, D>(
        &self,
        raw: &CudaBuffer<T>,
        layout: &Layout<D>,
        f: &mut Formatter<'_>,
        fmt_elem: fn(&T, &mut Formatter<'_>) -> core::fmt::Result,
    ) -> core::fmt::Result
    where
        T: Clone,
        D: DimAPI,
    {
        let max_print = MAX_PRINT.load(core::sync::atomic::Ordering::Relaxed);
        let min_print = MIN_PRINT.load(core::sync::atomic::Ordering::Relaxed);
        // Never a full-tensor D2H: fetch exactly the elements the walker
        // touches (absolute buffer indices, offset included). The staged vec
        // holds the FULL logical size so the walker's stride arithmetic lands
        // on filled slots; untouched slots are seeded with one fetched element
        // (never read by the walker, but keeps the vec sound). Printing then
        // goes through the shared walker on the original layout.
        let idxs = crate::format::touched_element_indices(layout, max_print, min_print);
        let size = layout.size();
        let mut staged: Vec<T> = Vec::with_capacity(size);
        if !idxs.is_empty() {
            let seed = transfer::d2h_scalar(raw, idxs[0]);
            staged.resize(size, seed);
            for &i in &idxs {
                staged[i] = transfer::d2h_scalar(raw, i);
            }
        }
        let staged_layout = crate::format::staged_layout_of(layout);
        format_tensor_cpu_serial(f, &staged, &staged_layout, max_print, min_print, fmt_elem)
    }
}

/// Per-dtype `DeviceFormatAPI` impls (see the coherence note above).
/// `LowerExp`/`UpperExp` only exist for floats; integers/bool get the
/// Display/Debug pair (rstsr errors on `{:e}` for them like the CPU device).
#[duplicate_item(
    T;
    [bool];
    [i8];
    [i16];
    [i32];
    [i64];
    [isize];
    [u8];
    [u16];
    [u32];
    [u64];
    [usize]
)]
impl<D> DeviceFormatAPI<T, D> for DeviceCuda
where
    D: DimAPI,
{
    fn format_display(&self, raw: &Self::Raw, layout: &Layout<D>, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format_with(raw, layout, f, core::fmt::Display::fmt)
    }
    fn format_debug(&self, raw: &Self::Raw, layout: &Layout<D>, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format_with(raw, layout, f, core::fmt::Debug::fmt)
    }
    fn format_lower_exp(&self, _raw: &Self::Raw, _layout: &Layout<D>, _f: &mut Formatter<'_>) -> core::fmt::Result {
        Err(core::fmt::Error)
    }
    fn format_upper_exp(&self, _raw: &Self::Raw, _layout: &Layout<D>, _f: &mut Formatter<'_>) -> core::fmt::Result {
        Err(core::fmt::Error)
    }
}

#[duplicate_item(
    T;
    [f32];
    [f64];
    [half::f16];
    [half::bf16];
    [Complex<f32>];
    [Complex<f64>]
)]
impl<D> DeviceFormatAPI<T, D> for DeviceCuda
where
    D: DimAPI,
{
    fn format_display(&self, raw: &Self::Raw, layout: &Layout<D>, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format_with(raw, layout, f, core::fmt::Display::fmt)
    }
    fn format_debug(&self, raw: &Self::Raw, layout: &Layout<D>, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format_with(raw, layout, f, core::fmt::Debug::fmt)
    }
    fn format_lower_exp(&self, raw: &Self::Raw, layout: &Layout<D>, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format_with(raw, layout, f, core::fmt::LowerExp::fmt)
    }
    fn format_upper_exp(&self, raw: &Self::Raw, layout: &Layout<D>, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.format_with(raw, layout, f, core::fmt::UpperExp::fmt)
    }
}
