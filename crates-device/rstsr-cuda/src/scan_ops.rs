//! `cumsum`/`cumprod` on CUDA.
//!
//! Two rows, one kernel (`kernels::scan`). Unlike the reduce family there is no plan object here:
//! a scan fixes the order of the arithmetic, so the only decisions are the line split and the
//! chunk carry, and both are properties of the kernel rather than of the layout.
//!
//! The output layout is computed exactly as the CPU does it — the shape with the scan axis grown by
//! one when `include_initial`, then `layout_for_array_copy(.., K)` — because the output layout is
//! observable (same numbers, different strides depending on the device would be a divergence).
//!
//! # Why the axis split needs a "probe" layout
//!
//! `layout_for_array_copy` derives the axis *arrangement* from a layout's shape and strides, and a
//! grown scan axis makes the two inconsistent, which `Layout::new` rejects. The CPU's
//! `cumulative_cpu_serial` builds it with `Layout::new_unchecked` for exactly that reason; this
//! mirrors it, with the same argument: the probe is never iterated or dereferenced.
//!
//! # Bounds
//!
//! `CudaReal` is narrower than the CPU's `Clone + Zero + Add`, and deliberately so: a complex scan
//! would have to shuffle a two-register value through the warp tree. Refused by bound, not by a
//! wrong answer.

#![allow(clippy::type_complexity)]

use core::mem::MaybeUninit;

use crate::dtype::{CudaDType, CudaReal};
use crate::kernels::scan::{scan_launch, ScanOp};
use crate::prelude_dev::*;

fn scan_impl<T>(
    device: &DeviceCuda,
    op: ScanOp,
    a: &CudaBuffer<T>,
    la: &Layout<IxD>,
    axis: isize,
    include_initial: bool,
) -> Result<(Storage<DataOwned<CudaBuffer<T>>, T, DeviceCuda>, Layout<IxD>)>
where
    T: CudaDType,
    DeviceCuda: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>,
{
    let axis = rstsr_check_axis!(axis, la.ndim())?;
    // `dim_split_axes` gives the scan axis on its own and everything else as one layout, which is
    // the "line" decomposition the kernel wants.
    let (ls, lm) = la.dim_split_axes(&[axis as isize])?;
    let size_m = ls.shape()[0];
    let step_in = ls.stride()[0];

    let mut shape_o = la.shape().clone();
    shape_o[axis] = if include_initial { size_m + 1 } else { size_m };
    // SAFETY: the probe layout is never iterated or dereferenced — it only feeds
    // `layout_for_array_copy`, which reads its shape and strides to derive the axis arrangement.
    // A grown scan axis makes shape and strides inconsistent, which is precisely why the checked
    // constructor cannot be used here.
    let probe = unsafe { Layout::new_unchecked(shape_o, la.stride().clone(), la.offset()) };
    let lo = layout_for_array_copy(&probe, TensorIterOrder::K)?;
    let lo_rest = lo.dim_chop(axis as isize)?;
    let step_out = lo.stride()[axis];

    // SAFETY: each line writes its own `size_m` (plus the initial slot when asked) consecutive
    // positions, so every element of `lo` is written before it is read.
    let mut st: Storage<DataOwned<CudaBuffer<T>>, T, DeviceCuda> = unsafe { device.empty_impl(lo.size())? };

    if lm.size() > 0 && lo.size() > 0 {
        let buf: &mut CudaBuffer<T> = st.raw_mut();
        // SAFETY: layout-identical `MaybeUninit` transmute, the crate's standing convention.
        let buf_uninit: &mut CudaBuffer<MaybeUninit<T>> =
            unsafe { &mut *(buf as *mut CudaBuffer<T> as *mut CudaBuffer<MaybeUninit<T>>) };
        scan_launch(
            device, op, buf_uninit, &lo_rest, a, &lm, size_m, step_in, step_out, include_initial,
        )?;
    }

    Ok((st, lo))
}

impl<T, D> OpCumSumAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>,
    T: CudaDType + CudaReal + num::Zero,
    D: DimAPI,
{
    type TOut = T;

    fn cumulative_sum(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axis: isize,
        include_initial: bool,
    ) -> Result<(
        Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>,
        Layout<IxD>,
    )> {
        scan_impl(self, ScanOp::Sum, a, &la.to_dim::<IxD>()?, axis, include_initial)
    }
}

impl<T, D> OpCumProdAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>,
    T: CudaDType + CudaReal + num::One,
    D: DimAPI,
{
    type TOut = T;

    fn cumulative_prod(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axis: isize,
        include_initial: bool,
    ) -> Result<(
        Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>,
        Layout<IxD>,
    )> {
        scan_impl(self, ScanOp::Prod, a, &la.to_dim::<IxD>()?, axis, include_initial)
    }
}
