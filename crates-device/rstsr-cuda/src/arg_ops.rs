//! `argmin`/`argmax` on CUDA: four rows, all sharing one kernel.
//!
//! | trait | op | output | NaN policy |
//! |---|---|---|---|
//! | `OpArgMaxAPI` | `argmax` | `usize` | a leading NaN wins; a later one is ignored |
//! | `OpArgMinAPI` | `argmin` | `usize` | same |
//! | `OpNanArgMaxAPI` | `nanargmax` | `usize` | NaN elements are skipped; all-NaN raises |
//! | `OpNanArgMinAPI` | `nanargmin` | `usize` | same |
//!
//! A sibling of [`crate::reduce_ops`] rather than a block inside it, because three of the four things
//! this family does are *not* what the reduce family does:
//!
//! 1. **An empty contraction is an error, not a fill.** `sum([])` is `0`; `argmax([])` has no answer,
//!    and the CPU raises (`reduce_all_unraveled_arg_fold_cpu_serial`: "empty sequence is not allowed
//!    for reduce_arg"). So this module does not use `reduce_axes_setup`, whose whole point is the
//!    fill-versus-allocate decision.
//! 2. **The index is raveled over the *logical* C-order of the reduced axes**, which the plan's
//!    stride-sorted traversal does not necessarily enumerate in. The kernel is refused unless
//!    `contracted_is_logical` holds — see `kernels::reduce_arg` — and that check has no analogue
//!    in the value-only reductions, where summation order is not observable.
//! 3. **The NaN policies have an error contract**, so their output is read back and checked. That is
//!    the price of parity with the CPU, and it is paid only by the two `nan` rows.
//!
//! The two `OpUnraveledArgMin/MaxAPI` rows (a multi-index output) are **not** implemented, and the
//! reason is not effort: their output is a tensor of index tuples with its own layout rule, and
//! nothing in REST calls them. They are deferred by name rather than half-implemented.
//!
//! # The read-back
//!
//! `nanargmin`/`nanargmax` raise when a slice is all-NaN, exactly as NumPy and the CPU do. A kernel
//! cannot raise, so it writes an out-of-range index (`n_contract`) and this module reads the output
//! back to look for it. That is an output-sized D2H on *every* call of the two NaN rows — a real
//! cost, and the reason the plain rows do not pay it. The alternative was to return an
//! out-of-range index silently, which is a wrong answer that looks like a number.

#![allow(clippy::type_complexity)]

use core::mem::MaybeUninit;

use crate::dtype::{CudaCmp, CudaDType};
use crate::kernels::reduce_arg::{reduce_arg_launch, ArgOp};
use crate::kernels::reduce_plan::classify_reduce;
use crate::prelude_dev::*;

/// The shared half of all four rows: split, layout, plan, launch, and (for the NaN rows) validate.
fn arg_axes_impl<TIn>(
    device: &DeviceCuda,
    op: ArgOp,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    axes: &[isize],
    validate: bool,
) -> Result<(Storage<DataOwned<CudaBuffer<usize>>, usize, DeviceCuda>, Layout<IxD>)>
where
    TIn: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<usize, Raw = CudaBuffer<usize>>
        + DeviceRawAPI<MaybeUninit<usize>, Raw = CudaBuffer<MaybeUninit<usize>>>
        + DeviceCreationAnyAPI<usize>
        + DeviceToCpuAPI<usize>,
{
    let (las, lm) = la.dim_split_axes(axes)?;
    if las.size() == 0 {
        rstsr_raise!(InvalidValue, "empty sequence is not allowed for reduce_arg")?;
    }

    let lo = layout_for_array_copy(&lm, TensorIterOrder::K)?;
    let n_out = lo.size();
    let plan = classify_reduce(la, la, &lo, axes, axes, device.default_order())?;

    // SAFETY: the kernel writes all `n_out` elements (or there are none), which is what `empty_impl`
    // asks the caller to guarantee before the buffer is read.
    let mut st: Storage<DataOwned<CudaBuffer<usize>>, usize, DeviceCuda> = unsafe { device.empty_impl(n_out)? };

    if n_out > 0 {
        let buf: &mut CudaBuffer<usize> = st.raw_mut();
        // SAFETY: layout-identical `MaybeUninit` transmute, the crate's standing convention.
        let buf_uninit: &mut CudaBuffer<MaybeUninit<usize>> =
            unsafe { &mut *(buf as *mut CudaBuffer<usize> as *mut CudaBuffer<MaybeUninit<usize>>) };
        reduce_arg_launch(device, op, buf_uninit, &lo, a, &las, &plan)?;
    }

    // The kernel writes `n_contract` for "no element won", which is only reachable for the NaN rows.
    // Checked here so the error arrives at the call the user wrote, not at whatever reads the index.
    if validate && op.skips_nan() && n_out > 0 {
        let host: Vec<usize> = DeviceToCpuAPI::<usize>::to_cpu_vec::<DataOwned<CudaBuffer<usize>>>(&st)?;
        if host.iter().any(|v| *v >= plan.n_contract) {
            rstsr_raise!(InvalidValue, "All-NaN slice encountered")?;
        }
    }

    Ok((st, lo))
}

/// `*_all`: reduce every axis and hand back the single index.
fn arg_all_impl<TIn>(device: &DeviceCuda, op: ArgOp, a: &CudaBuffer<TIn>, la: &Layout<IxD>) -> Result<usize>
where
    TIn: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<usize, Raw = CudaBuffer<usize>>
        + DeviceRawAPI<MaybeUninit<usize>, Raw = CudaBuffer<MaybeUninit<usize>>>
        + DeviceCreationAnyAPI<usize>
        + DeviceToCpuAPI<usize>,
{
    let axes: Vec<isize> = (0..la.ndim() as isize).collect();
    // `validate = false`: the single value is checked below, so a read-back here would be the same
    // copy twice.
    let (st, _lo) = arg_axes_impl(device, op, a, la, &axes, false)?;
    let host: Vec<usize> = DeviceToCpuAPI::<usize>::to_cpu_vec::<DataOwned<CudaBuffer<usize>>>(&st)?;
    match host.first() {
        None => rstsr_raise!(InvalidValue, "empty sequence is not allowed for reduce_arg"),
        Some(v) if *v >= la.size() && op.skips_nan() => {
            rstsr_raise!(InvalidValue, "All-NaN slice encountered")
        }
        Some(v) => Ok(*v),
    }
}

/// One row per trait. The four differ only in the `ArgOp`, so they are one macro invocation each
/// rather than four near-identical impls.
macro_rules! impl_arg {
    ($tr:ident, $op:expr, $all:ident, $axes:ident) => {
        impl<T, D> $tr<T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
                + DeviceRawAPI<usize, Raw = CudaBuffer<usize>>
                + DeviceRawAPI<MaybeUninit<usize>, Raw = CudaBuffer<MaybeUninit<usize>>>
                + DeviceCreationAnyAPI<usize>
                + DeviceToCpuAPI<usize>,
            // `CudaCmp` is the ordering bound, and it is what excludes complex — the same exclusion
            // the CPU makes with `PartialOrd`.
            T: CudaDType + CudaCmp,
            D: DimAPI,
        {
            type TOut = usize;

            fn $all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<usize> {
                arg_all_impl(self, $op, a, &la.to_dim::<IxD>()?)
            }

            fn $axes(
                &self,
                a: &CudaBuffer<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(
                Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>,
                Layout<IxD>,
            )> {
                arg_axes_impl(self, $op, a, &la.to_dim::<IxD>()?, axes, true)
            }
        }
    };
}

impl_arg!(OpArgMaxAPI, ArgOp::Max, argmax_all, argmax_axes);
impl_arg!(OpArgMinAPI, ArgOp::Min, argmin_all, argmin_axes);
impl_arg!(OpNanArgMaxAPI, ArgOp::NanMax, nanargmax_all, nanargmax_axes);
impl_arg!(OpNanArgMinAPI, ArgOp::NanMin, nanargmin_all, nanargmin_axes);
