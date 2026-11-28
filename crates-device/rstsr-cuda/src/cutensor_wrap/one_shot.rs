//! The tensor-side wrapper functions: every entry point that takes `TensorViewAPI` operands and
//! runs a whole cuTENSOR pipeline in one call, for callers that do not reuse a problem. All
//! operation families live here (contraction today; reduction, permutation, elementwise, trinary
//! contraction as they arrive), one file away from the RAII object layer.

use crate::cutensor_wrap::descriptor::TensorDescriptor;
use crate::cutensor_wrap::dtype::CutensorDType;
use crate::cutensor_wrap::error;
use crate::cutensor_wrap::operation::OperationDescriptor;
use crate::cutensor_wrap::plan::{Plan, PlanPreference};
use crate::cutensor_wrap::workspace::CutensorWorkspace;
use crate::prelude_dev::*;
use cudarc::cutensor::sys;

/// Execute the contraction `D = alpha * opA(A) * opB(B) + beta * C` in one call: descriptors,
/// operation, plan preference, plan and execute, all discarded afterwards.
///
/// The convenience path of [`Plan::contract`] — it pays the planning cost (20-50 us per call,
/// plus a one-time ~28 ms lazy cuBLAS initialization on the first cuBLAS-mapped problem) on
/// every call; hold a [`Plan`] when the same problem repeats. The handle comes from the device's
/// ordinal slot; the workspace defaults to the stream's slot (see [`CutensorWorkspace`]).
///
/// - `c` [`Option::None`] means **C is D**: the output pointer stands in for `C` at any `beta`
///   (in-place accumulation, `d += alpha * opA(A) * opB(B)` at `beta = 1`; two aliasing borrows are
///   not expressible in safe Rust, and cuTENSOR itself defines this form).
/// - All operands enter as views by value (`impl TensorViewAPI` / `TensorViewMutAPI`): pass
///   `t.view()` / `t.view_mut()`; an owned tensor may be moved into `a`, `b` or `c` as well.
/// - `mode_*` are cuTENSOR's raw `int32` mode labels, one per extent of the operand's layout; rank
///   0 is the empty slice.
/// - `op_a` / `op_b` are raw `cudarc` operators (`CUTENSOR_OP_CONJ` conjugates a complex operand);
///   `opC` has no parameter — cuTENSOR honours only `IDENTITY` there.
/// - `compute` is one of [`compute_desc`](crate::cutensor_wrap::compute_desc)'s descriptors.
/// - Same-dtype operands (`TA = TB = TD`) are the served path; other combinations are the library's
///   own table and decline (`NOT_SUPPORTED`) where it does not take them.
///
/// Refusals (named `InvalidValue` / `InvalidLayout` errors, never silent): negative strides,
/// zero extents, a stride-0 output, duplicate or inconsistent modes, an extent mismatch between
/// operands, `beta != 0` without `c`. Stride-0 *inputs* are passed through — cuTENSOR's runtime
/// status is the verdict (accepted from 2.6.0, `NOT_SUPPORTED` before).
#[allow(clippy::too_many_arguments)]
pub fn contract<TA, TB, TD>(
    device: &DeviceCuda,
    a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = IxD>,
    mode_a: &[i32],
    op_a: sys::cutensorOperator_t,
    b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = IxD>,
    mode_b: &[i32],
    op_b: sys::cutensorOperator_t,
    c: Option<impl TensorViewAPI<Type = TD, Backend = DeviceCuda, Dim = IxD>>,
    mut d: impl TensorViewMutAPI<Type = TD, Backend = DeviceCuda, Dim = IxD>,
    mode_d: &[i32],
    alpha: TD::Scalar,
    beta: TD::Scalar,
    compute: sys::cutensorComputeDescriptor_t,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<()>
where
    TA: CutensorDType,
    TB: CutensorDType,
    TD: CutensorDType,
{
    let handle = device.ordinal_obj().cutensor().cloned().ok_or_else(error::unavailable)?;
    // Workspace resolution: the argument, else the stream slot; a flag-driven slot that failed
    // to allocate is a `MemoryError`, not a silent no-workspace run.
    let slot_ws;
    let ws: Option<&CutensorWorkspace> = match workspace {
        Some(ws) => Some(ws),
        None => {
            let bytes = device.flags().cutensor_workspace_bytes;
            if bytes == 0 {
                None
            } else {
                let buf = device.stream().cutensor_workspace(bytes).ok_or_else(|| {
                    rstsr_error!(
                        MemoryError,
                        "cutensor: the default cuTENSOR workspace of {bytes} bytes could not \
                         be allocated"
                    )
                })?;
                slot_ws = Some(CutensorWorkspace::from_slot(buf.clone()));
                slot_ws.as_ref()
            }
        },
    };
    let (a, b, d) = (a.view(), b.view(), d.view_mut());
    let c = c.as_ref().map(|v| v.view());
    let d_view = d.view();
    let desc_d = TensorDescriptor::new::<TD>(&handle, &d_view)?;
    let desc_a = TensorDescriptor::new::<TA>(&handle, &a)?;
    let desc_b = TensorDescriptor::new::<TB>(&handle, &b)?;
    let op = OperationDescriptor::new_contraction::<TD>(
        &handle, &desc_a, mode_a, op_a, &desc_b, mode_b, op_b, &desc_d, mode_d, &desc_d, mode_d, compute,
    )?;
    let pref = PlanPreference::new(
        &handle,
        sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )?;
    let plan = Plan::new(&handle, op, &pref, ws.map_or(0, |w| w.len() as u64))?;
    plan.contract(device, a, b, c, d, alpha, beta, ws)
}
