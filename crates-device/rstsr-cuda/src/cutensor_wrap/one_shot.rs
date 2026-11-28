//! The tensor-side wrapper functions: every entry point that takes `TensorViewAPI` operands and
//! runs a whole cuTENSOR pipeline in one call, for callers that do not reuse a problem. All
//! operation families live here (contraction, trinary contraction, reduction, permutation,
//! elementwise binary/trinary), one file away from the RAII object layer.

use crate::cutensor_wrap::descriptor::TensorDescriptor;
use crate::cutensor_wrap::dtype::CutensorDType;
use crate::cutensor_wrap::error;
use crate::cutensor_wrap::handle::CutensorHandle;
use crate::cutensor_wrap::operation::OperationDescriptor;
use crate::cutensor_wrap::plan::{Plan, PlanPreference};
use crate::cutensor_wrap::workspace::CutensorWorkspace;
use crate::prelude_dev::*;
use cudarc::cutensor::sys;
use std::sync::Arc;

/// The one-shot preamble: the handle from the device's ordinal slot, and the workspace (the
/// argument if given, else the stream slot; a flag-driven slot that failed to allocate is a
/// `MemoryError`, not a silent no-workspace run). The `CutensorWorkspace` return is the slot
/// allocation (or a clone of the argument); keep it alive for the duration of the call.
fn entry(
    device: &DeviceCuda,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<(Arc<CutensorHandle>, Option<CutensorWorkspace>)> {
    let handle = device.ordinal_obj().cutensor().cloned().ok_or_else(error::unavailable)?;
    let ws = match workspace {
        Some(ws) => Some(ws.clone()),
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
                Some(CutensorWorkspace::from_slot(buf.clone()))
            }
        },
    };
    Ok((handle, ws))
}

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
/// operands. Stride-0 *inputs* are passed through — cuTENSOR's runtime status is the verdict
/// (accepted from 2.6.0, `NOT_SUPPORTED` before).
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
    let (handle, ws_slot) = entry(device, workspace)?;
    let ws = ws_slot.as_ref();
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

/// Execute the reduction `D = alpha * opReduce(opA(A)) + beta * C` in one call: the modes of `A`
/// absent from `mode_d` are the reduced ones (`i,j->j` sums over `i`; a rank-0 `mode_d` reduces
/// everything).
///
/// - `op_reduce` is the raw `cudarc` reduction operator, passed through unchecked (`ADD`, `MUL`,
///   `MIN`, `MAX` are the 2.x set; anything else is cuTENSOR's status to report); `op_a` stays raw
///   (`CUTENSOR_OP_CONJ` conjugates first).
/// - `c` [`Option::None`] means **C is D**: in-place accumulation at any `beta`.
/// - `mode_c` has no parameter: `descC == descD` is cuTENSOR's own requirement, and `C`'s modes are
///   `D`'s.
/// - Refusals: as for [`contract`] — plus an output mode that occurs in no input.
#[allow(clippy::too_many_arguments)]
pub fn reduce<T: CutensorDType>(
    device: &DeviceCuda,
    a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_a: &[i32],
    op_a: sys::cutensorOperator_t,
    c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
    mut d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_d: &[i32],
    op_reduce: sys::cutensorOperator_t,
    alpha: T::Scalar,
    beta: T::Scalar,
    compute: sys::cutensorComputeDescriptor_t,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<()> {
    let (handle, ws_slot) = entry(device, workspace)?;
    let ws = ws_slot.as_ref();
    let (a, d) = (a.view(), d.view_mut());
    let c = c.as_ref().map(|v| v.view());
    let d_view = d.view();
    let desc_d = TensorDescriptor::new::<T>(&handle, &d_view)?;
    let desc_a = TensorDescriptor::new::<T>(&handle, &a)?;
    let op = OperationDescriptor::new_reduction::<T>(
        &handle, &desc_a, mode_a, op_a, &desc_d, mode_d, &desc_d, mode_d, op_reduce, compute,
    )?;
    let pref = PlanPreference::new(
        &handle,
        sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )?;
    let plan = Plan::new(&handle, op, &pref, ws.map_or(0, |w| w.len() as u64))?;
    plan.reduce(device, a, c, d, alpha, beta, ws)
}

/// Execute the permutation `B = alpha * opA(A)` in one call: `mode_b` is the output's label
/// order (the differing order *is* the permutation), so `mode_a` and `mode_b` must share one
/// mode set. Same-dtype path; dtype-converting permutations are cuTENSOR's table and unexplored.
/// Refusals: negative strides, zero extents, a stride-0 output, mode-set mismatches.
#[allow(clippy::too_many_arguments)]
pub fn permute<T: CutensorDType>(
    device: &DeviceCuda,
    a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_a: &[i32],
    op_a: sys::cutensorOperator_t,
    mut b: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_b: &[i32],
    alpha: T::Scalar,
    compute: sys::cutensorComputeDescriptor_t,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<()> {
    let (handle, ws_slot) = entry(device, workspace)?;
    let ws = ws_slot.as_ref();
    let (a, b) = (a.view(), b.view_mut());
    let b_view = b.view();
    let desc_b = TensorDescriptor::new::<T>(&handle, &b_view)?;
    let desc_a = TensorDescriptor::new::<T>(&handle, &a)?;
    let op = OperationDescriptor::new_permutation::<T>(&handle, &desc_a, mode_a, op_a, &desc_b, mode_b, compute)?;
    let pref = PlanPreference::new(
        &handle,
        sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )?;
    let plan = Plan::new(&handle, op, &pref, ws.map_or(0, |w| w.len() as u64))?;
    plan.permute(device, a, b, alpha, ws)
}

/// Execute the elementwise binary `D = op_ac(alpha * opA(A), gamma * opC(C))` in one call —
/// `op_ac` is the *functional* join operator (`ADD`, `MUL`, `MAX`, `MIN`, ...), unlike
/// contraction's `opC`, which the wrapper pins to `IDENTITY` and does not take as a parameter.
/// Every operand shares `D`'s mode set; broadcast goes through stride-0 *views* (accepted by
/// cuTENSOR from 2.6.0).
///
/// `c` [`Option::None`] means **C is D**: in-place elementwise (`d = alpha * a + gamma * d` at
/// `op_ac = ADD`). `mode_c` has no parameter (`descC == descD` is cuTENSOR's own requirement).
#[allow(clippy::too_many_arguments)]
pub fn elementwise_binary<T: CutensorDType>(
    device: &DeviceCuda,
    a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_a: &[i32],
    op_a: sys::cutensorOperator_t,
    c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
    mut d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_d: &[i32],
    op_ac: sys::cutensorOperator_t,
    alpha: T::Scalar,
    gamma: T::Scalar,
    compute: sys::cutensorComputeDescriptor_t,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<()> {
    let (handle, ws_slot) = entry(device, workspace)?;
    let ws = ws_slot.as_ref();
    let (a, d) = (a.view(), d.view_mut());
    let c = c.as_ref().map(|v| v.view());
    let d_view = d.view();
    let desc_d = TensorDescriptor::new::<T>(&handle, &d_view)?;
    let desc_a = TensorDescriptor::new::<T>(&handle, &a)?;
    let op = OperationDescriptor::new_elementwise_binary::<T>(
        &handle, &desc_a, mode_a, op_a, &desc_d, mode_d, &desc_d, mode_d, op_ac, compute,
    )?;
    let pref = PlanPreference::new(
        &handle,
        sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )?;
    let plan = Plan::new(&handle, op, &pref, ws.map_or(0, |w| w.len() as u64))?;
    plan.elementwise_binary(device, a, c, d, alpha, gamma, ws)
}

/// Execute the elementwise trinary
/// `D = op_abc(op_ab(alpha * opA(A), beta * opB(B)), gamma * opC(C))` in one call: `op_ab`
/// joins `A` and `B`, `op_abc` joins that with `C`. Every operand shares `D`'s mode set.
///
/// `c` [`Option::None`] means **C is D**: in-place trinary (`d += a + b` at `op_ab = op_abc =
/// ADD`, `alpha = beta = gamma = 1`). `mode_c` has no parameter (`descC == descD` is cuTENSOR's
/// own requirement).
#[allow(clippy::too_many_arguments)]
pub fn elementwise_trinary<T: CutensorDType>(
    device: &DeviceCuda,
    a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_a: &[i32],
    op_a: sys::cutensorOperator_t,
    b: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_b: &[i32],
    op_b: sys::cutensorOperator_t,
    c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
    mut d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_d: &[i32],
    op_ab: sys::cutensorOperator_t,
    op_abc: sys::cutensorOperator_t,
    alpha: T::Scalar,
    beta: T::Scalar,
    gamma: T::Scalar,
    compute: sys::cutensorComputeDescriptor_t,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<()> {
    let (handle, ws_slot) = entry(device, workspace)?;
    let ws = ws_slot.as_ref();
    let (a, b, d) = (a.view(), b.view(), d.view_mut());
    let c = c.as_ref().map(|v| v.view());
    let d_view = d.view();
    let desc_d = TensorDescriptor::new::<T>(&handle, &d_view)?;
    let desc_a = TensorDescriptor::new::<T>(&handle, &a)?;
    let desc_b = TensorDescriptor::new::<T>(&handle, &b)?;
    let op = OperationDescriptor::new_elementwise_trinary::<T>(
        &handle, &desc_a, mode_a, op_a, &desc_b, mode_b, op_b, &desc_d, mode_d, &desc_d, mode_d, op_ab, op_abc, compute,
    )?;
    let pref = PlanPreference::new(
        &handle,
        sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )?;
    let plan = Plan::new(&handle, op, &pref, ws.map_or(0, |w| w.len() as u64))?;
    plan.elementwise_trinary(device, a, b, c, d, alpha, beta, gamma, ws)
}

/// Execute the trinary contraction `E = alpha * opA(A) * opB(B) * opC(C) + beta * opD(D)` in
/// one call: three contracted inputs accumulated with `D`. The contraction analog of the other
/// families' `c: None`: `d` [`Option::None`] means **D is E** (in-place
/// `e += alpha * a * b * c` at `beta = 1`); `mode_d` has no parameter (`descD == descE` is
/// cuTENSOR's own requirement).
///
/// Gated: requires libcutensor >= 2.1.1 (older libraries decline with `UnImplemented`). Some
/// dtype rows need Ampere+; cuTENSOR declines those itself at plan. `opD` is pinned to
/// `IDENTITY` — see [`contract`] for the pinned-operator policy.
#[allow(clippy::too_many_arguments)]
pub fn contract_trinary<T: CutensorDType>(
    device: &DeviceCuda,
    a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_a: &[i32],
    op_a: sys::cutensorOperator_t,
    b: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_b: &[i32],
    op_b: sys::cutensorOperator_t,
    c: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_c: &[i32],
    op_c: sys::cutensorOperator_t,
    d: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
    mut e: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
    mode_e: &[i32],
    alpha: T::Scalar,
    beta: T::Scalar,
    compute: sys::cutensorComputeDescriptor_t,
    workspace: Option<&CutensorWorkspace>,
) -> rstsr_core::prelude_dev::Result<()> {
    let (handle, ws_slot) = entry(device, workspace)?;
    let ws = ws_slot.as_ref();
    let (a, b, c, e) = (a.view(), b.view(), c.view(), e.view_mut());
    let d = d.as_ref().map(|v| v.view());
    let e_view = e.view();
    let desc_e = TensorDescriptor::new::<T>(&handle, &e_view)?;
    let desc_a = TensorDescriptor::new::<T>(&handle, &a)?;
    let desc_b = TensorDescriptor::new::<T>(&handle, &b)?;
    let desc_c = TensorDescriptor::new::<T>(&handle, &c)?;
    let op = OperationDescriptor::new_contraction_trinary::<T>(
        &handle, &desc_a, mode_a, op_a, &desc_b, mode_b, op_b, &desc_c, mode_c, op_c, &desc_e, mode_e, &desc_e, mode_e,
        compute,
    )?;
    let pref = PlanPreference::new(
        &handle,
        sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )?;
    let plan = Plan::new(&handle, op, &pref, ws.map_or(0, |w| w.len() as u64))?;
    plan.contract_trinary(device, a, b, c, d, e, alpha, beta, ws)
}
