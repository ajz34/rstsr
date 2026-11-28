//! Plan preference and plan: the planner (algorithm choice) and the reusable execution object.

use crate::cutensor_wrap::descriptor::ptr_alignment;
use crate::cutensor_wrap::dtype::{AlignedScale, CutensorDType, CutensorScalar};
use crate::cutensor_wrap::error::{self, Stage};
use crate::cutensor_wrap::handle::CutensorHandle;
use crate::cutensor_wrap::operation::{validate_operand, OpKind, OperandFacts, OperationDescriptor};
use crate::cutensor_wrap::workspace::CutensorWorkspace;
use crate::kernel::{order_input, order_output};
use crate::prelude_dev::*;
use cudarc::cutensor::sys;
use num::Zero;
use std::sync::Arc;

/// The algorithm and JIT choice of a plan. Defaults live at the call sites
/// (`CUTENSOR_ALGO_DEFAULT`, `CUTENSOR_JIT_MODE_NONE`); the raw enums stay the API, per the
/// wrapper's raw-FFI boundary.
pub struct PlanPreference {
    ptr: sys::cutensorPlanPreference_t,
}

impl PlanPreference {
    /// Create a preference on `handle`. Unlocked: the entry allocates a standalone preference
    /// and touches no shared handle state (the header carries no remark for it).
    pub fn new(handle: &CutensorHandle, algo: sys::cutensorAlgo_t, jit_mode: sys::cutensorJitMode_t) -> Result<Self> {
        handle.bind()?;
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorPlanPreference_t>::uninit();
        // SAFETY: `ptr` is a valid output slot for the one-pointer preference.
        let status = unsafe { sys::cutensorCreatePlanPreference(handle.ptr(), ptr.as_mut_ptr(), algo, jit_mode) };
        error::check(Stage::PlanPreference, status, "")?;
        // SAFETY: the preference was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Ok(Self { ptr })
    }

    pub(crate) fn ptr(&self) -> sys::cutensorPlanPreference_t {
        self.ptr
    }
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        // SAFETY: created by `cutensorCreatePlanPreference`, destroyed exactly once.
        let status = unsafe { sys::cutensorDestroyPlanPreference(self.ptr) };
        debug_assert_eq!(status, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
    }
}

impl core::fmt::Debug for PlanPreference {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlanPreference").finish_non_exhaustive()
    }
}

/// A planned cuTENSOR operation: everything the execute call needs except the buffers.
///
/// A plan is bound to its operation (the extent/stride/dtype/alignment facts and mode arrays it
/// was planned with) and to the [`CutensorHandle`] it was created on (the plan cache is per
/// handle; a plan must not outlive it — hence the `Arc`). It executes on any stream of its
/// device, against *new* buffers: [`Plan::contract`] re-derives every operand's facts from the
/// views and refuses a mismatch, because nothing else would catch a different view against the
/// same plan (cuTENSOR's descriptors hold no pointer, and it validates none of this itself).
pub struct Plan {
    ptr: sys::cutensorPlan_t,
    handle: Arc<CutensorHandle>,
    op: OperationDescriptor,
    required_workspace: u64,
}

impl Plan {
    /// Plan `op` on `handle`, with `workspace_limit` bytes as the workspace budget (the planner
    /// only picks algorithms that fit; execute must pass at least [`Self::required_workspace`]).
    ///
    /// Holds the handle lock across estimate + create + the `required_workspace` attribute read
    /// — the planning entries the header leaves without a thread-safety remark.
    pub fn new(
        handle: &Arc<CutensorHandle>,
        op: OperationDescriptor,
        pref: &PlanPreference,
        workspace_limit: u64,
    ) -> Result<Self> {
        handle.bind()?;
        let _lock = handle.lock();
        let mut estimate = 0u64;
        // SAFETY: `op` and `pref` were created on this handle.
        let status = unsafe {
            sys::cutensorEstimateWorkspaceSize(
                handle.ptr(),
                op.ptr(),
                pref.ptr(),
                sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
                core::ptr::addr_of_mut!(estimate),
            )
        };
        error::check(Stage::WorkspaceEstimate, status, "")?;
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorPlan_t>::uninit();
        // SAFETY: as above; the limit is the caller's budget.
        let status =
            unsafe { sys::cutensorCreatePlan(handle.ptr(), ptr.as_mut_ptr(), op.ptr(), pref.ptr(), workspace_limit) };
        error::check(Stage::Plan, status, "")?;
        // SAFETY: the plan was written on success.
        let ptr = unsafe { ptr.assume_init() };
        let mut required = 0u64;
        // SAFETY: the attribute buffer is one `u64`; the plan is valid.
        let status = unsafe {
            sys::cutensorPlanGetAttribute(
                handle.ptr(),
                ptr,
                sys::cutensorPlanAttribute_t::CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                core::ptr::addr_of_mut!(required).cast(),
                core::mem::size_of::<u64>(),
            )
        };
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            // SAFETY: destroy before returning; the plan never escapes this function.
            let destroy = unsafe { sys::cutensorDestroyPlan(ptr) };
            debug_assert_eq!(destroy, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
            return Err(error::status_error(Stage::Plan, status, "required workspace"));
        }
        Ok(Self { ptr, handle: handle.clone(), op, required_workspace: required })
    }

    /// The workspace in bytes this plan requires at execute time.
    pub fn required_workspace(&self) -> u64 {
        self.required_workspace
    }

    /// The operation the plan was created for (its operand facts drive the re-validation).
    pub fn operation(&self) -> &OperationDescriptor {
        &self.op
    }

    /// Execute the planned contraction `D = alpha * opA(A) * opB(B) + beta * C` on `device`'s
    /// stream, against new buffers.
    ///
    /// Every operand is re-validated against the plan (extent, stride, dtype, pointer alignment)
    /// and every view must live on `device`. `c` [`Option::None`] means **C is D**: the output
    /// pointer stands in for `C` at any `beta` (in-place accumulation, `d += alpha * opA(A) *
    /// opB(B)` at `beta = 1`; two aliasing borrows are not expressible in safe Rust, and cuTENSOR
    /// itself defines this form — probe-verified). The workspace must cover
    /// [`Self::required_workspace`] when given; [`Option::None`] is the caller's choice to run
    /// without one (the plan was created within that budget).
    #[allow(clippy::too_many_arguments)]
    pub fn contract<TA, TB, TD>(
        &self,
        device: &DeviceCuda,
        a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = IxD>,
        b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = IxD>,
        c: Option<impl TensorViewAPI<Type = TD, Backend = DeviceCuda, Dim = IxD>>,
        mut d: impl TensorViewMutAPI<Type = TD, Backend = DeviceCuda, Dim = IxD>,
        alpha: TD::Scalar,
        beta: TD::Scalar,
        workspace: Option<&CutensorWorkspace>,
    ) -> Result<()>
    where
        TA: CutensorDType,
        TB: CutensorDType,
        TD: CutensorDType,
    {
        self.handle.bind()?;
        self.check_kind(OpKind::Contraction)?;
        self.check_scalar_type::<TD>()?;
        self.check_workspace(workspace)?;
        let (a, b, d_view) = (a.view(), b.view(), d.view_mut());
        let c = c.as_ref().map(|v| v.view());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(b.device()), DeviceMismatch, "cutensor: operand B is on another device")?;
        rstsr_assert!(device.same_device(d_view.device()), DeviceMismatch, "cutensor: operand D is on another device")?;
        if let Some(cv) = c.as_ref() {
            rstsr_assert!(device.same_device(cv.device()), DeviceMismatch, "cutensor: operand C is on another device")?;
        }
        // Re-validation: the plan only serves the problem it was planned for. `C`'s facts equal
        // `D`'s (enforced at creation), so a `None` C needs no check of its own.
        let operands = self.op.operands();
        if operands.len() != 4 {
            return Err(rstsr_error!(RuntimeError, "cutensor: execute: the plan is not a contraction"));
        }
        let (plan_a, plan_b, plan_c, plan_d) = (&operands[0], &operands[1], &operands[2], &operands[3]);
        let facts_a = view_facts::<TA>(a.raw(), a.layout());
        validate_operand(plan_a, &facts_a, "A")?;
        let facts_b = view_facts::<TB>(b.raw(), b.layout());
        validate_operand(plan_b, &facts_b, "B")?;
        if let Some(cv) = c.as_ref() {
            let facts_c = view_facts::<TD>(cv.raw(), cv.layout());
            validate_operand(plan_c, &facts_c, "C")?;
        }
        let facts_d = view_facts::<TD>(d_view.raw(), d_view.layout());
        validate_operand(plan_d, &facts_d, "D")?;
        // Ordering: the operand buffers' allocation streams wait for `exec` before the launch,
        // and `D`'s write is made visible to its allocation stream after it (no-op when they are
        // the same stream).
        let exec = device.stream();
        order_input(a.raw(), exec)?;
        order_input(b.raw(), exec)?;
        let esz_d = core::mem::size_of::<TD>();
        let (pa, ga) = a.raw().read_ptr_on(exec);
        let (pb, gb) = b.raw().read_ptr_on(exec);
        let (pd, gd) = d_view.raw().read_ptr_on(exec);
        let ptr_a = operand_ptr(pa, a.offset(), core::mem::size_of::<TA>());
        let ptr_b = operand_ptr(pb, b.offset(), core::mem::size_of::<TB>());
        let ptr_d = operand_ptr(pd, d_view.offset(), esz_d);
        let (ptr_c, gc, gd_keep) = match c.as_ref() {
            Some(cv) => {
                // cuTENSOR does not read `C` at `beta == 0` (verified on 2.4.1-2.8.0): no input
                // ordering is needed there; the pointer guard is still kept past the enqueue.
                if !beta.is_zero() {
                    order_input(cv.raw(), exec)?;
                }
                let (pc, gc) = cv.raw().read_ptr_on(exec);
                (operand_ptr(pc, cv.offset(), esz_d), gc, Some(gd))
            },
            None => (ptr_d, gd, None),
        };
        let alpha_s = AlignedScale::new(alpha);
        let beta_s = AlignedScale::new(beta);
        let (ws_ptr, ws_len, ws_guard) = match workspace {
            Some(ws) => {
                let (ptr, guard) = ws.ptr_on(exec);
                (ptr, ws.len() as u64, Some(guard))
            },
            None => (core::ptr::null_mut(), 0u64, None),
        };
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer ordered on `exec`; the scalars live on the stack across the call.
        let status = unsafe {
            sys::cutensorContract(
                self.handle.ptr(),
                self.ptr,
                alpha_s.as_ptr().cast(),
                ptr_a,
                ptr_b,
                beta_s.as_ptr().cast(),
                ptr_c,
                ptr_d as *mut core::ffi::c_void,
                ws_ptr,
                ws_len,
                stream,
            )
        };
        drop((ga, gb, gc, gd_keep, ws_guard));
        error::check(Stage::Execute, status, "")?;
        order_output(d_view.raw(), exec)?;
        Ok(())
    }
    // ---- shared execute preamble -----------------------------------------------------------

    /// The execute kind must match the planned operation.
    fn check_kind(&self, want: OpKind) -> Result<()> {
        if self.op.kind() != want {
            return Err(rstsr_error!(
                RuntimeError,
                "cutensor: execute: the plan is a {:?}, not a {:?}",
                self.op.kind(),
                want
            ));
        }
        Ok(())
    }

    /// The output scalar type must still match the plan (the FFI reads the scalars blind).
    fn check_scalar_type<T: CutensorDType>(&self) -> Result<()> {
        if T::Scalar::DATA_TYPE != self.op.scalar_type() {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: execute: output scalar type {:?} does not match the planned {:?}",
                T::Scalar::DATA_TYPE,
                self.op.scalar_type()
            ));
        }
        Ok(())
    }

    /// The workspace must cover [`Self::required_workspace`] when given.
    fn check_workspace(&self, workspace: Option<&CutensorWorkspace>) -> Result<()> {
        if let Some(ws) = workspace {
            if (ws.len() as u64) < self.required_workspace {
                return Err(rstsr_error!(
                    InvalidValue,
                    "cutensor: execute: workspace of {} bytes is below the plan's required {}",
                    ws.len(),
                    self.required_workspace
                ));
            }
        }
        Ok(())
    }

    /// Execute the planned trinary contraction
    /// `E = alpha * opA(A) * opB(B) * opC(C) + beta * opD(D)` on `device`'s stream. Same
    /// re-validation discipline as [`Self::contract`]; `d` [`Option::None`] means **D is E**
    /// (in-place accumulate, `e += alpha * a * b * c` at `beta = 1`).
    #[allow(clippy::too_many_arguments)]
    pub fn contract_trinary<T: CutensorDType>(
        &self,
        device: &DeviceCuda,
        a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        b: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        c: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        d: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
        mut e: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        alpha: T::Scalar,
        beta: T::Scalar,
        workspace: Option<&CutensorWorkspace>,
    ) -> Result<()> {
        self.handle.bind()?;
        self.check_kind(OpKind::ContractionTrinary)?;
        self.check_scalar_type::<T>()?;
        self.check_workspace(workspace)?;
        let (a, b, c, e_view) = (a.view(), b.view(), c.view(), e.view_mut());
        let d = d.as_ref().map(|v| v.view());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(b.device()), DeviceMismatch, "cutensor: operand B is on another device")?;
        rstsr_assert!(device.same_device(c.device()), DeviceMismatch, "cutensor: operand C is on another device")?;
        rstsr_assert!(device.same_device(e_view.device()), DeviceMismatch, "cutensor: operand E is on another device")?;
        if let Some(dv) = d.as_ref() {
            rstsr_assert!(device.same_device(dv.device()), DeviceMismatch, "cutensor: operand D is on another device")?;
        }
        let operands = self.op.operands();
        if operands.len() != 5 {
            return Err(rstsr_error!(RuntimeError, "cutensor: execute: the plan is not a trinary contraction"));
        }
        let facts_a = view_facts::<T>(a.raw(), a.layout());
        validate_operand(&operands[0], &facts_a, "A")?;
        let facts_b = view_facts::<T>(b.raw(), b.layout());
        validate_operand(&operands[1], &facts_b, "B")?;
        let facts_c = view_facts::<T>(c.raw(), c.layout());
        validate_operand(&operands[2], &facts_c, "C")?;
        if let Some(dv) = d.as_ref() {
            let facts_d = view_facts::<T>(dv.raw(), dv.layout());
            validate_operand(&operands[3], &facts_d, "D")?;
        }
        let facts_e = view_facts::<T>(e_view.raw(), e_view.layout());
        validate_operand(&operands[4], &facts_e, "E")?;
        let exec = device.stream();
        order_input(a.raw(), exec)?;
        order_input(b.raw(), exec)?;
        order_input(c.raw(), exec)?;
        let esz = core::mem::size_of::<T>();
        let (pa, ga) = a.raw().read_ptr_on(exec);
        let (pb, gb) = b.raw().read_ptr_on(exec);
        let (pc, gc) = c.raw().read_ptr_on(exec);
        let (pe, ge) = e_view.raw().read_ptr_on(exec);
        let (ptr_d, gd, ge_keep) = match d.as_ref() {
            // `D` is ordered unconditionally: the "not read at `beta == 0`" behaviour is
            // established for the binary contraction and the reduction, not for this entry.
            Some(dv) => {
                order_input(dv.raw(), exec)?;
                let (pd, gd) = dv.raw().read_ptr_on(exec);
                (operand_ptr(pd, dv.offset(), esz), gd, Some(ge))
            },
            None => (operand_ptr(pe, e_view.offset(), esz), ge, None),
        };
        let alpha_s = AlignedScale::new(alpha);
        let beta_s = AlignedScale::new(beta);
        let (ws_ptr, ws_len, ws_guard) = match workspace {
            Some(ws) => {
                let (ptr, guard) = ws.ptr_on(exec);
                (ptr, ws.len() as u64, Some(guard))
            },
            None => (core::ptr::null_mut(), 0u64, None),
        };
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer whose ordering events are recorded on `exec`; the scalars live on the
        // stack across the call; the stream belongs to the device the plan is bound to.
        let status = unsafe {
            sys::cutensorContractTrinary(
                self.handle.ptr(),
                self.ptr,
                alpha_s.as_ptr().cast(),
                operand_ptr(pa, a.offset(), esz),
                operand_ptr(pb, b.offset(), esz),
                operand_ptr(pc, c.offset(), esz),
                beta_s.as_ptr().cast(),
                ptr_d,
                operand_ptr(pe, e_view.offset(), esz) as *mut core::ffi::c_void,
                ws_ptr,
                ws_len,
                stream,
            )
        };
        drop((ga, gb, gc, gd, ge_keep, ws_guard));
        error::check(Stage::Execute, status, "")?;
        order_output(e_view.raw(), exec)?;
        Ok(())
    }

    /// Execute the planned reduction `D = alpha * opReduce(opA(A)) + beta * C` on `device`'s
    /// stream. Same re-validation discipline as [`Self::contract`]; `c` [`Option::None`] means
    /// **C is D** (in-place accumulate, `d += alpha * reduce(a)` at `beta = 1`).
    #[allow(clippy::too_many_arguments)]
    pub fn reduce<T: CutensorDType>(
        &self,
        device: &DeviceCuda,
        a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
        mut d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        alpha: T::Scalar,
        beta: T::Scalar,
        workspace: Option<&CutensorWorkspace>,
    ) -> Result<()> {
        self.handle.bind()?;
        self.check_kind(OpKind::Reduction)?;
        self.check_scalar_type::<T>()?;
        self.check_workspace(workspace)?;
        let (a, d_view) = (a.view(), d.view_mut());
        let c = c.as_ref().map(|v| v.view());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(d_view.device()), DeviceMismatch, "cutensor: operand D is on another device")?;
        if let Some(cv) = c.as_ref() {
            rstsr_assert!(device.same_device(cv.device()), DeviceMismatch, "cutensor: operand C is on another device")?;
        }
        let operands = self.op.operands();
        if operands.len() != 3 {
            return Err(rstsr_error!(RuntimeError, "cutensor: execute: the plan is not a reduction"));
        }
        let (plan_a, plan_c, plan_d) = (&operands[0], &operands[1], &operands[2]);
        let facts_a = view_facts::<T>(a.raw(), a.layout());
        validate_operand(plan_a, &facts_a, "A")?;
        if let Some(cv) = c.as_ref() {
            let facts_c = view_facts::<T>(cv.raw(), cv.layout());
            validate_operand(plan_c, &facts_c, "C")?;
        }
        let facts_d = view_facts::<T>(d_view.raw(), d_view.layout());
        validate_operand(plan_d, &facts_d, "D")?;
        let exec = device.stream();
        order_input(a.raw(), exec)?;
        let esz = core::mem::size_of::<T>();
        let (pa, ga) = a.raw().read_ptr_on(exec);
        let (pd, gd) = d_view.raw().read_ptr_on(exec);
        let ptr_a = operand_ptr(pa, a.offset(), esz);
        let ptr_d = operand_ptr(pd, d_view.offset(), esz);
        let (ptr_c, gc, gd_keep) = match c.as_ref() {
            Some(cv) => {
                // as `contract`: `C` is not read at `beta == 0`, so it is ordered only otherwise.
                if !beta.is_zero() {
                    order_input(cv.raw(), exec)?;
                }
                let (pc, gc) = cv.raw().read_ptr_on(exec);
                (operand_ptr(pc, cv.offset(), esz), gc, Some(gd))
            },
            None => (ptr_d, gd, None),
        };
        let alpha_s = AlignedScale::new(alpha);
        let beta_s = AlignedScale::new(beta);
        let (ws_ptr, ws_len, ws_guard) = match workspace {
            Some(ws) => {
                let (ptr, guard) = ws.ptr_on(exec);
                (ptr, ws.len() as u64, Some(guard))
            },
            None => (core::ptr::null_mut(), 0u64, None),
        };
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer whose ordering events are recorded on `exec`; the scalars live on the
        // stack across the call; the stream belongs to the device the plan is bound to.
        let status = unsafe {
            sys::cutensorReduce(
                self.handle.ptr(),
                self.ptr,
                alpha_s.as_ptr().cast(),
                ptr_a,
                beta_s.as_ptr().cast(),
                ptr_c,
                ptr_d as *mut core::ffi::c_void,
                ws_ptr,
                ws_len,
                stream,
            )
        };
        drop((ga, gc, gd_keep, ws_guard));
        error::check(Stage::Execute, status, "")?;
        order_output(d_view.raw(), exec)?;
        Ok(())
    }

    /// Execute the planned permutation `B = alpha * opA(A)` on `device`'s stream. Same
    /// re-validation discipline as [`Self::contract`].
    pub fn permute<T: CutensorDType>(
        &self,
        device: &DeviceCuda,
        a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        mut b: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        alpha: T::Scalar,
        workspace: Option<&CutensorWorkspace>,
    ) -> Result<()> {
        self.handle.bind()?;
        self.check_kind(OpKind::Permutation)?;
        self.check_scalar_type::<T>()?;
        self.check_workspace(workspace)?;
        let (a, b_view) = (a.view(), b.view_mut());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(b_view.device()), DeviceMismatch, "cutensor: operand B is on another device")?;
        let operands = self.op.operands();
        if operands.len() != 2 {
            return Err(rstsr_error!(RuntimeError, "cutensor: execute: the plan is not a permutation"));
        }
        let (plan_a, plan_b) = (&operands[0], &operands[1]);
        let facts_a = view_facts::<T>(a.raw(), a.layout());
        validate_operand(plan_a, &facts_a, "A")?;
        let facts_b = view_facts::<T>(b_view.raw(), b_view.layout());
        validate_operand(plan_b, &facts_b, "B")?;
        let exec = device.stream();
        order_input(a.raw(), exec)?;
        let esz = core::mem::size_of::<T>();
        let (pa, ga) = a.raw().read_ptr_on(exec);
        let (pb, gb) = b_view.raw().read_ptr_on(exec);
        let alpha_s = AlignedScale::new(alpha);
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer ordered on `exec`; the scalars live on the stack across the call.
        let status = unsafe {
            sys::cutensorPermute(
                self.handle.ptr(),
                self.ptr,
                alpha_s.as_ptr().cast(),
                operand_ptr(pa, a.offset(), esz),
                operand_ptr(pb, b_view.offset(), esz) as *mut core::ffi::c_void,
                stream,
            )
        };
        drop((ga, gb));
        error::check(Stage::Execute, status, "")?;
        order_output(b_view.raw(), exec)?;
        Ok(())
    }

    /// Execute the planned elementwise binary `D = op_AC(alpha * opA(A), gamma * opC(C))` on
    /// `device`'s stream. Same re-validation discipline as [`Self::contract`]; `c`
    /// [`Option::None`] means **C is D** (in-place elementwise, `d = alpha * a + gamma * d` at
    /// `op_ac = ADD`).
    pub fn elementwise_binary<T: CutensorDType>(
        &self,
        device: &DeviceCuda,
        a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
        mut d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        alpha: T::Scalar,
        gamma: T::Scalar,
        workspace: Option<&CutensorWorkspace>,
    ) -> Result<()> {
        self.handle.bind()?;
        self.check_kind(OpKind::ElementwiseBinary)?;
        self.check_scalar_type::<T>()?;
        self.check_workspace(workspace)?;
        let (a, d_view) = (a.view(), d.view_mut());
        let c = c.as_ref().map(|v| v.view());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(d_view.device()), DeviceMismatch, "cutensor: operand D is on another device")?;
        if let Some(cv) = c.as_ref() {
            rstsr_assert!(device.same_device(cv.device()), DeviceMismatch, "cutensor: operand C is on another device")?;
        }
        let operands = self.op.operands();
        if operands.len() != 3 {
            return Err(rstsr_error!(RuntimeError, "cutensor: execute: the plan is not an elementwise binary"));
        }
        let (plan_a, plan_c, plan_d) = (&operands[0], &operands[1], &operands[2]);
        let facts_a = view_facts::<T>(a.raw(), a.layout());
        validate_operand(plan_a, &facts_a, "A")?;
        if let Some(cv) = c.as_ref() {
            let facts_c = view_facts::<T>(cv.raw(), cv.layout());
            validate_operand(plan_c, &facts_c, "C")?;
        }
        let facts_d = view_facts::<T>(d_view.raw(), d_view.layout());
        validate_operand(plan_d, &facts_d, "D")?;
        let exec = device.stream();
        order_input(a.raw(), exec)?;
        let esz = core::mem::size_of::<T>();
        let (pa, ga) = a.raw().read_ptr_on(exec);
        let (pd, gd) = d_view.raw().read_ptr_on(exec);
        let (ptr_c, gc, gd_keep) = match c.as_ref() {
            Some(cv) => {
                order_input(cv.raw(), exec)?;
                let (pc, gc) = cv.raw().read_ptr_on(exec);
                (operand_ptr(pc, cv.offset(), esz), gc, Some(gd))
            },
            None => (operand_ptr(pd, d_view.offset(), esz), gd, None),
        };
        let alpha_s = AlignedScale::new(alpha);
        let gamma_s = AlignedScale::new(gamma);
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer ordered on `exec`; the scalars live on the stack across the call.
        let status = unsafe {
            sys::cutensorElementwiseBinaryExecute(
                self.handle.ptr(),
                self.ptr,
                alpha_s.as_ptr().cast(),
                operand_ptr(pa, a.offset(), esz),
                gamma_s.as_ptr().cast(),
                ptr_c,
                operand_ptr(pd, d_view.offset(), esz) as *mut core::ffi::c_void,
                stream,
            )
        };
        drop((ga, gc, gd_keep));
        error::check(Stage::Execute, status, "")?;
        order_output(d_view.raw(), exec)?;
        Ok(())
    }

    /// Execute the planned elementwise trinary `D = op_ABC(op_AB(alpha * opA(A), beta * opB(B)),
    /// gamma * opC(C))` on `device`'s stream. Same re-validation discipline as
    /// [`Self::contract`]; `c` [`Option::None`] means **C is D**.
    #[allow(clippy::too_many_arguments)]
    pub fn elementwise_trinary<T: CutensorDType>(
        &self,
        device: &DeviceCuda,
        a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        b: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = IxD>>,
        mut d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = IxD>,
        alpha: T::Scalar,
        beta: T::Scalar,
        gamma: T::Scalar,
        workspace: Option<&CutensorWorkspace>,
    ) -> Result<()> {
        self.handle.bind()?;
        self.check_kind(OpKind::ElementwiseTrinary)?;
        self.check_scalar_type::<T>()?;
        self.check_workspace(workspace)?;
        let (a, b, d_view) = (a.view(), b.view(), d.view_mut());
        let c = c.as_ref().map(|v| v.view());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(b.device()), DeviceMismatch, "cutensor: operand B is on another device")?;
        rstsr_assert!(device.same_device(d_view.device()), DeviceMismatch, "cutensor: operand D is on another device")?;
        if let Some(cv) = c.as_ref() {
            rstsr_assert!(device.same_device(cv.device()), DeviceMismatch, "cutensor: operand C is on another device")?;
        }
        let operands = self.op.operands();
        if operands.len() != 4 {
            return Err(rstsr_error!(RuntimeError, "cutensor: execute: the plan is not an elementwise trinary"));
        }
        let (plan_a, plan_b, plan_c, plan_d) = (&operands[0], &operands[1], &operands[2], &operands[3]);
        let facts_a = view_facts::<T>(a.raw(), a.layout());
        validate_operand(plan_a, &facts_a, "A")?;
        let facts_b = view_facts::<T>(b.raw(), b.layout());
        validate_operand(plan_b, &facts_b, "B")?;
        if let Some(cv) = c.as_ref() {
            let facts_c = view_facts::<T>(cv.raw(), cv.layout());
            validate_operand(plan_c, &facts_c, "C")?;
        }
        let facts_d = view_facts::<T>(d_view.raw(), d_view.layout());
        validate_operand(plan_d, &facts_d, "D")?;
        let exec = device.stream();
        order_input(a.raw(), exec)?;
        order_input(b.raw(), exec)?;
        let esz = core::mem::size_of::<T>();
        let (pa, ga) = a.raw().read_ptr_on(exec);
        let (pb, gb) = b.raw().read_ptr_on(exec);
        let (pd, gd) = d_view.raw().read_ptr_on(exec);
        let (ptr_c, gc, gd_keep) = match c.as_ref() {
            Some(cv) => {
                order_input(cv.raw(), exec)?;
                let (pc, gc) = cv.raw().read_ptr_on(exec);
                (operand_ptr(pc, cv.offset(), esz), gc, Some(gd))
            },
            None => (operand_ptr(pd, d_view.offset(), esz), gd, None),
        };
        let alpha_s = AlignedScale::new(alpha);
        let beta_s = AlignedScale::new(beta);
        let gamma_s = AlignedScale::new(gamma);
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer ordered on `exec`; the scalars live on the stack across the call.
        let status = unsafe {
            sys::cutensorElementwiseTrinaryExecute(
                self.handle.ptr(),
                self.ptr,
                alpha_s.as_ptr().cast(),
                operand_ptr(pa, a.offset(), esz),
                beta_s.as_ptr().cast(),
                operand_ptr(pb, b.offset(), esz),
                gamma_s.as_ptr().cast(),
                ptr_c,
                operand_ptr(pd, d_view.offset(), esz) as *mut core::ffi::c_void,
                stream,
            )
        };
        drop((ga, gb, gc, gd_keep));
        error::check(Stage::Execute, status, "")?;
        order_output(d_view.raw(), exec)?;
        Ok(())
    }
}

/// The buffer pointer of one operand at its layout offset, as `void *`.
fn operand_ptr<T>(base: *const T, offset: usize, esz: usize) -> *const core::ffi::c_void {
    (base as usize).wrapping_add(offset.wrapping_mul(esz)) as *const core::ffi::c_void
}

/// The facts of one operand buffer + layout, for re-validation (works for views and mutable
/// views alike: both are `TensorBase` over a `CudaBuffer`).
fn view_facts<T: CutensorDType>(raw: &CudaBuffer<T>, layout: &Layout<IxD>) -> OperandFacts {
    let addr = raw.device_ptr() as usize + layout.offset() * core::mem::size_of::<T>();
    let shape: &[usize] = layout.shape().as_ref();
    let strides: &[isize] = layout.stride().as_ref();
    OperandFacts {
        extent: shape.iter().map(|&e| e as i64).collect(),
        stride: strides.iter().map(|&s| s as i64).collect(),
        dtype: T::DATA_TYPE,
        alignment: ptr_alignment(addr),
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: created by `cutensorCreatePlan`, destroyed exactly once (thread-safe per the
        // header).
        let status = unsafe { sys::cutensorDestroyPlan(self.ptr) };
        debug_assert_eq!(status, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
    }
}

impl core::fmt::Debug for Plan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("required_workspace", &self.required_workspace)
            .field("operation", &self.op)
            .finish_non_exhaustive()
    }
}
