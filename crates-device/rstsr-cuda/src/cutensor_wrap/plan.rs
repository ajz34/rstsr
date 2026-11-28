//! Plan preference and plan: the planner (algorithm choice) and the reusable execution object.

use crate::cutensor_wrap::descriptor::ptr_alignment;
use crate::cutensor_wrap::dtype::{AlignedScale, CutensorDType, CutensorScalar};
use crate::cutensor_wrap::error::{self, Stage};
use crate::cutensor_wrap::handle::CutensorHandle;
use crate::cutensor_wrap::operation::{validate_operand, OperandFacts, OperationDescriptor};
use crate::cutensor_wrap::workspace::CutensorWorkspace;
use crate::kernel::{order_input, order_output};
use crate::prelude_dev::*;
use cudarc::cutensor::sys;
use std::sync::Arc;

/// The algorithm and JIT choice of a plan. Defaults live at the call sites
/// (`CUTENSOR_ALGO_DEFAULT`, `CUTENSOR_JIT_MODE_NONE`); the raw enums stay the API, per the
/// wrapper's raw-FFI boundary.
pub struct PlanPreference {
    ptr: sys::cutensorPlanPreference_t,
}

impl PlanPreference {
    /// Create a preference on `handle` (thread-safe per the header; no lock).
    pub fn new(handle: &CutensorHandle, algo: sys::cutensorAlgo_t, jit_mode: sys::cutensorJitMode_t) -> Result<Self> {
        handle.bind()?;
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorPlanPreference_t>::uninit();
        // SAFETY: `ptr` is a valid output slot for the one-pointer preference.
        let status = unsafe { sys::cutensorCreatePlanPreference(handle.ptr(), ptr.as_mut_ptr(), algo, jit_mode) };
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            return Err(error::status_error(Stage::PlanPreference, status, ""));
        }
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
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            return Err(error::status_error(Stage::WorkspaceEstimate, status, ""));
        }
        let _estimate = estimate;
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorPlan_t>::uninit();
        // SAFETY: as above; the limit is the caller's budget.
        let status =
            unsafe { sys::cutensorCreatePlan(handle.ptr(), ptr.as_mut_ptr(), op.ptr(), pref.ptr(), workspace_limit) };
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            return Err(error::status_error(Stage::Plan, status, ""));
        }
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
        let (a, b, d_view) = (a.view(), b.view(), d.view_mut());
        let c = c.as_ref().map(|v| v.view());
        rstsr_assert!(device.same_device(a.device()), DeviceMismatch, "cutensor: operand A is on another device")?;
        rstsr_assert!(device.same_device(b.device()), DeviceMismatch, "cutensor: operand B is on another device")?;
        rstsr_assert!(device.same_device(d_view.device()), DeviceMismatch, "cutensor: operand D is on another device")?;
        if let Some(cv) = c.as_ref() {
            rstsr_assert!(device.same_device(cv.device()), DeviceMismatch, "cutensor: operand C is on another device")?;
        }
        if TD::Scalar::DATA_TYPE != self.op.scalar_type() {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: execute: output scalar type {:?} does not match the planned {:?}",
                TD::Scalar::DATA_TYPE,
                self.op.scalar_type()
            ));
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
                order_input(cv.raw(), exec)?;
                let (pc, gc) = cv.raw().read_ptr_on(exec);
                (operand_ptr(pc, cv.offset(), esz_d), gc, Some(gd))
            },
            None => (ptr_d, gd, None),
        };
        let alpha_s = AlignedScale::new(alpha);
        let beta_s = AlignedScale::new(beta);
        let (ws_ptr, ws_len) = match workspace {
            Some(ws) => (ws.ptr_on(exec), ws.len() as u64),
            None => (core::ptr::null_mut(), 0u64),
        };
        let stream = exec.inner().cu_stream().cast::<core::ffi::c_void>() as sys::cudaStream_t;
        // SAFETY: the plan was created on this handle for this device; every pointer is a live
        // device buffer whose ordering events are recorded on `exec`; the scalars live on the
        // stack across the call; the stream belongs to the device the plan is bound to.
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
        drop((ga, gb, gc, gd_keep));
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            return Err(error::status_error(Stage::Execute, status, ""));
        }
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
