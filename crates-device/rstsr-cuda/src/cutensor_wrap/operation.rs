//! Operation descriptors: one cuTENSOR operation (its operands, modes, operators and compute).

use crate::cutensor_wrap::descriptor::TensorDescriptor;
use crate::cutensor_wrap::dtype::{CutensorDType, CutensorScalar};
use crate::cutensor_wrap::error::{self, Stage};
use crate::cutensor_wrap::handle::CutensorHandle;
use crate::prelude_dev::*;
use cudarc::cutensor::sys;

/// cuTENSOR's per-operand mode limit.
const MAX_MODES: usize = 64;

/// The planned facts of one operand of an operation: what the descriptors said when the
/// operation (and its plan) was created. Execute-time re-validation compares the views against
/// these and refuses a mismatch — a plan is only valid for the problem it was planned for, and
/// nothing else would catch a different view against the same plan (the descriptors hold no
/// pointer; cuTENSOR itself does not check).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OperandPlan {
    pub extent: Vec<i64>,
    pub stride: Vec<i64>,
    pub dtype: sys::cudaDataType_t,
    pub alignment: u32,
}

/// One operand's facts at re-validation time, derived from the view.
pub(crate) struct OperandFacts {
    pub extent: Vec<i64>,
    pub stride: Vec<i64>,
    pub dtype: sys::cudaDataType_t,
    pub alignment: u32,
}

/// Compare one view's facts against its plan; `name` names the operand in the error.
pub(crate) fn validate_operand(plan: &OperandPlan, facts: &OperandFacts, name: &str) -> Result<()> {
    if facts.extent != plan.extent.as_slice() || facts.stride != plan.stride.as_slice() {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operand {name} does not match the planned layout (planned extent \
             {:?} stride {:?}, got extent {:?} stride {:?}); a plan only serves the layout it \
             was planned for",
            plan.extent,
            plan.stride,
            facts.extent,
            facts.stride
        ));
    }
    if facts.dtype != plan.dtype {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operand {name} has dtype {got:?}, planned for {want:?}",
            got = facts.dtype,
            want = plan.dtype
        ));
    }
    if facts.alignment < plan.alignment {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operand {name} has pointer alignment {} bytes, below the planned \
             declaration of {} bytes",
            facts.alignment,
            plan.alignment
        ));
    }
    Ok(())
}

/// One operand's mode array + extent array at operation creation.
type OperandModes<'a> = (&'a [i32], &'a [i64], &'a [i64]);

fn check_lengths(name: &str, modes: &[i32], extent: &[i64]) -> Result<()> {
    if modes.len() != extent.len() {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operation descriptor: operand {name} has {} modes but {} extents",
            modes.len(),
            extent.len()
        ));
    }
    if modes.len() > MAX_MODES {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operation descriptor: operand {name} has {} modes, above cuTENSOR's \
             limit of {MAX_MODES}",
            modes.len()
        ));
    }
    Ok(())
}

fn check_no_duplicates(name: &str, modes: &[i32]) -> Result<()> {
    for (i, m) in modes.iter().enumerate() {
        if modes[..i].contains(m) {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: mode {m} appears twice in operand {name} \
                 (describe a diagonal view instead: extent/stride of the repeated mode)"
            ));
        }
    }
    Ok(())
}

/// The contraction's operand-consistency checks, run **before** cuTENSOR is called.
///
/// No version of cuTENSOR validates these (the probe: an extent mismatch sails through create,
/// estimate and plan, and produces wrong numbers), so the wrapper is the authority:
/// - mode lengths match the descriptors, at most 64 modes per operand, no duplicate label within
///   one operand;
/// - `descC` must equal `descD` (cuTENSOR's own requirement) — same modes and extents;
/// - every D mode occurs in A or B;
/// - one extent per mode label across all operands;
/// - the output has no stride-0 mode with extent > 1 (garbage on every version probed).
fn check_contraction_operands(
    a: OperandModes<'_>,
    b: OperandModes<'_>,
    c: OperandModes<'_>,
    d: OperandModes<'_>,
) -> Result<()> {
    for (name, modes, extent) in [("A", a.0, a.1), ("B", b.0, b.1), ("C", c.0, c.1), ("D", d.0, d.1)] {
        check_lengths(name, modes, extent)?;
        check_no_duplicates(name, modes)?;
    }
    if c.0 != d.0 || c.1 != d.1 {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operation descriptor: descC must equal descD (cuTENSOR requirement); \
             got C modes {:?} extents {:?} vs D modes {:?} extents {:?}",
            c.0,
            c.1,
            d.0,
            d.1
        ));
    }
    for m in d.0 {
        if !a.0.contains(m) && !b.0.contains(m) {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: output mode {m} occurs in no input"
            ));
        }
    }
    let mut extents: std::collections::HashMap<i32, i64> = std::collections::HashMap::new();
    for (name, modes, extent) in [("A", a.0, a.1), ("B", b.0, b.1), ("C", c.0, c.1), ("D", d.0, d.1)] {
        for (&m, &e) in modes.iter().zip(extent.iter()) {
            match extents.entry(m) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(e);
                },
                std::collections::hash_map::Entry::Occupied(slot) => {
                    if *slot.get() != e {
                        return Err(rstsr_error!(
                            InvalidValue,
                            "cutensor: operation descriptor: mode {m} has extent {e} in \
                             operand {name} but {} elsewhere",
                            slot.get()
                        ));
                    }
                },
            }
        }
    }
    for (&m, &s) in d.0.iter().zip(d.2.iter()) {
        let e = d.1[d.0.iter().position(|&x| x == m).expect("mode in its own array")];
        if s == 0 && e > 1 {
            return Err(rstsr_error!(
                InvalidLayout,
                "cutensor: operation descriptor: output mode {m} has extent {e} and stride 0 \
                 (a broadcast output writes garbage on every cuTENSOR version)"
            ));
        }
    }
    Ok(())
}

/// One cuTENSOR operation: an RAII `cutensorOperationDescriptor_t` plus the operand facts the
/// wrapper planned it with (for execute-time re-validation).
///
/// Contraction's `opC` is pinned to `CUTENSOR_OP_IDENTITY` and has no parameter: no cuTENSOR
/// version honours a non-identity `opC` (silently ignored through 2.7.0, rejected by 2.8.0).
pub struct OperationDescriptor {
    ptr: sys::cutensorOperationDescriptor_t,
    /// Operand facts in cuTENSOR's operand order: `[A, B, C, D]` for a contraction.
    operands: Vec<OperandPlan>,
    /// The scalar type cuTENSOR derived from the compute descriptor (read back at creation).
    scalar_type: sys::cudaDataType_t,
}

impl OperationDescriptor {
    /// Create a contraction `D = alpha * opA(A) * opB(B) + beta * C`.
    ///
    /// The output dtype `T` fixes the host scalar type of `alpha` / `beta`
    /// (`f32` for half and `f32` outputs, `f64`, `Complex<f32>`, `Complex<f64>`); the compute
    /// descriptor decides the actual value, and the wrapper verifies cuTENSOR's derived
    /// `CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE` against it, closing the never-checked-pointer
    /// hole (`f32 x c32` passes creation but declines at plan; the scalar check runs regardless).
    ///
    /// `desc_c` and `mode_c` are required even for `beta == 0` — pass the output descriptor (the
    /// one-shot does); execute then passes the output pointer as `C`, which cuTENSOR does not
    /// read at `beta == 0`.
    // The raw pointers are the wrapper's own RAII descriptors, never caller-provided: the safe
    // wrapper premise is that every `unsafe` stays inside this module.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new_contraction<T: CutensorDType>(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: sys::cutensorOperator_t,
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
        op_b: sys::cutensorOperator_t,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        compute: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self> {
        check_contraction_operands(
            (mode_a, &desc_a.extent, &desc_a.stride),
            (mode_b, &desc_b.extent, &desc_b.stride),
            (mode_c, &desc_c.extent, &desc_c.stride),
            (mode_d, &desc_d.extent, &desc_d.stride),
        )?;
        if mode_c != mode_d || desc_c.plan() != desc_d.plan() {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: descC must equal descD (cuTENSOR requirement); \
                 C and D differ in modes or layout facts"
            ));
        }
        handle.bind()?;
        let _lock = handle.lock();
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorOperationDescriptor_t>::uninit();
        // SAFETY: all descriptors were created on this handle; the mode arrays live to the end
        // of the call.
        let status = unsafe {
            sys::cutensorCreateContraction(
                handle.ptr(),
                ptr.as_mut_ptr(),
                desc_a.ptr(),
                mode_a.as_ptr(),
                op_a,
                desc_b.ptr(),
                mode_b.as_ptr(),
                op_b,
                desc_c.ptr(),
                mode_c.as_ptr(),
                sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
                desc_d.ptr(),
                mode_d.as_ptr(),
                compute,
            )
        };
        if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            return Err(error::status_error(Stage::OperationDescriptor, status, ""));
        }
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        let scalar_type = read_scalar_type(handle, ptr)?;
        let want = T::Scalar::DATA_TYPE;
        if scalar_type != want {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: the compute descriptor implies scalar type \
                 {scalar_type:?} but the output dtype needs {want:?}; the host scalars would be \
                 misread (cuTENSOR never checks the pointer)"
            ));
        }
        Ok(Self { ptr, operands: vec![desc_a.plan(), desc_b.plan(), desc_c.plan(), desc_d.plan()], scalar_type })
    }

    /// The operand plans, in cuTENSOR's operand order.
    pub(crate) fn operands(&self) -> &[OperandPlan] {
        &self.operands
    }

    /// The scalar type the operation was verified against.
    pub(crate) fn scalar_type(&self) -> sys::cudaDataType_t {
        self.scalar_type
    }

    pub(crate) fn ptr(&self) -> sys::cutensorOperationDescriptor_t {
        self.ptr
    }
}

/// Read `CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE`; the caller holds the handle lock.
fn read_scalar_type(handle: &CutensorHandle, ptr: sys::cutensorOperationDescriptor_t) -> Result<sys::cudaDataType_t> {
    let mut scalar = sys::cudaDataType_t::CUDA_R_16F;
    // SAFETY: `ptr` came from a create call on this handle; the buffer is one `cudaDataType_t`.
    let status = unsafe {
        sys::cutensorOperationDescriptorGetAttribute(
            handle.ptr(),
            ptr,
            sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
            core::ptr::addr_of_mut!(scalar).cast(),
            core::mem::size_of::<sys::cudaDataType_t>(),
        )
    };
    if status != sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
        return Err(error::status_error(Stage::OperationDescriptor, status, "scalar type"));
    }
    Ok(scalar)
}

impl Drop for OperationDescriptor {
    fn drop(&mut self) {
        // SAFETY: created by a create call, destroyed exactly once (thread-safe per the header).
        let status = unsafe { sys::cutensorDestroyOperationDescriptor(self.ptr) };
        debug_assert_eq!(status, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
    }
}

impl core::fmt::Debug for OperationDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OperationDescriptor")
            .field("operands", &self.operands)
            .field("scalar_type", &self.scalar_type)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GEMM-shaped operand set passes.
    #[test]
    fn gemm_modes_pass() {
        check_contraction_operands(
            (&[0, 1], &[7, 13], &[13, 1]),
            (&[1, 2], &[13, 5], &[5, 1]),
            (&[0, 2], &[7, 5], &[5, 1]),
            (&[0, 2], &[7, 5], &[5, 1]),
        )
        .expect("ij,jk->ik");
    }

    /// The probe's refusal cases, reproduced as wrapper refusals.
    #[test]
    fn probe_refusals() {
        // probe2 D5a: a D mode in no input.
        let err = check_contraction_operands(
            (&[1], &[5], &[1]),
            (&[1, 3], &[5, 11], &[11, 1]),
            (&[0, 3], &[1, 11], &[11, 1]),
            (&[0, 3], &[1, 11], &[11, 1]),
        )
        .expect_err("mode 0 in no input");
        assert!(matches!(err.inner, RSTSRError::InvalidValue(_)));
        // probe2 D7: an extent mismatch on the contracted mode (cuTENSOR misses this one).
        let err = check_contraction_operands(
            (&[0, 1], &[7, 3], &[3, 1]),
            (&[1, 2], &[4, 5], &[5, 1]),
            (&[0, 2], &[7, 5], &[5, 1]),
            (&[0, 2], &[7, 5], &[5, 1]),
        )
        .expect_err("contracted extent mismatch");
        assert!(matches!(err.inner, RSTSRError::InvalidValue(_)));
        // descC != descD.
        let err = check_contraction_operands(
            (&[0, 1], &[7, 13], &[13, 1]),
            (&[1, 2], &[13, 5], &[5, 1]),
            (&[2, 0], &[5, 7], &[7, 1]),
            (&[0, 2], &[7, 5], &[5, 1]),
        )
        .expect_err("C != D");
        assert!(matches!(err.inner, RSTSRError::InvalidValue(_)));
        // a duplicate label within one operand.
        let err = check_contraction_operands(
            (&[0, 0], &[4, 4], &[4, 1]),
            (&[0, 1], &[4, 5], &[5, 1]),
            (&[1], &[5], &[1]),
            (&[1], &[5], &[1]),
        )
        .expect_err("duplicate mode");
        assert!(matches!(err.inner, RSTSRError::InvalidValue(_)));
        // a stride-0 output mode.
        let err = check_contraction_operands(
            (&[0, 1], &[7, 13], &[13, 1]),
            (&[1, 2], &[13, 5], &[5, 1]),
            (&[0, 2], &[7, 5], &[0, 1]),
            (&[0, 2], &[7, 5], &[0, 1]),
        )
        .expect_err("stride-0 output");
        assert!(matches!(err.inner, RSTSRError::InvalidLayout(_)));
        // over the mode limit.
        let modes: Vec<i32> = (0..65).collect();
        let extents: Vec<i64> = vec![2; 65];
        let strides: Vec<i64> = vec![1; 65];
        let err = check_contraction_operands(
            (&modes, &extents, &strides),
            (&[0, 1], &[2, 2], &[2, 1]),
            (&[1], &[2], &[1]),
            (&[1], &[2], &[1]),
        )
        .expect_err("65 modes");
        assert!(matches!(err.inner, RSTSRError::InvalidValue(_)));
    }

    /// Re-validation: matching facts pass; extent, stride, dtype and alignment mismatches refuse.
    #[test]
    fn operand_revalidation() {
        let plan = OperandPlan {
            extent: vec![7, 13],
            stride: vec![13, 1],
            dtype: sys::cudaDataType_t::CUDA_R_64F,
            alignment: 256,
        };
        let facts = |extent: &[i64], stride: &[i64], dtype, alignment: u32| OperandFacts {
            extent: extent.to_vec(),
            stride: stride.to_vec(),
            dtype,
            alignment,
        };
        assert!(validate_operand(&plan, &facts(&[7, 13], &[13, 1], sys::cudaDataType_t::CUDA_R_64F, 256), "A").is_ok());
        assert!(validate_operand(&plan, &facts(&[7, 13], &[1, 7], sys::cudaDataType_t::CUDA_R_64F, 256), "A").is_err());
        assert!(validate_operand(&plan, &facts(&[13, 7], &[13, 1], sys::cudaDataType_t::CUDA_R_64F, 256), "A").is_err());
        assert!(validate_operand(&plan, &facts(&[7, 13], &[13, 1], sys::cudaDataType_t::CUDA_R_32F, 256), "A").is_err());
        assert!(validate_operand(&plan, &facts(&[7, 13], &[13, 1], sys::cudaDataType_t::CUDA_R_64F, 8), "A").is_err());
    }
}
