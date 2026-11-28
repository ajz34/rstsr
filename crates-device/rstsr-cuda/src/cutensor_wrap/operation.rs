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
/// No version of cuTENSOR validates these (an extent mismatch sails through create, estimate and
/// plan and produces wrong numbers), so the wrapper is the authority:
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
    check_extent_consistency([("A", a.0, a.1), ("B", b.0, b.1), ("C", c.0, c.1), ("D", d.0, d.1)])?;
    check_output_strides(d.0, d.1, d.2)
}

/// `desc_<a> == desc_<b>` by full facts: the same mode labels and the same descriptor facts
/// (extent, stride, dtype, alignment) — cuTENSOR's requirement for the `C`/`D` (and `D`/`E`)
/// pairs, checked before the call.
fn check_same_operand(
    name_a: &str,
    desc_a: &TensorDescriptor,
    mode_a: &[i32],
    name_b: &str,
    desc_b: &TensorDescriptor,
    mode_b: &[i32],
) -> Result<()> {
    if mode_a != mode_b || desc_a.plan() != desc_b.plan() {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operation descriptor: desc{name_a} must equal desc{name_b} (cuTENSOR \
             requirement); {name_a} and {name_b} differ in modes or layout facts"
        ));
    }
    Ok(())
}

/// The pre-FFI checks of one operand: mode/extent lengths, the mode limit, no duplicate labels.
fn check_operand(name: &str, modes: &[i32], extent: &[i64]) -> Result<()> {
    check_lengths(name, modes, extent)?;
    check_no_duplicates(name, modes)
}

/// One extent per mode label across all operands (cuTENSOR misses the mismatch itself).
fn check_extent_consistency<'a>(operands: impl IntoIterator<Item = (&'a str, &'a [i32], &'a [i64])>) -> Result<()> {
    let mut extents: std::collections::HashMap<i32, i64> = std::collections::HashMap::new();
    for (name, modes, extent) in operands {
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
    Ok(())
}

/// The output must have no stride-0 mode with extent > 1 (garbage on every version probed).
fn check_output_strides(modes: &[i32], extent: &[i64], stride: &[i64]) -> Result<()> {
    for (&m, &s) in modes.iter().zip(stride.iter()) {
        let e = extent[modes.iter().position(|&x| x == m).expect("mode in its own array")];
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

/// The trinary contraction's operand checks: `E = alpha*opA(A)*opB(B)*opC(C) + beta*opD(D)`.
/// `descD == descE` (the accumulator's descriptor, cuTENSOR's own requirement); every output
/// mode occurs in an input; the rest as for a contraction.
fn check_contraction_trinary_operands(
    a: OperandModes<'_>,
    b: OperandModes<'_>,
    c: OperandModes<'_>,
    d: OperandModes<'_>,
    e: OperandModes<'_>,
) -> Result<()> {
    check_operand("A", a.0, a.1)?;
    check_operand("B", b.0, b.1)?;
    check_operand("C", c.0, c.1)?;
    check_operand("D", d.0, d.1)?;
    check_operand("E", e.0, e.1)?;
    if d.0 != e.0 || d.1 != e.1 {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operation descriptor: descD must equal descE (cuTENSOR requirement); \
             got D modes {:?} extents {:?} vs E modes {:?} extents {:?}",
            d.0,
            d.1,
            e.0,
            e.1
        ));
    }
    for m in e.0 {
        if !a.0.contains(m) && !b.0.contains(m) && !c.0.contains(m) && !d.0.contains(m) {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: output mode {m} occurs in no input"
            ));
        }
    }
    check_extent_consistency([("A", a.0, a.1), ("B", b.0, b.1), ("C", c.0, c.1), ("D", d.0, d.1), ("E", e.0, e.1)])?;
    check_output_strides(e.0, e.1, e.2)
}

/// The reduction's operand checks: `D = alpha * opReduce(opA(A)) + beta * C`. descC == descD
/// (header requirement); every output mode is kept from A (the other A modes are reduced); the
/// rest as for a contraction.
fn check_reduction_operands(a: OperandModes<'_>, c: OperandModes<'_>, d: OperandModes<'_>) -> Result<()> {
    check_operand("A", a.0, a.1)?;
    check_operand("C", c.0, c.1)?;
    check_operand("D", d.0, d.1)?;
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
        if !a.0.contains(m) {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: output mode {m} occurs in no input"
            ));
        }
    }
    check_extent_consistency([("A", a.0, a.1), ("C", c.0, c.1), ("D", d.0, d.1)])?;
    check_output_strides(d.0, d.1, d.2)
}

/// The permutation's operand checks: `B = alpha * opA(A)` over permuted mode labels. A and B
/// have the same rank and the same mode *set* (in any order — the differing order is the
/// permutation), with one extent per label.
fn check_permutation_operands(a: OperandModes<'_>, b: OperandModes<'_>) -> Result<()> {
    check_operand("A", a.0, a.1)?;
    check_operand("B", b.0, b.1)?;
    let mut sa: Vec<&i32> = a.0.iter().collect();
    let mut sb: Vec<&i32> = b.0.iter().collect();
    sa.sort();
    sb.sort();
    if sa != sb {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: operation descriptor: permutation operands must share one mode set \
             (sorted); got A {:?} vs B {:?}",
            a.0,
            b.0
        ));
    }
    check_extent_consistency([("A", a.0, a.1), ("B", b.0, b.1)])?;
    check_output_strides(b.0, b.1, b.2)
}

/// The elementwise operands' checks (binary: `[a, c]`; trinary: `[a, b, c]`): every input's mode
/// *set* equals the output's (elementwise has no reduction; broadcast goes through stride-0
/// *views*, which the descriptor accepts from 2.6.0), plus the contraction-standard checks.
fn check_elementwise_operands<'a>(
    inputs: impl IntoIterator<Item = (&'a str, OperandModes<'a>)>,
    d: OperandModes<'_>,
) -> Result<()> {
    check_operand("D", d.0, d.1)?;
    let mut set_d: Vec<&i32> = d.0.iter().collect();
    set_d.sort_unstable();
    let mut names = Vec::new();
    for (name, om) in inputs {
        check_operand(name, om.0, om.1)?;
        let mut set: Vec<&i32> = om.0.iter().collect();
        set.sort_unstable();
        if set != set_d {
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: elementwise operand {name} must have the \
                 output's mode set (sorted); got {:?} vs D {:?}",
                om.0,
                d.0
            ));
        }
        names.push((name, om.0, om.1));
    }
    names.push(("D", d.0, d.1));
    check_extent_consistency(names)?;
    check_output_strides(d.0, d.1, d.2)
}

/// Which family an operation descriptor encodes (the execute calls dispatch on this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpKind {
    Contraction,
    ContractionTrinary,
    Reduction,
    Permutation,
    ElementwiseBinary,
    ElementwiseTrinary,
}

/// One cuTENSOR operation: an RAII `cutensorOperationDescriptor_t` plus the operand facts the
/// wrapper planned it with (for execute-time re-validation).
///
/// Contraction's `opC` is pinned to `CUTENSOR_OP_IDENTITY` and has no parameter: no cuTENSOR
/// version honours a non-identity `opC` (silently ignored through 2.7.0, rejected by 2.8.0).
pub struct OperationDescriptor {
    ptr: sys::cutensorOperationDescriptor_t,
    /// Which family this operation encodes (the execute calls dispatch on it).
    kind: OpKind,
    /// Operand facts in cuTENSOR's operand order: `[A, B, C, D]` for a contraction or trinary
    /// elementwise, `[A, C, D]` for a reduction or binary elementwise, `[A, B]` for a permutation.
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
        check_same_operand("C", desc_c, mode_c, "D", desc_d, mode_d)?;
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
        error::check(Stage::OperationDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Self::finish::<T>(OpKind::Contraction, handle, ptr, vec![
            desc_a.plan(),
            desc_b.plan(),
            desc_c.plan(),
            desc_d.plan(),
        ])
    }

    /// Create a trinary contraction `E = alpha * opA(A) * opB(B) * opC(C) + beta * opD(D)`:
    /// three contracted inputs accumulated with `D`. `opD` is pinned to `IDENTITY` (no
    /// parameter) — no cuTENSOR version honours a non-identity unary on the accumulate side,
    /// the same reasoning as contraction's `opC`; `descD == descE` is cuTENSOR's own requirement.
    ///
    /// Gated: the trinary contraction arrived in libcutensor 2.1.1; older libraries decline
    /// with `UnImplemented` (cudarc's own `load` panics on a missing symbol, so the wrapper
    /// probes first). Some dtype rows need Ampere+; cuTENSOR declines those itself at plan.
    // The raw pointers are the wrapper's own RAII descriptors, never caller-provided.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new_contraction_trinary<T: CutensorDType>(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: sys::cutensorOperator_t,
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
        op_b: sys::cutensorOperator_t,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        op_c: sys::cutensorOperator_t,
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        desc_e: &TensorDescriptor,
        mode_e: &[i32],
        compute: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self> {
        if !handle.has_symbol(b"cutensorCreateContractionTrinary\0") {
            return Err(rstsr_error!(UnImplemented, "cutensor: trinary contraction requires libcutensor >= 2.1.1"));
        }
        check_contraction_trinary_operands(
            (mode_a, &desc_a.extent, &desc_a.stride),
            (mode_b, &desc_b.extent, &desc_b.stride),
            (mode_c, &desc_c.extent, &desc_c.stride),
            (mode_d, &desc_d.extent, &desc_d.stride),
            (mode_e, &desc_e.extent, &desc_e.stride),
        )?;
        check_same_operand("D", desc_d, mode_d, "E", desc_e, mode_e)?;
        handle.bind()?;
        let _lock = handle.lock();
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorOperationDescriptor_t>::uninit();
        // SAFETY: all descriptors were created on this handle; the mode arrays live to the end
        // of the call; the symbol was probed above.
        let status = unsafe {
            sys::cutensorCreateContractionTrinary(
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
                op_c,
                desc_d.ptr(),
                mode_d.as_ptr(),
                sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
                desc_e.ptr(),
                mode_e.as_ptr(),
                compute,
            )
        };
        error::check(Stage::OperationDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Self::finish::<T>(OpKind::ContractionTrinary, handle, ptr, vec![
            desc_a.plan(),
            desc_b.plan(),
            desc_c.plan(),
            desc_d.plan(),
            desc_e.plan(),
        ])
    }

    /// Verify the derived scalar type and assemble (the shared constructor tail).
    fn finish<T: CutensorDType>(
        kind: OpKind,
        handle: &CutensorHandle,
        ptr: sys::cutensorOperationDescriptor_t,
        operands: Vec<OperandPlan>,
    ) -> Result<Self> {
        // On a failed attribute read the struct (and its `Drop`) is never built: destroy here.
        let scalar_type = read_scalar_type(handle, ptr).inspect_err(|_| {
            // SAFETY: `ptr` came from a create call on this handle; destroyed exactly once.
            let destroy = unsafe { sys::cutensorDestroyOperationDescriptor(ptr) };
            debug_assert_eq!(destroy, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
        })?;
        let this = Self { ptr, kind, operands, scalar_type };
        let want = T::Scalar::DATA_TYPE;
        if scalar_type != want {
            // `this` drops on return, destroying the descriptor.
            return Err(rstsr_error!(
                InvalidValue,
                "cutensor: operation descriptor: the compute descriptor implies scalar type \
                 {scalar_type:?} but the output dtype needs {want:?}; the host scalars would be \
                 misread (cuTENSOR never checks the pointer)"
            ));
        }
        Ok(this)
    }

    /// Which family this operation encodes.
    pub(crate) fn kind(&self) -> OpKind {
        self.kind
    }

    /// The operand plans, in cuTENSOR's operand order.
    pub(crate) fn operands(&self) -> &[OperandPlan] {
        &self.operands
    }

    /// Create a reduction `D = alpha * opReduce(opA(A)) + beta * C` (the modes of `A` absent
    /// from `D` are the reduced ones). The output dtype `T` fixes the host scalar type, as in
    /// [`Self::new_contraction`]; `descC == descD` is cuTENSOR's own requirement.
    // The raw pointers are the wrapper's own RAII descriptors, never caller-provided.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new_reduction<T: CutensorDType>(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: sys::cutensorOperator_t,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        op_reduce: sys::cutensorOperator_t,
        compute: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self> {
        check_reduction_operands(
            (mode_a, &desc_a.extent, &desc_a.stride),
            (mode_c, &desc_c.extent, &desc_c.stride),
            (mode_d, &desc_d.extent, &desc_d.stride),
        )?;
        check_same_operand("C", desc_c, mode_c, "D", desc_d, mode_d)?;
        handle.bind()?;
        let _lock = handle.lock();
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorOperationDescriptor_t>::uninit();
        // SAFETY: all descriptors were created on this handle; the mode arrays live to the end
        // of the call.
        let status = unsafe {
            sys::cutensorCreateReduction(
                handle.ptr(),
                ptr.as_mut_ptr(),
                desc_a.ptr(),
                mode_a.as_ptr(),
                op_a,
                desc_c.ptr(),
                mode_c.as_ptr(),
                sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
                desc_d.ptr(),
                mode_d.as_ptr(),
                op_reduce,
                compute,
            )
        };
        error::check(Stage::OperationDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Self::finish::<T>(OpKind::Reduction, handle, ptr, vec![desc_a.plan(), desc_c.plan(), desc_d.plan()])
    }

    /// Create a permutation `B = alpha * opA(A)` over permuted mode labels; `mode_b` is the
    /// output's label order (the differing order *is* the permutation). The wrapper serves the
    /// same-dtype path; dtype-converting permutations are cuTENSOR's table and unexplored here.
    // The raw pointers are the wrapper's own RAII descriptors, never caller-provided.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new_permutation<T: CutensorDType>(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: sys::cutensorOperator_t,
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
        compute: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self> {
        check_permutation_operands((mode_a, &desc_a.extent, &desc_a.stride), (mode_b, &desc_b.extent, &desc_b.stride))?;
        handle.bind()?;
        let _lock = handle.lock();
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorOperationDescriptor_t>::uninit();
        // SAFETY: all descriptors were created on this handle; the mode arrays live to the end
        // of the call.
        let status = unsafe {
            sys::cutensorCreatePermutation(
                handle.ptr(),
                ptr.as_mut_ptr(),
                desc_a.ptr(),
                mode_a.as_ptr(),
                op_a,
                desc_b.ptr(),
                mode_b.as_ptr(),
                compute,
            )
        };
        error::check(Stage::OperationDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Self::finish::<T>(OpKind::Permutation, handle, ptr, vec![desc_a.plan(), desc_b.plan()])
    }

    /// Create an elementwise binary `D = op_AC(alpha * opA(A), gamma * opC(C))` — `op_ac` is the
    /// *functional* join operator (`ADD`/`MUL`/`MAX`/...), unlike contraction's `opC`, which the
    /// wrapper pins to `IDENTITY` and does not take as a parameter. `descC == descD` is
    /// cuTENSOR's own requirement.
    // The raw pointers are the wrapper's own RAII descriptors, never caller-provided.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new_elementwise_binary<T: CutensorDType>(
        handle: &CutensorHandle,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: sys::cutensorOperator_t,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        op_ac: sys::cutensorOperator_t,
        compute: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self> {
        check_elementwise_operands(
            [("A", (mode_a, desc_a.extent.as_slice(), desc_a.stride.as_slice()))],
            (mode_d, &desc_d.extent, &desc_d.stride),
        )?;
        check_same_operand("C", desc_c, mode_c, "D", desc_d, mode_d)?;
        handle.bind()?;
        let _lock = handle.lock();
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorOperationDescriptor_t>::uninit();
        // SAFETY: all descriptors were created on this handle; the mode arrays live to the end
        // of the call.
        let status = unsafe {
            sys::cutensorCreateElementwiseBinary(
                handle.ptr(),
                ptr.as_mut_ptr(),
                desc_a.ptr(),
                mode_a.as_ptr(),
                op_a,
                desc_c.ptr(),
                mode_c.as_ptr(),
                sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
                desc_d.ptr(),
                mode_d.as_ptr(),
                op_ac,
                compute,
            )
        };
        error::check(Stage::OperationDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Self::finish::<T>(OpKind::ElementwiseBinary, handle, ptr, vec![desc_a.plan(), desc_c.plan(), desc_d.plan()])
    }

    /// Create an elementwise trinary `D = op_ABC(op_AB(alpha * opA(A), beta * opB(B)),
    /// gamma * opC(C))`: `op_ab` joins `A` and `B`, `op_abc` joins that with `C`.
    /// `descC == descD` is cuTENSOR's own requirement.
    // The raw pointers are the wrapper's own RAII descriptors, never caller-provided.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new_elementwise_trinary<T: CutensorDType>(
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
        op_ab: sys::cutensorOperator_t,
        op_abc: sys::cutensorOperator_t,
        compute: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self> {
        check_elementwise_operands(
            [
                ("A", (mode_a, desc_a.extent.as_slice(), desc_a.stride.as_slice())),
                ("B", (mode_b, desc_b.extent.as_slice(), desc_b.stride.as_slice())),
            ],
            (mode_d, &desc_d.extent, &desc_d.stride),
        )?;
        check_same_operand("C", desc_c, mode_c, "D", desc_d, mode_d)?;
        handle.bind()?;
        let _lock = handle.lock();
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorOperationDescriptor_t>::uninit();
        // SAFETY: all descriptors were created on this handle; the mode arrays live to the end
        // of the call.
        let status = unsafe {
            sys::cutensorCreateElementwiseTrinary(
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
                op_ab,
                op_abc,
                compute,
            )
        };
        error::check(Stage::OperationDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Self::finish::<T>(OpKind::ElementwiseTrinary, handle, ptr, vec![
            desc_a.plan(),
            desc_b.plan(),
            desc_c.plan(),
            desc_d.plan(),
        ])
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
    error::check(Stage::OperationDescriptor, status, "scalar type")?;
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
            .field("kind", &self.kind)
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

    /// The pre-FFI refusals of the contraction checks, by variant.
    #[test]
    fn contraction_refusals() {
        // an output mode in no input.
        let err = check_contraction_operands(
            (&[1], &[5], &[1]),
            (&[1, 3], &[5, 11], &[11, 1]),
            (&[0, 3], &[1, 11], &[11, 1]),
            (&[0, 3], &[1, 11], &[11, 1]),
        )
        .expect_err("mode 0 in no input");
        assert!(matches!(err.inner, RSTSRError::InvalidValue(_)));
        // an extent mismatch on the contracted mode (cuTENSOR itself misses this one).
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
