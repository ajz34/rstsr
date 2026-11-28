//! Error mapping from cuTENSOR statuses to RSTSR errors.

use cudarc::cutensor::sys;
use rstsr_core::prelude_dev::*;

/// The stage of the cuTENSOR pipeline an error occurred in; names the error message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `libcutensor` is not loadable.
    LoadLibrary,
    /// `cutensorCreate`.
    CreateHandle,
    /// `cutensorCreateTensorDescriptor`.
    TensorDescriptor,
    /// `cutensorCreate{Contraction,ContractionTrinary,Reduction,Permutation,Elementwise*}`.
    OperationDescriptor,
    /// `cutensorCreatePlanPreference`.
    PlanPreference,
    /// `cutensorEstimateWorkspaceSize`.
    WorkspaceEstimate,
    /// `cutensorCreatePlan`.
    Plan,
    /// `cutensorContract` / `cutensorReduce` / `cutensorPermute` / `cutensorElementwise*Execute`.
    Execute,
}

impl Stage {
    /// The stage as it appears in error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::LoadLibrary => "load library",
            Stage::CreateHandle => "create handle",
            Stage::TensorDescriptor => "tensor descriptor",
            Stage::OperationDescriptor => "operation descriptor",
            Stage::PlanPreference => "plan preference",
            Stage::WorkspaceEstimate => "workspace estimate",
            Stage::Plan => "plan",
            Stage::Execute => "execute",
        }
    }
}

/// The "libcutensor is not loadable" error (an error, never cudarc's panic).
pub fn not_loadable() -> Error {
    rstsr_error!(
        UnImplemented,
        "cutensor: load library: libcutensor is not loadable (the pip wheel ships only \
         `libcutensor.so.2`, which cudarc does not search for; create a `libcutensor.so` symlink \
         to it in a directory on LD_LIBRARY_PATH)"
    )
}

/// The "cuTENSOR is not usable on this device" error for the entry points that resolve the
/// per-ordinal handle: the library did not load, or its handle could not be created.
pub fn unavailable() -> Error {
    rstsr_error!(
        UnImplemented,
        "cutensor: load library: cuTENSOR is not available on this device (libcutensor did not \
         load, or cutensorCreate failed on it)"
    )
}

/// Map a cuTENSOR status to an RSTSR error, naming the stage, the library's own error string and
/// the caller's context. The variant carries the meaning:
///
/// | status | variant |
/// |---|---|
/// | `NOT_SUPPORTED` | `UnImplemented` (a decline; try the next tier) |
/// | `INVALID_VALUE` | `InvalidValue` |
/// | `ALLOC_FAILED`, `INSUFFICIENT_WORKSPACE` | `MemoryError` |
/// | `CUDA_ERROR`, `CUBLAS_ERROR`, `EXECUTION_FAILED`, `ARCH_MISMATCH`, `INSUFFICIENT_DRIVER` | `DeviceError` |
/// | everything else | `RuntimeError` |
pub fn status_error(stage: Stage, status: sys::cutensorStatus_t, ctx: &str) -> Error {
    let ctx = if ctx.is_empty() { String::new() } else { format!("; {ctx}") };
    let msg = format!("cutensor: {}: {} ({status:?}){ctx}", stage.as_str(), error_string(status));
    status_variant(status, msg)
}

/// The variant half of [`status_error`], factored out so the table is unit-testable without the
/// library (`cutensorGetErrorString` is FFI).
fn status_variant(status: sys::cutensorStatus_t, msg: String) -> Error {
    match status {
        sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_SUPPORTED => rstsr_error!(UnImplemented, "{msg}"),
        sys::cutensorStatus_t::CUTENSOR_STATUS_INVALID_VALUE => rstsr_error!(InvalidValue, "{msg}"),
        sys::cutensorStatus_t::CUTENSOR_STATUS_ALLOC_FAILED
        | sys::cutensorStatus_t::CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE => {
            rstsr_error!(MemoryError, "{msg}")
        },
        sys::cutensorStatus_t::CUTENSOR_STATUS_CUDA_ERROR
        | sys::cutensorStatus_t::CUTENSOR_STATUS_CUBLAS_ERROR
        | sys::cutensorStatus_t::CUTENSOR_STATUS_EXECUTION_FAILED
        | sys::cutensorStatus_t::CUTENSOR_STATUS_ARCH_MISMATCH
        | sys::cutensorStatus_t::CUTENSOR_STATUS_INSUFFICIENT_DRIVER => {
            rstsr_error!(DeviceError, "{msg}")
        },
        _ => rstsr_error!(RuntimeError, "{msg}"),
    }
}

/// The library's own description of `status`, or the status name when the library is not
/// loadable (cudarc's loader panics there; this must never).
fn error_string(status: sys::cutensorStatus_t) -> String {
    if !crate::cutensor_wrap::handle::libcutensor_available() {
        return format!("cuTENSOR status {status:?}");
    }
    // SAFETY: returns a static NUL-terminated string owned by the library; the library is loaded.
    let ptr = unsafe { sys::cutensorGetErrorString(status) };
    if ptr.is_null() {
        return format!("cuTENSOR status {status:?}");
    }
    // SAFETY: the pointer is NUL-terminated for the lifetime of the process.
    unsafe { core::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

/// Whether `err` means "this call is not served" (`true`) rather than "something went wrong":
/// `true` for `UnImplemented`, `InvalidValue` and `InvalidLayout` — the variants a decline uses
/// (a refused operand shape, a library that cannot load, a `NOT_SUPPORTED` status). A caller with
/// a fallback (the einsum tiers) checks this instead of matching messages.
pub fn is_decline(err: &Error) -> bool {
    matches!(&err.inner, RSTSRError::UnImplemented(_) | RSTSRError::InvalidValue(_) | RSTSRError::InvalidLayout(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variant_of(err: &Error) -> &'static str {
        match &err.inner {
            RSTSRError::ValueOutOfRange(_) => "ValueOutOfRange",
            RSTSRError::InvalidValue(_) => "InvalidValue",
            RSTSRError::InvalidLayout(_) => "InvalidLayout",
            RSTSRError::RuntimeError(_) => "RuntimeError",
            RSTSRError::DeviceMismatch(_) => "DeviceMismatch",
            RSTSRError::UnImplemented(_) => "UnImplemented",
            RSTSRError::MemoryError(_) => "MemoryError",
            RSTSRError::DeviceError(_) => "DeviceError",
            _ => "other",
        }
    }

    /// Every status maps to its documented variant.
    #[test]
    fn status_table() {
        use sys::cutensorStatus_t as S;
        let cases = [
            (S::CUTENSOR_STATUS_NOT_SUPPORTED, "UnImplemented"),
            (S::CUTENSOR_STATUS_INVALID_VALUE, "InvalidValue"),
            (S::CUTENSOR_STATUS_ALLOC_FAILED, "MemoryError"),
            (S::CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE, "MemoryError"),
            (S::CUTENSOR_STATUS_CUDA_ERROR, "DeviceError"),
            (S::CUTENSOR_STATUS_CUBLAS_ERROR, "DeviceError"),
            (S::CUTENSOR_STATUS_EXECUTION_FAILED, "DeviceError"),
            (S::CUTENSOR_STATUS_ARCH_MISMATCH, "DeviceError"),
            (S::CUTENSOR_STATUS_INSUFFICIENT_DRIVER, "DeviceError"),
            (S::CUTENSOR_STATUS_NOT_INITIALIZED, "RuntimeError"),
            (S::CUTENSOR_STATUS_MAPPING_ERROR, "RuntimeError"),
            (S::CUTENSOR_STATUS_INTERNAL_ERROR, "RuntimeError"),
            (S::CUTENSOR_STATUS_LICENSE_ERROR, "RuntimeError"),
            (S::CUTENSOR_STATUS_IO_ERROR, "RuntimeError"),
        ];
        for (status, want) in cases {
            let err = status_variant(status, "msg".into());
            assert_eq!(variant_of(&err), want, "status {status:?}");
        }
    }

    /// The message names the stage, the library string and the context.
    #[test]
    fn message_names_stage_and_context() {
        let err =
            status_error(Stage::Plan, sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_SUPPORTED, "ij,jk->ik f32 x c32");
        let text = err.to_string();
        assert!(text.contains("cutensor: plan:"), "{text}");
        assert!(text.contains("CUTENSOR_STATUS_NOT_SUPPORTED"), "{text}");
        assert!(text.contains("ij,jk->ik f32 x c32"), "{text}");
    }

    /// Declines are exactly the three variants a fallback tier can retry past.
    #[test]
    fn decline_by_variant() {
        assert!(is_decline(&status_error(Stage::Plan, sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_SUPPORTED, "")));
        assert!(is_decline(&rstsr_error!(InvalidValue, "bad mode length")));
        assert!(is_decline(&rstsr_error!(InvalidLayout, "stride-0 output")));
        assert!(is_decline(&not_loadable()));
        assert!(!is_decline(&status_error(
            Stage::Execute,
            sys::cutensorStatus_t::CUTENSOR_STATUS_EXECUTION_FAILED,
            ""
        )));
        assert!(!is_decline(&status_error(Stage::Plan, sys::cutensorStatus_t::CUTENSOR_STATUS_ALLOC_FAILED, "")));
        assert!(!is_decline(&rstsr_error!(DeviceMismatch, "cross-device")));
        assert!(!is_decline(&rstsr_error!(MemoryError, "oom")));
    }

    /// All stages render distinctly.
    #[test]
    fn stage_strings_are_distinct() {
        let stages = [
            Stage::LoadLibrary,
            Stage::CreateHandle,
            Stage::TensorDescriptor,
            Stage::OperationDescriptor,
            Stage::PlanPreference,
            Stage::WorkspaceEstimate,
            Stage::Plan,
            Stage::Execute,
        ];
        let strings: Vec<_> = stages.iter().map(|s| s.as_str()).collect();
        for (i, a) in strings.iter().enumerate() {
            for b in &strings[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
