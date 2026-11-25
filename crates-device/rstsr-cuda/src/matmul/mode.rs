//! Compute modes of the matmul drivers, keyed by the **output** type.
//!
//! One enum per output family ([`ComputeType32F`], [`ComputeType64F`], [`ComputeType16F`]) and
//! one flags field per family: an `f64` call never reads the 32F field, so a mode cannot be set
//! for a family the call does not use. [`CudaMatMulMode`] maps an output type to its family.

use crate::dtype::CudaDType;
use crate::flags::CudaRuntimeFlags;
use cudarc::cublas::sys;
use duplicate::duplicate_item;
use num::Complex;

/// 32-bit-float compute-type family (`cublasComputeType_t` names).
///
/// Read by calls with an `f32` (or `f32`-family) output only. A mode the type triple cannot serve
/// resolves to [`Self::Default`] silently; [`Self::Pedantic`] is never downgraded and is an error.
///
/// A driver that cannot express a mode declines the problem, and the next candidate runs
/// (`Fast16BF` has no classic-entry math mode, so the classic driver hands it on; the cuBLASLt
/// and `cublasGemmEx` drivers serve it as a compute type).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputeType32F {
    /// Default compute: best available performance, internal precision at
    /// least the requested one.
    #[default]
    Default,
    /// Exact arithmetic semantics and the specified internal storage format.
    Pedantic,
    /// Allow down-converting inputs to FP16 for faster tensor-core math.
    Fast16F,
    /// Allow down-converting inputs to BF16 for faster tensor-core math.
    Fast16BF,
    /// Allow down-converting inputs to TF32 for faster tensor-core math.
    FastTF32,
}

/// 64-bit-float compute-type family.
///
/// Read by calls with an `f64` output only, with the same resolution rule as [`ComputeType32F`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputeType64F {
    /// Default compute: best available performance.
    #[default]
    Default,
    /// Exact arithmetic semantics and the specified internal storage format.
    Pedantic,
    /// Emulate the `f64` product in fixed point. Only the classic cuBLAS driver serves it, and it
    /// needs `cudarc`'s CUDA 13.0 bindings (`CUDARC_CUDA_VERSION=13000` or newer); the mantissa
    /// width is [`CudaRuntimeFlags::fixedpoint_mantissa_bits`].
    EmulatedFixedPoint,
}

/// 16-bit-float output family (`f16`/`bf16` outputs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputeType16F {
    /// Default compute: accumulate in 32-bit floats.
    #[default]
    Default,
    /// Exact arithmetic semantics and the specified internal storage format.
    Pedantic,
    /// Opt-in accumulation in 16 bits (faster, less accurate).
    Acc16F,
}

/// An output type with a compute-mode family: [`Self::Mode`] is the enum the flags carry for it.
///
/// The integer types have no compute modes (`Mode = ()`): they only run on the naive driver.
pub trait CudaMatMulMode: CudaDType {
    /// The compute-mode enum of this output type's family.
    type Mode: Copy + core::fmt::Debug;

    /// The mode this output type resolves to under `flags`.
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode;
}

impl ComputeType32F {
    /// The `cublasGemmEx` compute type of this mode.
    pub(crate) fn gemm_ex(self) -> sys::cublasComputeType_t {
        use sys::cublasComputeType_t as C;
        match self {
            Self::Default => C::CUBLAS_COMPUTE_32F,
            Self::Pedantic => C::CUBLAS_COMPUTE_32F_PEDANTIC,
            Self::Fast16F => C::CUBLAS_COMPUTE_32F_FAST_16F,
            Self::Fast16BF => C::CUBLAS_COMPUTE_32F_FAST_16BF,
            Self::FastTF32 => C::CUBLAS_COMPUTE_32F_FAST_TF32,
        }
    }
}

impl ComputeType64F {
    /// The `cublasGemmEx` compute type of this mode, or `None` where the entry has none: the
    /// fixed-point emulation is a math mode of the classic handle only.
    pub(crate) fn gemm_ex(self) -> Option<sys::cublasComputeType_t> {
        use sys::cublasComputeType_t as C;
        match self {
            Self::Default => Some(C::CUBLAS_COMPUTE_64F),
            Self::Pedantic => Some(C::CUBLAS_COMPUTE_64F_PEDANTIC),
            Self::EmulatedFixedPoint => None,
        }
    }
}

impl CudaMatMulMode for f32 {
    type Mode = ComputeType32F;
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode {
        flags.compute_type_32f
    }
}

impl CudaMatMulMode for f64 {
    type Mode = ComputeType64F;
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode {
        flags.compute_type_64f
    }
}

impl CudaMatMulMode for Complex<f32> {
    type Mode = ComputeType32F;
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode {
        flags.compute_type_32f
    }
}

impl CudaMatMulMode for Complex<f64> {
    type Mode = ComputeType64F;
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode {
        flags.compute_type_64f
    }
}

impl CudaMatMulMode for half::f16 {
    type Mode = ComputeType16F;
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode {
        flags.compute_type_16f
    }
}

impl CudaMatMulMode for half::bf16 {
    type Mode = ComputeType16F;
    fn matmul_mode(flags: &CudaRuntimeFlags) -> Self::Mode {
        flags.compute_type_16f
    }
}

#[duplicate_item(
    T;
    [i8]; [i16]; [i32]; [i64]; [isize];
    [u8]; [u16]; [u32]; [u64]; [usize];
)]
impl CudaMatMulMode for T {
    type Mode = ();
    fn matmul_mode(_: &CudaRuntimeFlags) -> Self::Mode {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_32f_mode_is_a_gemm_ex_compute_type() {
        use sys::cublasComputeType_t as C;
        assert_eq!(ComputeType32F::default().gemm_ex(), C::CUBLAS_COMPUTE_32F);
        assert_eq!(ComputeType32F::Pedantic.gemm_ex(), C::CUBLAS_COMPUTE_32F_PEDANTIC);
        assert_eq!(ComputeType32F::Fast16F.gemm_ex(), C::CUBLAS_COMPUTE_32F_FAST_16F);
        assert_eq!(ComputeType32F::Fast16BF.gemm_ex(), C::CUBLAS_COMPUTE_32F_FAST_16BF);
        assert_eq!(ComputeType32F::FastTF32.gemm_ex(), C::CUBLAS_COMPUTE_32F_FAST_TF32);
    }

    #[test]
    fn fixed_point_emulation_has_no_gemm_ex_compute_type() {
        use sys::cublasComputeType_t as C;
        assert_eq!(ComputeType64F::default().gemm_ex(), Some(C::CUBLAS_COMPUTE_64F));
        assert_eq!(ComputeType64F::Pedantic.gemm_ex(), Some(C::CUBLAS_COMPUTE_64F_PEDANTIC));
        assert_eq!(ComputeType64F::EmulatedFixedPoint.gemm_ex(), None);
    }

    /// A call reads the field of its own family only.
    #[test]
    fn mode_read_from_the_family_field() {
        let flags = CudaRuntimeFlags {
            compute_type_32f: ComputeType32F::FastTF32,
            compute_type_64f: ComputeType64F::EmulatedFixedPoint,
            ..CudaRuntimeFlags::default()
        };
        assert_eq!(<f32 as CudaMatMulMode>::matmul_mode(&flags), ComputeType32F::FastTF32);
        assert_eq!(<f64 as CudaMatMulMode>::matmul_mode(&flags), ComputeType64F::EmulatedFixedPoint);
        assert_eq!(<Complex<f32> as CudaMatMulMode>::matmul_mode(&flags), ComputeType32F::FastTF32);
    }
}
