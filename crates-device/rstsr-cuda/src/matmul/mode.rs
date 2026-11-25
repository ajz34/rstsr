//! Compute modes of the matmul drivers, keyed by the **output** type.
//!
//! One enum per output family ([`ComputeType32F`], [`ComputeType64F`], [`ComputeType16F`]) and
//! one flags field per family: an `f64` call never reads the 32F field, so a mode cannot be set
//! for a family the call does not use. [`CudaMatMulMode`] maps an output type to its family.

use crate::dtype::CudaDType;
use crate::flags::CudaRuntimeFlags;
use duplicate::duplicate_item;
use num::Complex;

/// 32-bit-float compute-type family (`cublasComputeType_t` names).
///
/// Read by calls with an `f32` (or `f32`-family) output only. A mode the type triple cannot serve
/// resolves to [`Self::Default`] silently; [`Self::Pedantic`] is never downgraded and is an error.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputeType64F {
    /// Default compute: best available performance.
    #[default]
    Default,
    /// Exact arithmetic semantics and the specified internal storage format.
    Pedantic,
    /// Emulate the `f64` product in fixed point; the mantissa width is
    /// [`CudaRuntimeFlags::fixedpoint_mantissa_bits`].
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
