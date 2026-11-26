//! Runtime flags for the CUDA device.

use crate::matmul::{ComputeType16F, ComputeType64F, MatMulDrivers, MatMulUseSyrk};
use rstsr_core::prelude_dev::*;
use std::sync::RwLock;

/// Old path of [`ComputeType32F`], which now lives in [`crate::matmul`].
pub use crate::matmul::ComputeType32F;

/// Runtime flags of a CUDA device (additive; keep small and `Copy`).
///
/// Two layers:
/// - a process-global default, read by [`default_flags`] and writable via [`set_default_flags`];
/// - a per-device `Option<CudaRuntimeFlags>` override: `None` tracks the global, `Some(x)` pins
///   ([`DeviceCuda::with_flags`][crate::device::DeviceCuda::with_flags]).
///
/// Flags are resolved at launch time by flag-consuming ops, but compared as
/// stored (see `DeviceCuda::same_device`). No flag is seeded from the environment: op behaviour is
/// set programmatically; only the kernel-build inputs (arch, strict-fp, include paths, cache
/// location) are environment variables, because they are part of the PTX cache key.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct CudaRuntimeFlags {
    /// Compute type for 32-bit-float matmul (read by `f32`-output calls only).
    pub compute_type_32f: ComputeType32F,
    /// Compute type for 64-bit-float matmul (read by `f64`-output calls only).
    pub compute_type_64f: ComputeType64F,
    /// Compute type for 16-bit-float matmul (read by `f16`/`bf16`-output calls only).
    pub compute_type_16f: ComputeType16F,
    /// Mantissa bits of the 64-bit-float fixed-point emulation; `0` keeps the library
    /// default (the only value `cudarc` 0.19 can honour: any other is an error naming this
    /// field). Read only when `compute_type_64f` is `EmulatedFixedPoint`.
    pub fixedpoint_mantissa_bits: u32,
    /// The matmul drivers a device may use (default: all of them).
    pub matmul_drivers: MatMulDrivers,
    /// When `a @ a.t()` runs as a symmetric rank-k update (default: [`MatMulUseSyrk::Scientific`]).
    pub matmul_use_syrk: MatMulUseSyrk,
    /// Print one line per matmul launch: the triple, the shape, the mode and the driver tried.
    pub trace_matmul: bool,
}

static CUDA_FLAGS_DEFAULT: RwLock<CudaRuntimeFlags> = RwLock::new(CudaRuntimeFlags {
    compute_type_32f: ComputeType32F::Default,
    compute_type_64f: ComputeType64F::Default,
    compute_type_16f: ComputeType16F::Default,
    fixedpoint_mantissa_bits: 0,
    matmul_drivers: MatMulDrivers::all(),
    matmul_use_syrk: MatMulUseSyrk::Scientific,
    trace_matmul: false,
});

/// Read the process-global default flags.
pub fn default_flags() -> CudaRuntimeFlags {
    *CUDA_FLAGS_DEFAULT.read().unwrap()
}

/// Set the process-global default flags. Affects devices whose flags are
/// `None` (resolved at launch time), not devices with pinned overrides.
pub fn set_default_flags(flags: CudaRuntimeFlags) {
    *CUDA_FLAGS_DEFAULT.write().unwrap() = flags;
}
