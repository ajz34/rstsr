//! Runtime flags for the CUDA device.

use rstsr_core::prelude_dev::*;
use std::sync::RwLock;

/// 32-bit-float compute-type family (`cublasComputeType_t` names).
///
/// The process-global default can be seeded from the environment variable
/// `RSTSR_CUDA_COMPUTE_TYPE_32F` = `default` | `pedantic` | `fast16f` |
/// `fast16bf` | `fasttf32` (unknown values keep the default).
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

/// Runtime flags of a CUDA device (additive; keep small and `Copy`).
///
/// Two layers:
/// - a process-global default, read by [`default_flags`] and writable via [`set_default_flags`]
///   (env-seeded on first use);
/// - a per-device `Option<CudaRuntimeFlags>` override: `None` tracks the global, `Some(x)` pins.
///
/// Flags are resolved at launch time by flag-consuming ops, but compared as
/// stored (see `DeviceCuda::same_device`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct CudaRuntimeFlags {
    /// Compute type for 32-bit-float GEMMs.
    pub compute_type_32f: ComputeType32F,
}

impl CudaRuntimeFlags {
    /// Seed from process environment (called once at first device creation).
    fn from_env() -> Self {
        let mut flags = Self::default();
        if let Ok(v) = std::env::var("RSTSR_CUDA_COMPUTE_TYPE_32F") {
            flags.compute_type_32f = match v.to_ascii_lowercase().as_str() {
                "default" => ComputeType32F::Default,
                "pedantic" => ComputeType32F::Pedantic,
                "fast16f" => ComputeType32F::Fast16F,
                "fast16bf" => ComputeType32F::Fast16BF,
                "fasttf32" => ComputeType32F::FastTF32,
                _ => flags.compute_type_32f, // unknown values keep the default
            };
        }
        flags
    }
}

static CUDA_FLAGS_DEFAULT: RwLock<CudaRuntimeFlags> =
    RwLock::new(CudaRuntimeFlags { compute_type_32f: ComputeType32F::Default });

static CUDA_FLAGS_SEEDED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Read the process-global default flags (seeds from env on first call).
pub fn default_flags() -> CudaRuntimeFlags {
    CUDA_FLAGS_SEEDED.get_or_init(|| {
        let seeded = CudaRuntimeFlags::from_env();
        *CUDA_FLAGS_DEFAULT.write().unwrap() = seeded;
    });
    *CUDA_FLAGS_DEFAULT.read().unwrap()
}

/// Set the process-global default flags. Affects devices whose flags are
/// `None` (resolved at launch time), not devices with pinned overrides.
pub fn set_default_flags(flags: CudaRuntimeFlags) {
    default_flags(); // ensure env seeding happens before the first override
    *CUDA_FLAGS_DEFAULT.write().unwrap() = flags;
}
