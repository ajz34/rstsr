//! User-facing prelude of rstsr-cuda.
//!
//! Glob the whole prelude for the device types and traits:
//!
//! ```
//! use rstsr_cuda::prelude::*;
//! ```
//!
//! - [`rstsr_traits`]: the API traits ([`CudaSyncAPI`], [`TensorCudaMatMulAPI`],
//!   [`TensorPinnedHostViewAPI`], ...).
//! - [`rstsr_structs`]: the device and buffer types ([`DeviceCuda`], [`CudaStreamObj`],
//!   [`CudaEvent`], [`PinnedBuffer`], ...).
//! - [`rstsr_funcs`]: free functions (`device`, `device_count`, `is_available`, `synchronize`,
//!   ...), not globbed into the root (the `rstsr-core`/`rstsr` convention); through the `rstsr`
//!   facade they are reachable as `rt::cuda::*`.
//!
//! The `rstsr` facade crate forwards these namespaces under its `cuda` feature.

pub mod rstsr_traits {
    pub use crate::matmul::TensorCudaMatMulAPI;
    pub use crate::pinned::{TensorPinnedHostViewAPI, TensorPinnedHostViewMutAPI};
    pub use crate::sync::CudaSyncAPI;
}

pub mod rstsr_structs {
    pub use crate::buffer::CudaBuffer;
    pub use crate::device::DeviceCuda;
    pub use crate::event::CudaEvent;
    pub use crate::flags::CudaRuntimeFlags;
    pub use crate::matmul::{
        ComputeType16F, ComputeType32F, ComputeType64F, CudaMatMulMode, GemmDriver, MatMulDrivers, MatMulUseSyrk,
        WithMode,
    };
    pub use crate::pinned::{pinned_flags, DeviceCudaPinned, PinnedBuffer};
    pub use crate::registry::CudaOrdinalObj;
    pub use crate::stream::CudaStreamObj;
}

pub mod rstsr_funcs {
    pub use crate::device::{device, device_f, memory_info, memory_info_f};
    pub use crate::flags::{default_flags, set_default_flags};
    pub use crate::registry::{device_count, is_available};
    pub use crate::sync::{synchronize, synchronize_f, synchronize_tensor, synchronize_tensor_f};
}

// final re-exports

pub use rstsr_structs::*;
pub use rstsr_traits::*;
