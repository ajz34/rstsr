//! Alias of [`rstsr_cuda::prelude`] — the CUDA device backend (`rstsr-cuda`),
//! wired by the cargo feature `cuda`.
//!
//! Glob [`prelude`] for the device types and traits; the free functions live in
//! [`prelude::rstsr_funcs`], also reachable as `rt::cuda::*` through the facade
//! prelude's `rt` module.
//!
//! The cuTENSOR wrapper is reachable as [`cutensor_wrap`] under the `cutensor`
//! feature (needs a CUDA >= 12.0 toolchain).

pub use rstsr_cuda::prelude;

#[cfg(feature = "cutensor")]
pub use rstsr_cuda::cutensor_wrap;
