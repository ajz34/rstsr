//! Alias of [`rstsr_cuda::prelude`] — the CUDA device backend (`rstsr-cuda`),
//! wired by the cargo feature `cuda`.
//!
//! Glob [`prelude`] for the device types and traits; the free functions live in
//! [`prelude::rstsr_funcs`], also reachable as `rt::cuda::*` through the facade
//! prelude's `rt` module.

pub use rstsr_cuda::prelude;
