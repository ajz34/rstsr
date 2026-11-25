//! Matrix multiplication on `DeviceCuda`.
//!
//! `c = alpha * (a @ b) + beta * c` is served by a chain of drivers, tried in this order until one
//! takes the problem: cuBLASLt, `cublasGemmEx`, the classic cuBLAS entries, the tiled SIMT kernel,
//! and the naive kernel. Which drivers are candidates is a pure table of the type triple and the
//! compute mode (the candidate table in `drivers`); which of them can *express the layout* is answered at
//! run time by each driver (a stepped output, a reversed operand, a non-affine batch are handed
//! on). The naive driver accepts everything, so the chain is total. Drivers enabled for a device:
//! [`MatMulDrivers`] in [`CudaRuntimeFlags::matmul_drivers`][crate::flags::CudaRuntimeFlags].
//!
//! # Compute modes
//!
//! The mode is keyed by the **output** type: [`CudaMatMulMode`] names the enum of its family
//! ([`ComputeType32F`], [`ComputeType64F`], [`ComputeType16F`]), and the flags carry one field per
//! family. `DeviceMatMulAPI` reads the mode of the output type from the device flags; the
//! tensor-level `*_mode` entries (passing it per call) are not built yet.
//!
//! The cuBLAS drivers take the mode as given; the tiled and naive kernels compute in the output
//! precision (or better) whatever the mode. A driver that cannot express a mode declines and the
//! next candidate runs: `Fast16BF` on the classic entries, and `EmulatedFixedPoint` without
//! `cudarc`'s CUDA 13.0 bindings.
//!
//! # Type triples
//!
//! | `(a, b) -> c` | accelerated drivers | always |
//! |---|---|---|
//! | `f32 -> f32`, `f64 -> f64` | `cublasGemmEx`, classic cuBLAS, tiled | naive |
//! | everything else (integers, complex, half, mixed operands) | — (later batches) | naive |
//!
//! # Layouts
//!
//! Every layout the rule table admits is served: strided, negative and zero strides, broadcast
//! and non-affine batches, either default order. The inner dot (`[k] @ [k]`) has no GEMM geometry
//! and is refused by name.

pub(crate) mod cublas_args;
pub(crate) mod drivers;
pub(crate) mod geom;
pub(crate) mod inner;
pub mod mode;

mod flags;

pub use drivers::GemmDriver;
pub use flags::MatMulDrivers;
pub use mode::{ComputeType16F, ComputeType32F, ComputeType64F, CudaMatMulMode};
