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
//! tensor-level entries of [`TensorCudaMatMulAPI`] (`*_mode`; `a.matmul_with_mode(&b, mode).out::<TC>()`, and the fallible
//! [`matmul_with_mode_f`]) pass it per call instead.
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
//! | `f32 -> f32`, `f64 -> f64` | cuBLASLt, `cublasGemmEx`, classic cuBLAS, tiled | naive |
//! | `c32`, `c64` | cuBLASLt, `cublasGemmEx`, classic cuBLAS | naive |
//! | `f16 -> f16`, `f16 -> f32`, `bf16 -> bf16`, `bf16 -> f32` | cuBLASLt, `cublasGemmEx` | naive |
//! | everything else (integers, operands of different dtypes) | — | naive |
//!
//! A mode the triple cannot serve is the family default by the time a driver sees it: the `Fast*`
//! modes convert `f32` inputs, so they do not apply to complex inputs nor to half inputs with an
//! `f32` output; `Acc16F` (which scales in `f16`) is `f16` only; the fixed-point emulation is
//! real `f64` only. `Pedantic` is served by every accelerated triple.
//!
//! Operands of different dtypes need no entry of their own: `matmul_from` / `matmul_with_output`
//! carry no `Mul` bound, and the naive kernel casts each operand to the output type.
//!
//! # Layouts
//!
//! Every layout the rule table admits is served: strided, negative and zero strides, broadcast
//! and non-affine batches, either default order. The inner dot (`[k] @ [k]`) is served as the
//! `1 x k @ k x 1` GEMM it is, which cuBLAS runs at memory bandwidth.
//!
//! # Resources
//!
//! The cuBLAS drivers share one handle per stream, kept behind a lock; cuBLASLt keeps one handle
//! and one heuristic cache per ordinal (a problem is described to it once, then reused), and one
//! 32 MiB workspace per stream, allocated on the first matmul that stream runs through it.

pub(crate) mod cublas_args;
pub(crate) mod drivers;
pub(crate) mod geom;
pub(crate) mod inner;
pub mod mode;
pub mod tensor;

mod flags;

pub use drivers::GemmDriver;
pub use flags::MatMulDrivers;
pub use mode::{ComputeType16F, ComputeType32F, ComputeType64F, CudaMatMulMode};
pub use tensor::{matmul_with_mode_f, TensorCudaMatMulAPI, WithMode};
