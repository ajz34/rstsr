#![doc = include_str!("doc.md")]

pub(crate) mod cublas_args;
pub(crate) mod drivers;
pub(crate) mod geom;
pub(crate) mod inner;
pub mod mode;
pub mod tensor;

mod flags;

pub use drivers::GemmDriver;
pub use flags::{MatMulDrivers, MatMulUseSyrk};
pub use mode::{ComputeType16F, ComputeType32F, ComputeType64F, CudaMatMulMode};
pub use tensor::{matmul_with_mode_f, TensorCudaMatMulAPI, WithMode};
