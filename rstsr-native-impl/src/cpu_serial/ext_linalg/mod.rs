//! Extended linear algebra kernels for the CPU-serial backend.
//!
//! These are the array-API promotion-compatible forms of the linalg operations:
//! the operands may have different dtypes and each pair is promoted to its
//! common dtype inside the kernel. The same-dtype kernels live in the
//! corresponding `linalg`/`matmul_naive` modules.

pub mod matmul;
