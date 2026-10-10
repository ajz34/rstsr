//! Extended linear algebra kernels for the CPU-rayon backend.
//!
//! Rayon twins of the serial `ext_linalg` kernels: the operands may have
//! different dtypes and each pair is promoted to its common dtype inside the
//! kernel.

pub mod matmul;
pub mod tensordot;
pub mod vecdot;
