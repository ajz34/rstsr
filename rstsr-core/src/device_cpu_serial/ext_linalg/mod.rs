//! Extended (array-API promotion-compatible) linear algebra for the serial CPU device.
//!
//! The operands may have different dtypes, each pair promoted to its common dtype
//! inside the kernel. The same-dtype entries live in the sibling `linalg` module.

pub mod matmul;
