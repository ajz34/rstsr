//! Extended (array-API promotion-compatible) linear algebra for the faer device.
//!
//! The operands may have different dtypes, each pair promoted to its common dtype
//! inside the kernel. The same-dtype entries live in the sibling `matmul` module.

pub mod matmul;
pub mod tensordot;
pub mod vecdot;
