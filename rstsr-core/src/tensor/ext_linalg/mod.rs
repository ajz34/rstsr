//! Extended (array-API promotion-compatible) linear algebra.
//!
//! These are the promotion-compatible forms of the linalg operations: the
//! operands may have different dtypes, each pair promoted to their common dtype
//! before the kernel. The same-dtype entries live in [`crate::tensor::linalg`].

pub mod matmul;
pub mod outer;
pub mod tensordot;
pub mod vecdot;

pub mod exports {
    use super::*;

    pub use matmul::*;
    pub use outer::*;
    pub use tensordot::*;
    pub use vecdot::*;
}
