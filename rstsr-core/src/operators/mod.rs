//! Operations on tensors.

pub mod adv_indexing;
pub mod assignment;
pub mod combined_trait;
pub mod ext_linalg;
pub mod linalg;
pub mod matmul;
pub mod ops;
pub mod reduction;
pub mod searching;
pub mod set;
pub mod sorting;

pub mod exports {
    use super::*;

    pub use adv_indexing::*;
    pub use assignment::*;
    pub use combined_trait::*;
    pub use ext_linalg::*;
    pub use linalg::*;
    pub use matmul::*;
    pub use ops::*;
    pub use reduction::*;
    pub use searching::*;
    pub use set::*;
    pub use sorting::*;
}
