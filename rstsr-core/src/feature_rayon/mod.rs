//! This module contains some features for rayon (parallel).
//!
//! - Layout parallel iterator
//! - Tensor parallel iterator

pub mod assignment;
pub mod device;
pub mod layout_par_iter;
pub mod op_with_func;

pub use assignment::*;
pub use device::*;
pub use layout_par_iter::*;
pub use op_with_func::*;
