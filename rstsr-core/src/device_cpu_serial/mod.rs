//! Backend for CPU, serial only.

pub mod adv_indexing;
pub mod array_indexing;
pub mod assignment;
pub mod conversion;
pub mod creation;
pub mod device;
pub mod ext_linalg;
pub mod linalg;
pub mod nonzero;
pub mod operators;
pub mod reduction;
pub mod searching;
pub mod set;
pub mod sorting;
pub mod take_along_axis;

pub use device::*;
pub use operators::*;
