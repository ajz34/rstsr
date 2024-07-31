#![allow(refining_impl_trait)]

pub mod error;
pub use error::{Error, Result};

pub mod layout;

pub mod data;
pub mod device;

pub mod tensor;

pub mod cpu_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;
