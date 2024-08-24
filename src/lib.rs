#![allow(refining_impl_trait)]
#![allow(clippy::needless_return)]
#![cfg_attr(not(test), no_std)]
#![doc = include_str!("readme.md")]

/* #region Configuration */

pub enum Order {
    C,
    F,
}

impl Default for Order {
    fn default() -> Self {
        if cfg!(feature = "c_prefer") {
            Order::C
        } else {
            Order::F
        }
    }
}

/* #endregion */

pub mod prelude_dev;

pub mod error;

pub mod layout;
pub use layout::{DimAPI, Layout};

pub mod storage;

pub mod tensor;
pub mod tensorbase;
pub use tensorbase::{Tensor, TensorBase};

pub mod format;

pub mod cpu_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;
