#![recursion_limit = "512"]
#![allow(refining_impl_trait)]
#![allow(clippy::needless_return)]
#![cfg_attr(not(test), no_std)]
#![doc = include_str!("docs/lib.md")]

pub mod prelude_dev;

pub mod error;
pub mod flags;

pub mod layout;
pub use layout::{DimAPI, Layout};

pub mod storage;

pub mod tensor;
pub mod tensorbase;
pub use tensorbase::{Tensor, TensorBase, TensorCow, TensorView, TensorViewMut};

pub mod format;

#[cfg(feature = "rayon")]
pub mod feature_rayon;

#[cfg(feature = "rayon")]
pub mod device_cpu_rayon;
pub mod device_cpu_serial;

mod dev_utilities;

pub type DeviceCpu = device_cpu_serial::DeviceCpuSerial;

pub mod doc_api_specification {
    #![doc = include_str!("docs/api_specification.md")]

    #[allow(unused_imports)]
    use crate::prelude_dev::*;
}
