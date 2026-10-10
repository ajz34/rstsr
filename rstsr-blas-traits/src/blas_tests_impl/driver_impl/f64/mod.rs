// per-device alias defined by the test entry
pub use crate::DeviceBLAS;
pub use rstsr::prelude::*;
pub use rstsr_core::prelude_dev::fingerprint;
pub use rstsr_test_manifest::get_vec;

mod gesdd;
mod gesv;
mod gesvd;
mod getrf_getri;
mod potrf;
mod syev;
mod syevd;
mod sygv;
mod sygvd;
mod sysv;
