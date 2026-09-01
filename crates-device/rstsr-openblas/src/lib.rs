#![allow(clippy::needless_return)]
#![allow(non_camel_case_types)]
#![doc = include_str!("../readme.md")]

pub mod conversion;
pub mod device;
pub mod matmul;
pub mod matmul_impl;
pub mod prelude_dev;
pub mod rayon_auto_impl;
pub mod threading;

#[cfg(any(feature = "use_batched_gemm", feature = "use_batched_gemm_strided"))]
pub mod batched_gemm_impl;

pub mod driver_impl;
#[cfg(feature = "linalg")]
pub mod linalg_auto_impl;

#[cfg(feature = "sci")]
pub mod sci_auto_impl;

use rstsr_core::prelude_dev::DeviceCpuRayon;

#[derive(Clone, Debug)]
pub struct DeviceOpenBLAS {
    base: DeviceCpuRayon,
}

pub(crate) use rstsr_openblas_ffi as lapack_ffi;
pub(crate) use DeviceOpenBLAS as DeviceBLAS;
pub(crate) use DeviceOpenBLAS as DeviceRayonAutoImpl;
