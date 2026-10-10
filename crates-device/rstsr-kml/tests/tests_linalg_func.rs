// device alias for the shared test sources (rstsr-blas-traits/src/blas_tests_impl)
#[cfg(feature = "linalg")]
pub use rstsr_kml::DeviceKML as DeviceBLAS;
#[cfg(feature = "linalg")]
mod linalg_func;
