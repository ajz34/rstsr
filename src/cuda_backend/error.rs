/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    CudaDriver(#[from] cudarc::driver::result::DriverError),
}

pub use CudaError::CudaDriver as CudaDriverError;
