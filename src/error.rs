#[cfg(feature = "cuda")]
use crate::cuda_backend::error::CudaError;

#[derive(thiserror::Error, Debug)]

pub enum Error {
    /* #region Wrapped Errors */
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),
    /* #endregion */
}

pub type Result<T> = std::result::Result<T, Error>;
