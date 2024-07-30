use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::cuda_backend::error::CudaError;
use crate::layout::Shape;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /* #region Layout Errors */
    #[error("Index out of bound: index {index:?}, shape {shape:?}")]
    IndexOutOfBound { index: Vec<usize>, shape: Vec<usize> },

    /* #region Wrapped Errors */
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),
    /* #endregion */
}

pub type Result<T> = std::result::Result<T, Error>;
