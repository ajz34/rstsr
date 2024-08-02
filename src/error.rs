use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::cuda_backend::error::CudaError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Index out of bound: index {index:}, shape {shape:}")]
    IndexOutOfBound { index: isize, shape: isize },

    #[error("Value out of range: value {value:?}, min {min:?}, max {max:?}")]
    ValueOutOfRange { value: isize, min: isize, max: isize },

    #[error("Invalid integer: value {value:?}, msg {msg:?}")]
    InvalidInteger { value: isize, msg: String },

    #[error("Value not match: got {got:?}, expect {expect:?}")]
    USizeNotMatch { got: usize, expect: usize },

    /* #region Wrapped Errors */
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),

    #[error("Error with message: {0:?}")]
    Msg(String),
    /* #endregion */
}

pub type Result<T> = std::result::Result<T, Error>;
