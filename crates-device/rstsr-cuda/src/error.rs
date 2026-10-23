//! Error mapping from cudarc driver errors to RSTSR errors.

use cudarc::driver::{sys, DriverError};
use rstsr_core::prelude_dev::*;

pub(crate) fn map_driver_error(err: DriverError) -> Error {
    rstsr_error!(DeviceError, "CUDA driver error: {err}")
}

pub(crate) fn map_driver_result<T>(res: core::result::Result<T, DriverError>) -> Result<T> {
    res.map_err(map_driver_error)
}

/// Map a driver error on an allocation of `requested` bytes (OOM-aware).
pub(crate) fn map_driver_result_alloc<T>(res: core::result::Result<T, DriverError>, requested: usize) -> Result<T> {
    res.map_err(|e| {
        if e.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            rstsr_error!(MemoryError, "CUDA allocation failed (requested {requested} bytes): {e}")
        } else {
            map_driver_error(e)
        }
    })
}
