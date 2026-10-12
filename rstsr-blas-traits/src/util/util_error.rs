use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// Map a raw LAPACK `info` return to a rstsr error.
///
/// `info < 0`: illegal argument — an internal bug (builders validate all
/// arguments before the FFI call), reported with the Fortran argument index.
/// `info > 0`: algorithmic failure, semantics defined by the routine itself.
/// `info == 0` returns `Ok`.
pub fn map_lapack_info(info: blas_int, routine: &str) -> Result<()> {
    if info == 0 {
        return Ok(());
    }
    // tier-internal allocation sentinels (-1010/-1011)
    if info == -1010 || info == -1011 {
        return Err(rstsr_error!(MemoryError, "Lapack {routine}: workspace allocation failed"));
    }
    let info = blas_int_as_i32(info);
    let neg_info = -info;
    if info < 0 {
        Err(rstsr_error!(
            InvalidValue,
            "Lapack {routine}: illegal argument in argument {neg_info} (internal bug, please report)"
        ))
    } else {
        Err(rstsr_error!(InvalidValue, "Lapack {routine} (info = {info})"))
    }
}

/// Checked conversion of `blas_int` to `i32` (the info channel of
/// `RSTSRError::ErrorCode`); saturates instead of truncating under ilp64.
pub fn blas_int_as_i32(x: blas_int) -> i32 {
    #[cfg(feature = "ilp64")]
    {
        i32::try_from(x).unwrap_or(if x > 0 { i32::MAX } else { i32::MIN })
    }
    #[cfg(not(feature = "ilp64"))]
    {
        x
    }
}
