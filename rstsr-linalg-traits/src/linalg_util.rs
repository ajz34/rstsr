//! Shared helpers for batched linalg entry points.

use rstsr_core::prelude_dev::*;

/// Split the shape of a stack of matrices into the batch shape and the 2-D
/// matrix shape.
///
/// The two matrix axes are the last two under `RowMajor` and the first two
/// under `ColMajor` (the device order rule); all remaining dims are the batch.
/// Errors when `ndim < 2`.
pub(crate) fn batch_and_matrix_shape(shape: &[usize], order: FlagOrder) -> Result<(Vec<usize>, [usize; 2])> {
    let ndim = shape.len();
    rstsr_assert!(ndim >= 2, InvalidLayout, "linalg: expected at least 2 dimensions, got {ndim}")?;
    let (batch_shape, mat) = match order {
        RowMajor => (shape[..ndim - 2].to_vec(), [shape[ndim - 2], shape[ndim - 1]]),
        ColMajor => (shape[2..].to_vec(), [shape[0], shape[1]]),
    };
    Ok((batch_shape, mat))
}

/// As [`batch_and_matrix_shape`], additionally requiring the matrix to be square;
/// returns the batch shape.
pub(crate) fn batch_and_square_shape(shape: &[usize], order: FlagOrder) -> Result<Vec<usize>> {
    let (batch_shape, [m, n]) = batch_and_matrix_shape(shape, order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "linalg: expected square matrices, got {m}x{n}")?;
    Ok(batch_shape)
}
