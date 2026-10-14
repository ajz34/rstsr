//! Linear-algebra Python bindings for the rstsr-faer-py array-API namespace.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod linalg;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linalg::linalg_cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_det, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_trace, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_eig, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_eigh, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_eigvals, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_eigvalsh, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_inv, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_matrix_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_matrix_norm, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_matrix_power, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_matrix_rank, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_outer, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_cross, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_pinv, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_qr, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_solve, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_svd, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_svdvals, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_slogdet, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_vecdot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_vector_norm, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linalg_tensordot, m)?)?;
    Ok(())
}
