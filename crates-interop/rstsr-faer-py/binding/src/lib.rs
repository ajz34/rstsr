//! rstsr_faer — pyo3 binding of rstsr (faer device) for array-API validation.
//!
//! Layer split: the crates under `crates-interop/rstsr-faer-py/` own tensor
//! construction, dtype dispatch and the rstsr calls; the Python package
//! `rstsr_faer.api` owns only signatures, protocol objects and marshalling. No
//! numeric algorithms beyond what rstsr itself provides.
//!
//! The surface is spread across `rstsr-faer-py-core` (handle, dtypes, dispatch
//! macros) and the domain sub-crates (`-create`, `-index`, `-linalg` and the
//! `-ops-*` family); this crate is just the `#[pymodule]` that chains their
//! registration. The split is a build-memory fix: one rustc used to
//! monomorphize the whole dispatch surface and needed ~45 GB.

use pyo3::prelude::*;

#[pymodule]
fn rstsr_faer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rstsr_faer_py_core::add_module(m)?;
    rstsr_faer_py_create::add_module(m)?;
    rstsr_faer_py_index::add_module(m)?;
    rstsr_faer_py_linalg::add_module(m)?;
    rstsr_faer_py_ops_unary::add_module(m)?;
    rstsr_faer_py_ops_binary::add_module(m)?;
    rstsr_faer_py_ops_binary_float::add_module(m)?;
    rstsr_faer_py_ops_bitwise::add_module(m)?;
    rstsr_faer_py_ops_arith::add_module(m)?;
    rstsr_faer_py_ops_compare_eq::add_module(m)?;
    rstsr_faer_py_ops_compare_ord::add_module(m)?;
    rstsr_faer_py_ops_reduce::add_module(m)?;
    Ok(())
}
