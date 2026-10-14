//! Ordering-comparison Python bindings for the rstsr-faer-py array-API namespace
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod compare_ord;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_ord::less, m)?)?;
    m.add_function(wrap_pyfunction!(compare_ord::less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(compare_ord::greater, m)?)?;
    m.add_function(wrap_pyfunction!(compare_ord::greater_equal, m)?)?;
    Ok(())
}
