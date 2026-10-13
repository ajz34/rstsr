//! Promoted arithmetic Python bindings for the rstsr-faer-py array-API namespace
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod arith;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(arith::add, m)?)?;
    m.add_function(wrap_pyfunction!(arith::subtract, m)?)?;
    m.add_function(wrap_pyfunction!(arith::multiply, m)?)?;
    m.add_function(wrap_pyfunction!(arith::divide, m)?)?;
    Ok(())
}
