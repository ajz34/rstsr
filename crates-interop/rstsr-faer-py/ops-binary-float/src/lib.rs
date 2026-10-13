//! Floating-point promoted binary Python bindings for the rstsr-faer-py array-API namespace
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod binary_float;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_float::copysign, m)?)?;
    m.add_function(wrap_pyfunction!(binary_float::hypot, m)?)?;
    m.add_function(wrap_pyfunction!(binary_float::nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(binary_float::logaddexp, m)?)?;
    Ok(())
}
