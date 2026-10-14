//! Bitwise, shift and logical Python bindings for the rstsr-faer-py array-API namespace
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod bitwise;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bitwise::remainder, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::bitwise_left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::bitwise_right_shift, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::pow, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise::logical_xor, m)?)?;
    Ok(())
}
