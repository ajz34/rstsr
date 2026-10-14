//! Reduction and predicate Python bindings for the rstsr-faer-py array-API namespace
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod reduce;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reduce::all, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::any, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::isnan, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::isinf, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::sum, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::prod, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::max, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::min, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::mean, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::var, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::std, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::cumulative_sum, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::cumulative_prod, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::argmax, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::argmin, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::sum_bool, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::where_, m)?)?;
    m.add_function(wrap_pyfunction!(reduce::clip, m)?)?;
    Ok(())
}
