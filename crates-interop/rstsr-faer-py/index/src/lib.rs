//! Indexing, set, sorting and searching Python bindings for the rstsr-faer-py array-API namespace.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod indexing;
mod searching;
mod set;
mod sorting;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(indexing::take, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::getitem_basic, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_basic, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::getitem_mask, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_mask, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_mask_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_array, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::setitem_array_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::take_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(set::unique_values, m)?)?;
    m.add_function(wrap_pyfunction!(set::unique_counts, m)?)?;
    m.add_function(wrap_pyfunction!(set::unique_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(set::unique_all, m)?)?;
    m.add_function(wrap_pyfunction!(set::isin, m)?)?;
    m.add_function(wrap_pyfunction!(sorting::sort, m)?)?;
    m.add_function(wrap_pyfunction!(sorting::argsort, m)?)?;
    m.add_function(wrap_pyfunction!(searching::searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(searching::nonzero, m)?)?;
    Ok(())
}
