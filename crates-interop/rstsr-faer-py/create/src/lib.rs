//! Creation and manipulation Python bindings for the rstsr-faer-py array-API namespace.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod creation;
mod manipulation;
mod shape_ops;

pub use creation::ScalarCastTarget;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(creation::asarray_from_flat, m)?)?;
    m.add_function(wrap_pyfunction!(creation::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ones, m)?)?;
    m.add_function(wrap_pyfunction!(creation::empty, m)?)?;
    m.add_function(wrap_pyfunction!(creation::full, m)?)?;
    m.add_function(wrap_pyfunction!(creation::arange, m)?)?;
    m.add_function(wrap_pyfunction!(creation::astype, m)?)?;
    m.add_function(wrap_pyfunction!(creation::eye, m)?)?;
    m.add_function(wrap_pyfunction!(creation::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::tril, m)?)?;
    m.add_function(wrap_pyfunction!(creation::triu, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::broadcast_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::concat, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::stack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::unstack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::flip, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::moveaxis, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::repeat, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::roll, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::tile, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::diff, m)?)?;
    m.add_function(wrap_pyfunction!(shape_ops::reshape, m)?)?;
    m.add_function(wrap_pyfunction!(shape_ops::transpose, m)?)?;
    m.add_function(wrap_pyfunction!(shape_ops::getitem_int, m)?)?;
    m.add_function(wrap_pyfunction!(shape_ops::broadcast_to, m)?)?;
    Ok(())
}
