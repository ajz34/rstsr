//! Unary elementwise Python bindings for the rstsr-faer-py array-API namespace
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

mod unary;

use pyo3::prelude::*;

/// Register this domain's functions on the module.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unary::acos, m)?)?;
    m.add_function(wrap_pyfunction!(unary::acosh, m)?)?;
    m.add_function(wrap_pyfunction!(unary::asin, m)?)?;
    m.add_function(wrap_pyfunction!(unary::asinh, m)?)?;
    m.add_function(wrap_pyfunction!(unary::atan, m)?)?;
    m.add_function(wrap_pyfunction!(unary::atanh, m)?)?;
    m.add_function(wrap_pyfunction!(unary::cos, m)?)?;
    m.add_function(wrap_pyfunction!(unary::cosh, m)?)?;
    m.add_function(wrap_pyfunction!(unary::exp, m)?)?;
    m.add_function(wrap_pyfunction!(unary::expm1, m)?)?;
    m.add_function(wrap_pyfunction!(unary::log, m)?)?;
    m.add_function(wrap_pyfunction!(unary::log1p, m)?)?;
    m.add_function(wrap_pyfunction!(unary::log2, m)?)?;
    m.add_function(wrap_pyfunction!(unary::log10, m)?)?;
    m.add_function(wrap_pyfunction!(unary::reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(unary::sin, m)?)?;
    m.add_function(wrap_pyfunction!(unary::sinh, m)?)?;
    m.add_function(wrap_pyfunction!(unary::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(unary::tan, m)?)?;
    m.add_function(wrap_pyfunction!(unary::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(unary::ceil, m)?)?;
    m.add_function(wrap_pyfunction!(unary::floor, m)?)?;
    m.add_function(wrap_pyfunction!(unary::trunc, m)?)?;
    m.add_function(wrap_pyfunction!(unary::round, m)?)?;
    m.add_function(wrap_pyfunction!(unary::positive, m)?)?;
    m.add_function(wrap_pyfunction!(unary::square, m)?)?;
    m.add_function(wrap_pyfunction!(unary::sign, m)?)?;
    m.add_function(wrap_pyfunction!(unary::conj, m)?)?;
    m.add_function(wrap_pyfunction!(unary::signbit, m)?)?;
    m.add_function(wrap_pyfunction!(unary::real, m)?)?;
    m.add_function(wrap_pyfunction!(unary::imag, m)?)?;
    m.add_function(wrap_pyfunction!(unary::invert, m)?)?;
    m.add_function(wrap_pyfunction!(unary::negative, m)?)?;
    m.add_function(wrap_pyfunction!(unary::abs, m)?)?;
    Ok(())
}
