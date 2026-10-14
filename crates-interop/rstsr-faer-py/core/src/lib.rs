//! Shared core of the `rstsr-faer-py` binding: the dtype-erased tensor handle
//! (`AnyTensor`/`NativeArray`), the dtype singletons, the `dispatch_*` macros
//! and the module-registration entry point.
//!
//! This crate exists so that no single `rustc` invocation has to monomorphize
//! the binding's whole dtype-dispatch surface: the domain sub-crates each
//! expand a slice of the macros, and this crate holds only the shared items.

pub mod any_tensor;
pub mod device;
pub mod dlpack;
pub mod dtype;
pub mod info;
pub mod promotion;

pub use any_tensor::{
    any_of, device_faer, dim_from, err_py, idx_lift, idx_lift_tensor, item_t, lift, lift_vec, liftp, parse_leaf,
    proj_name, proj_ndim, proj_shape, proj_size, tolist_t, type_err, AnyTensor, AnyTensorRef, DtypeName, FTensor,
    IntoAnyTensor, NativeArray, PyScalar,
};
pub use dtype::Dtype;

use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyComplex};

/// Name -> dtype singleton registry, filled once at module init.
static DTYPES: OnceLock<HashMap<&'static str, Py<Dtype>>> = OnceLock::new();

pub fn dtype_by_name(py: Python<'_>, name: &str) -> PyResult<Py<Dtype>> {
    DTYPES
        .get()
        .and_then(|m| m.get(name))
        .map(|d| d.clone_ref(py))
        .ok_or_else(|| PyValueError::new_err(format!("unknown dtype {name:?}")))
}

#[pyfunction]
fn finfo<'py>(py: Python<'py>, dtype: &Bound<'py, Dtype>) -> PyResult<info::Finfo> {
    let d = dtype_by_name(py, dtype.borrow().name)?;
    match dtype.borrow().name {
        "float32" => Ok(info::finfo32(d)),
        "float64" => Ok(info::finfo64(d)),
        _ => Err(PyValueError::new_err("finfo: only real floating-point dtypes are allowed")),
    }
}

#[pyfunction]
fn iinfo<'py>(py: Python<'py>, dtype: &Bound<'py, Dtype>) -> PyResult<info::Iinfo> {
    let d = dtype_by_name(py, dtype.borrow().name)?;
    Ok(match dtype.borrow().name {
        "int8" => info::iinfo_i8(d),
        "int16" => info::iinfo_i16(d),
        "int32" => info::iinfo_i32(d),
        "int64" => info::iinfo_i64(d),
        "uint8" => info::iinfo_u8(d),
        "uint16" => info::iinfo_u16(d),
        "uint32" => info::iinfo_u32(d),
        "uint64" => info::iinfo_u64(d),
        _ => return Err(PyValueError::new_err("iinfo: only integral dtypes are allowed")),
    })
}

/// Kind string used by the Python layer for weak-scalar ordering.
#[pyfunction]
fn dtype_kind(dtype: &Bound<'_, Dtype>) -> &'static str {
    match dtype.borrow().name {
        "bool" => "bool",
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64" => "integral",
        "float32" | "float64" => "real floating",
        _ => "complex floating",
    }
}

/// Default dtype for a Python scalar (spec: bool/int -> int64, float ->
/// float64, complex -> complex128).
#[pyfunction]
fn default_dtype_for<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Py<Dtype>> {
    let name = if value.is_instance_of::<PyBool>() {
        "bool"
    } else if value.cast::<PyComplex>().is_ok() {
        "complex128"
    } else if value.extract::<i64>().is_ok() {
        "int64"
    } else if value.extract::<f64>().is_ok() {
        "float64"
    } else {
        return Err(PyValueError::new_err(format!("not a supported scalar type: {}", value.get_type().name()?)));
    };
    dtype_by_name(py, name)
}

/// Register the core surface (dtype singletons, classes, dtype helpers and the
/// DLPack bridge) on the module. Domain sub-crates chain their own
/// `add_module` around this one.
pub fn add_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dtype::add_dtype_objects(m)?;
    device::add_device_object(m)?;

    // Collect the singletons just registered into the name-keyed registry.
    let mut map: HashMap<&'static str, Py<Dtype>> = HashMap::new();
    macro_rules! reg {
        ($attr:ident) => {{
            let name: &'static str = stringify!($attr);
            let obj: Py<Dtype> = m.getattr(name)?.extract()?;
            map.insert(name, obj);
        }};
    }
    reg!(bool);
    reg!(int8);
    reg!(int16);
    reg!(int32);
    reg!(int64);
    reg!(uint8);
    reg!(uint16);
    reg!(uint32);
    reg!(uint64);
    reg!(float32);
    reg!(float64);
    reg!(complex64);
    reg!(complex128);
    let _ = DTYPES.set(map);

    m.add("e", std::f64::consts::E)?;
    m.add("inf", f64::INFINITY)?;
    m.add("nan", f64::NAN)?;
    m.add("pi", std::f64::consts::PI)?;

    m.add_class::<NativeArray>()?;
    m.add_class::<Dtype>()?;
    m.add_class::<device::Device>()?;
    m.add_class::<info::Finfo>()?;
    m.add_class::<info::Iinfo>()?;

    m.add_function(wrap_pyfunction!(finfo, m)?)?;
    m.add_function(wrap_pyfunction!(iinfo, m)?)?;
    m.add_function(wrap_pyfunction!(dtype_kind, m)?)?;
    m.add_function(wrap_pyfunction!(promotion::dtype_promote, m)?)?;
    m.add_function(wrap_pyfunction!(default_dtype_for, m)?)?;

    m.add_function(wrap_pyfunction!(dlpack::dlpack_export, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack::dlpack_import, m)?)?;

    Ok(())
}
