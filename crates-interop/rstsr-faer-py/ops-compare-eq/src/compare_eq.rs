//! Equality comparisons over the full dtype-pair promotion matrix.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr_core::tensor::operators::exports::{TensorEqualAPI, TensorNotEqualAPI};

use rstsr_faer_py_core::{dispatch_bin_promote_eq, lift, AnyTensor, FTensor, NativeArray};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

fn op_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::equal_f(a, b)
}

fn op_not_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::not_equal_f(a, b)
}

#[pyfunction]
pub fn equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_promote_eq!(x1.t, x2.t, "equal", op_equal)? })
}

#[pyfunction]
pub fn not_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_promote_eq!(x1.t, x2.t, "not_equal", op_not_equal)? })
}
