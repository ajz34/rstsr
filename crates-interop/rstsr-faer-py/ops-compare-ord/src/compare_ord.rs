//! Ordering comparisons over the promoted dtype matrix.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr_core::tensor::operators::exports::{
    TensorGreaterAPI, TensorGreaterEqualAPI, TensorLessAPI, TensorLessEqualAPI,
};

use rstsr_faer_py_core::{dispatch_bin_promote, lift, type_err, AnyTensor, FTensor, NativeArray};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

fn op_less<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorLessAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::less_f(a, b)
}

fn op_less_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorLessEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::less_equal_f(a, b)
}

fn op_greater<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorGreaterAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::greater_f(a, b)
}

fn op_greater_equal<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorGreaterEqualAPI<&'x FTensor<U>, Output = FTensor<bool>>,
{
    rt::greater_equal_f(a, b)
}

#[pyfunction]
pub fn less(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_promote!(x1.t, x2.t, "less", op_less)? })
}

#[pyfunction]
pub fn less_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_promote!(x1.t, x2.t, "less_equal", op_less_equal)? })
}

#[pyfunction]
pub fn greater(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_promote!(x1.t, x2.t, "greater", op_greater)? })
}

#[pyfunction]
pub fn greater_equal(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_promote!(x1.t, x2.t, "greater_equal", op_greater_equal)? })
}
