//! Set surface: `unique_values`/`unique_counts`/`unique_inverse`/`unique_all`
//! and `isin` over the rstsr set kernels.
//!
//! All 13 dtypes are admitted (kernel bounds are `Clone + PartialEq +
//! ExtSortCmp`). The `usize` index outputs (`indices`/`inverse_indices`/
//! `counts`) are lifted to int64; multi-output results cross to Python as
//! `Vec<NativeArray>`-style tuples and the Python layer builds the spec
//! namedtuples.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::operators::set::{OpIsinAPI, OpUniqueAPI};
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};
use rstsr_dtype_traits::ExtSortCmp;

use crate::any_tensor::{err_py, lift, type_err, AnyTensor, FTensor, NativeArray};
use crate::ops::idx_lift_tensor;

fn op_unique_values<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    DeviceFaer:
        DeviceRawAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + DeviceCreationAnyAPI<usize> + OpUniqueAPI<T, IxD>,
{
    rt::unique_values_f(t)
}

/// (values, counts) — counts lifted to int64 by the caller.
fn op_unique_counts<T>(t: &FTensor<T>) -> rt::Result<(FTensor<T>, FTensor<usize>)>
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    DeviceFaer:
        DeviceRawAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + DeviceCreationAnyAPI<usize> + OpUniqueAPI<T, IxD>,
{
    Ok(rt::unique_counts_f(t)?.into())
}

/// (values, inverse_indices) — inverse lifted to int64 by the caller.
fn op_unique_inverse<T>(t: &FTensor<T>) -> rt::Result<(FTensor<T>, FTensor<usize>)>
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    DeviceFaer:
        DeviceRawAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + DeviceCreationAnyAPI<usize> + OpUniqueAPI<T, IxD>,
{
    Ok(rt::unique_inverse_f(t)?.into())
}

/// (values, indices, inverse_indices, counts) — index outputs lifted to
/// int64 by the caller.
fn op_unique_all<T>(t: &FTensor<T>) -> rt::Result<(FTensor<T>, FTensor<usize>, FTensor<usize>, FTensor<usize>)>
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    DeviceFaer:
        DeviceRawAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + DeviceCreationAnyAPI<usize> + OpUniqueAPI<T, IxD>,
{
    let u = rt::unique_all_f(t)?;
    Ok((u.values, u.indices, u.inverse_indices, u.counts))
}

fn op_isin<T>(x1: &FTensor<T>, x2: &FTensor<T>, invert: bool) -> rt::Result<FTensor<bool>>
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    DeviceFaer: DeviceRawAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<bool> + OpIsinAPI<T, IxD>,
{
    rt::isin_f(x1, x2, invert)
}

/// Unary dispatch whose output keeps the input dtype (unique values).
macro_rules! dispatch_unique_values {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(t) => lift(($f::<bool>)(&t, $($arg),*), AnyTensor::Bool),
            AnyTensor::I8(t) => lift(($f::<i8>)(&t, $($arg),*), AnyTensor::I8),
            AnyTensor::I16(t) => lift(($f::<i16>)(&t, $($arg),*), AnyTensor::I16),
            AnyTensor::I32(t) => lift(($f::<i32>)(&t, $($arg),*), AnyTensor::I32),
            AnyTensor::I64(t) => lift(($f::<i64>)(&t, $($arg),*), AnyTensor::I64),
            AnyTensor::U8(t) => lift(($f::<u8>)(&t, $($arg),*), AnyTensor::U8),
            AnyTensor::U16(t) => lift(($f::<u16>)(&t, $($arg),*), AnyTensor::U16),
            AnyTensor::U32(t) => lift(($f::<u32>)(&t, $($arg),*), AnyTensor::U32),
            AnyTensor::U64(t) => lift(($f::<u64>)(&t, $($arg),*), AnyTensor::U64),
            AnyTensor::F32(t) => lift(($f::<f32>)(&t, $($arg),*), AnyTensor::F32),
            AnyTensor::F64(t) => lift(($f::<f64>)(&t, $($arg),*), AnyTensor::F64),
            AnyTensor::C32(t) => lift(($f::<Complex<f32>>)(&t, $($arg),*), AnyTensor::C32),
            AnyTensor::C64(t) => lift(($f::<Complex<f64>>)(&t, $($arg),*), AnyTensor::C64),
        }
    };
}

/// Lift a (values, index-output) pair: values keep their dtype, the index
/// tensor goes through `idx_lift_tensor`.
fn pair_lift<T>(
    r: rt::Result<(FTensor<T>, FTensor<usize>)>,
    ctor: impl FnOnce(FTensor<T>) -> AnyTensor,
) -> PyResult<(NativeArray, NativeArray)> {
    let (values, idx) = err_py(r)?;
    Ok((NativeArray { t: ctor(values) }, idx_lift_tensor(idx)?))
}

#[pyfunction]
pub fn unique_values(x: &NativeArray) -> PyResult<NativeArray> {
    dispatch_unique_values!(x.t, op_unique_values()).map(|t| NativeArray { t })
}

/// Returns (values, counts-as-int64); the Python layer wraps the namedtuple.
#[pyfunction]
pub fn unique_counts(x: &NativeArray) -> PyResult<(NativeArray, NativeArray)> {
    macro_rules! pair {
        ($t:expr, $ctor:expr) => {
            pair_lift(op_unique_counts($t), $ctor)
        };
    }
    match &x.t {
        AnyTensor::Bool(t) => pair!(t, AnyTensor::Bool),
        AnyTensor::I8(t) => pair!(t, AnyTensor::I8),
        AnyTensor::I16(t) => pair!(t, AnyTensor::I16),
        AnyTensor::I32(t) => pair!(t, AnyTensor::I32),
        AnyTensor::I64(t) => pair!(t, AnyTensor::I64),
        AnyTensor::U8(t) => pair!(t, AnyTensor::U8),
        AnyTensor::U16(t) => pair!(t, AnyTensor::U16),
        AnyTensor::U32(t) => pair!(t, AnyTensor::U32),
        AnyTensor::U64(t) => pair!(t, AnyTensor::U64),
        AnyTensor::F32(t) => pair!(t, AnyTensor::F32),
        AnyTensor::F64(t) => pair!(t, AnyTensor::F64),
        AnyTensor::C32(t) => pair!(t, AnyTensor::C32),
        AnyTensor::C64(t) => pair!(t, AnyTensor::C64),
    }
}

/// Returns (values, inverse_indices-as-int64); the Python layer wraps the namedtuple.
#[pyfunction]
pub fn unique_inverse(x: &NativeArray) -> PyResult<(NativeArray, NativeArray)> {
    macro_rules! pair {
        ($t:expr, $ctor:expr) => {
            pair_lift(op_unique_inverse($t), $ctor)
        };
    }
    match &x.t {
        AnyTensor::Bool(t) => pair!(t, AnyTensor::Bool),
        AnyTensor::I8(t) => pair!(t, AnyTensor::I8),
        AnyTensor::I16(t) => pair!(t, AnyTensor::I16),
        AnyTensor::I32(t) => pair!(t, AnyTensor::I32),
        AnyTensor::I64(t) => pair!(t, AnyTensor::I64),
        AnyTensor::U8(t) => pair!(t, AnyTensor::U8),
        AnyTensor::U16(t) => pair!(t, AnyTensor::U16),
        AnyTensor::U32(t) => pair!(t, AnyTensor::U32),
        AnyTensor::U64(t) => pair!(t, AnyTensor::U64),
        AnyTensor::F32(t) => pair!(t, AnyTensor::F32),
        AnyTensor::F64(t) => pair!(t, AnyTensor::F64),
        AnyTensor::C32(t) => pair!(t, AnyTensor::C32),
        AnyTensor::C64(t) => pair!(t, AnyTensor::C64),
    }
}

/// Returns (values, indices, inverse_indices, counts), index outputs as
/// int64; the Python layer wraps the namedtuple.
#[pyfunction]
pub fn unique_all(x: &NativeArray) -> PyResult<(NativeArray, NativeArray, NativeArray, NativeArray)> {
    macro_rules! quad {
        ($t:expr, $ctor:expr) => {{
            let (v, i, inv, c) = err_py(op_unique_all($t))?;
            Ok((NativeArray { t: $ctor(v) }, idx_lift_tensor(i)?, idx_lift_tensor(inv)?, idx_lift_tensor(c)?))
        }};
    }
    match &x.t {
        AnyTensor::Bool(t) => quad!(t, AnyTensor::Bool),
        AnyTensor::I8(t) => quad!(t, AnyTensor::I8),
        AnyTensor::I16(t) => quad!(t, AnyTensor::I16),
        AnyTensor::I32(t) => quad!(t, AnyTensor::I32),
        AnyTensor::I64(t) => quad!(t, AnyTensor::I64),
        AnyTensor::U8(t) => quad!(t, AnyTensor::U8),
        AnyTensor::U16(t) => quad!(t, AnyTensor::U16),
        AnyTensor::U32(t) => quad!(t, AnyTensor::U32),
        AnyTensor::U64(t) => quad!(t, AnyTensor::U64),
        AnyTensor::F32(t) => quad!(t, AnyTensor::F32),
        AnyTensor::F64(t) => quad!(t, AnyTensor::F64),
        AnyTensor::C32(t) => quad!(t, AnyTensor::C32),
        AnyTensor::C64(t) => quad!(t, AnyTensor::C64),
    }
}

/// Same-dtype x1/x2 (cross-dtype pairs need promotion rstsr does not provide
/// for isin; G-009); the Python layer pre-casts the scalar form.
#[pyfunction]
pub fn isin(x1: &NativeArray, x2: &NativeArray, invert: bool) -> PyResult<NativeArray> {
    macro_rules! arms {
        ($($dv:ident);* $(;)?) => {
            match (&x1.t, &x2.t) {
                $((AnyTensor::$dv(a), AnyTensor::$dv(b)) =>
                    lift(op_isin(a, b, invert), AnyTensor::Bool).map(|t| NativeArray { t }),)*
                _ => type_err(
                    "isin: mixed-dtype operands are not provided by rstsr (gap G-009); \
                     use matching dtypes or cast first",
                ),
            }
        };
    }
    arms!(Bool; I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
}
