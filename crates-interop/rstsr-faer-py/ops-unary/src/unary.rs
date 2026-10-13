//! Unary elementwise surface: transcendental, rounding and numeric kernels.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::tensor::operators::exports::{
    TensorExtNegAPI, TensorPositiveAPI, TensorReciprocalAPI, TensorSquareAPI,
};
use rstsr_dtype_traits::DTypeIntoFloatAPI;

use rstsr_faer_py_core::{
    dispatch_t_float_complex_same, dispatch_t_into_float, dispatch_t_numeric_same, dispatch_t_real_bool,
    dispatch_t_real_numeric_same, lift, type_err, AnyTensor, FTensor, NativeArray,
};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

fn op_ext_neg<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'a> &'a FTensor<T>: TensorExtNegAPI<Output = FTensor<T>>,
{
    rt::ext_neg_f(t)
}

// ------------------------------------------------- W2 elementwise surface ---
//
// Binding-only additions: every wrapper is a thin call into `rt::`; dtype
// policy lives in the dispatch macros of any_tensor.rs. Divergences that a
// binding cannot fix (integer/complex kernels rstsr lacks, mixed-dtype
// arithmetic) are declined with a register reference instead of worked around.

/// Output-type shape is taken from rstsr's own dtype traits (`FloatType`,
/// promoted `Res`), so no dtype table is duplicated in the shim.
macro_rules! unary_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(t: &FTensor<T>) -> rt::Result<FTensor<T::FloatType>>
        where
            T: DTypeIntoFloatAPI,
            for<'x> &'x FTensor<T>: $Trait<Output = FTensor<T::FloatType>>,
        {
            rt::$rt(t)
        }
    };
}

macro_rules! unary_wrapper_same {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
        where
            for<'x> &'x FTensor<T>: $Trait<Output = FTensor<T>>,
        {
            rt::$rt(t)
        }
    };
}

// transcendental family (bool rejected; integers -> float64)
unary_wrapper!(op_acos, acos_f, TensorAcosAPI);
unary_wrapper!(op_acosh, acosh_f, TensorAcoshAPI);
unary_wrapper!(op_asin, asin_f, TensorAsinAPI);
unary_wrapper!(op_asinh, asinh_f, TensorAsinhAPI);
unary_wrapper!(op_atan, atan_f, TensorAtanAPI);
unary_wrapper!(op_atanh, atanh_f, TensorAtanhAPI);
unary_wrapper!(op_cos, cos_f, TensorCosAPI);
unary_wrapper!(op_cosh, cosh_f, TensorCoshAPI);
unary_wrapper!(op_exp, exp_f, TensorExpAPI);
unary_wrapper!(op_log, log_f, TensorLogAPI);
unary_wrapper!(op_log1p, log1p_f, TensorLog1pAPI);
unary_wrapper!(op_log2, log2_f, TensorLog2API);
unary_wrapper!(op_log10, log10_f, TensorLog10API);
unary_wrapper!(op_reciprocal, reciprocal_f, TensorReciprocalAPI);
unary_wrapper!(op_sin, sin_f, TensorSinAPI);
unary_wrapper!(op_sinh, sinh_f, TensorSinhAPI);
unary_wrapper!(op_sqrt, sqrt_f, TensorSqrtAPI);
unary_wrapper!(op_tan, tan_f, TensorTanAPI);
unary_wrapper!(op_tanh, tanh_f, TensorTanhAPI);
unary_wrapper!(op_expm1, expm1_f, TensorExpm1API);
// dtype-preserving rounding (integers are already integral; `round` also complex)
unary_wrapper_same!(op_ceil, ceil_f, TensorCeilAPI);
unary_wrapper_same!(op_floor, floor_f, TensorFloorAPI);
unary_wrapper_same!(op_trunc, trunc_f, TensorTruncAPI);
unary_wrapper_same!(op_round, round_f, TensorRoundAPI);
// dtype-preserving numeric kernels
unary_wrapper_same!(op_square, square_f, TensorSquareAPI);
unary_wrapper_same!(op_sign, sign_f, TensorSignAPI);
unary_wrapper_same!(op_conj, conj_f, TensorConjAPI);

macro_rules! py_unary_into_float {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_into_float!(x.t, $wrapper())? })
            }
        )*
    };
}

macro_rules! py_unary_real_numeric_same {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_real_numeric_same!(x.t, stringify!($pyname), $wrapper())? })
            }
        )*
    };
}

macro_rules! py_unary_numeric_same {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray { t: dispatch_t_numeric_same!(x.t, stringify!($pyname), $wrapper())? })
            }
        )*
    };
}

py_unary_into_float!(
    acos => op_acos,
    acosh => op_acosh,
    asin => op_asin,
    asinh => op_asinh,
    atan => op_atan,
    atanh => op_atanh,
    cos => op_cos,
    cosh => op_cosh,
    exp => op_exp,
    expm1 => op_expm1,
    log => op_log,
    log2 => op_log2,
    log10 => op_log10,
    log1p => op_log1p,
    reciprocal => op_reciprocal,
    sin => op_sin,
    sinh => op_sinh,
    sqrt => op_sqrt,
    tan => op_tan,
    tanh => op_tanh,
);
py_unary_real_numeric_same!(ceil => op_ceil, floor => op_floor, trunc => op_trunc);
py_unary_numeric_same!(round => op_round);
/// `positive`: identity function, routed through rstsr's `TensorPositiveAPI`
/// (rust-side trait added 2026-10-05; no operator trait bound on the dtype).
fn op_positive<T>(t: &FTensor<T>) -> rt::Result<FTensor<T>>
where
    for<'a> &'a FTensor<T>: TensorPositiveAPI<Output = FTensor<T>>,
{
    rt::positive_f(t)
}

py_unary_numeric_same!(
    square => op_square,
    sign => op_sign,
);
py_unary_numeric_same!(positive => op_positive);

/// `conj`: the complex conjugate, identity for real dtypes. Integer and
/// boolean inputs return a copy — rstsr's `conj` routes integers through the
/// into-float block, which the spec forbids (output dtype must equal the input
/// dtype); register G-052.
#[pyfunction]
pub fn conj(x: &NativeArray) -> PyResult<NativeArray> {
    match &x.t {
        AnyTensor::Bool(_)
        | AnyTensor::I8(_)
        | AnyTensor::I16(_)
        | AnyTensor::I32(_)
        | AnyTensor::I64(_)
        | AnyTensor::U8(_)
        | AnyTensor::U16(_)
        | AnyTensor::U32(_)
        | AnyTensor::U64(_) => Ok(NativeArray { t: x.t.deep_copy() }),
        _ => Ok(NativeArray { t: dispatch_t_float_complex_same!(x.t, "conj", op_conj())? }),
    }
}

/// `signbit`: the IEEE 754 sign bit as a boolean tensor (true for negative
/// values, `-0.0` and negatively-signed NaN; real dtypes only).
fn op_signbit<T>(t: &FTensor<T>) -> rt::Result<FTensor<bool>>
where
    for<'x> &'x FTensor<T>: TensorSignBitAPI<Output = FTensor<bool>>,
{
    rt::signbit_f(t)
}

#[pyfunction]
pub fn signbit(x: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t_real_bool!(x.t, "signbit", op_signbit())? })
}

#[pyfunction]
pub fn real(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::C32(v) => lift(rt::real_f(v), AnyTensor::F32)?,
        AnyTensor::C64(v) => lift(rt::real_f(v), AnyTensor::F64)?,
        _ => return type_err("real: only complex dtypes are allowed"),
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn imag(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::C32(v) => lift(rt::imag_f(v), AnyTensor::F32)?,
        AnyTensor::C64(v) => lift(rt::imag_f(v), AnyTensor::F64)?,
        _ => return type_err("imag: only complex dtypes are allowed"),
    };
    Ok(NativeArray { t })
}

/// `logical_not` (bool) and `bitwise_invert` (integer/bool): same dtype.
#[pyfunction]
pub fn invert(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::Bool(v) => lift(rt::not_f(v), AnyTensor::Bool)?,
        AnyTensor::I8(v) => lift(rt::not_f(v), AnyTensor::I8)?,
        AnyTensor::I16(v) => lift(rt::not_f(v), AnyTensor::I16)?,
        AnyTensor::I32(v) => lift(rt::not_f(v), AnyTensor::I32)?,
        AnyTensor::I64(v) => lift(rt::not_f(v), AnyTensor::I64)?,
        AnyTensor::U8(v) => lift(rt::not_f(v), AnyTensor::U8)?,
        AnyTensor::U16(v) => lift(rt::not_f(v), AnyTensor::U16)?,
        AnyTensor::U32(v) => lift(rt::not_f(v), AnyTensor::U32)?,
        AnyTensor::U64(v) => lift(rt::not_f(v), AnyTensor::U64)?,
        _ => return type_err("invert: only integer or boolean dtypes are allowed"),
    };
    Ok(NativeArray { t })
}

/// negative: numeric dtypes (bool rejected); integers, unsigned included,
/// wrap around the two's-complement modulus, matching NumPy's unary minus
/// (G-044). `__neg__` shares this path.
#[pyfunction]
pub fn negative(x: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t_numeric_same!(x.t, "negative", op_ext_neg())? })
}

/// abs: numeric only; complex abs yields a REAL tensor (device TOut is the
/// component float), so the output variant changes for C32/C64.
#[pyfunction]
pub fn abs(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::Bool(_) => return type_err("abs: not defined for bool dtype"),
        AnyTensor::I8(t) => lift(rt::abs_f(t), AnyTensor::I8)?,
        AnyTensor::I16(t) => lift(rt::abs_f(t), AnyTensor::I16)?,
        AnyTensor::I32(t) => lift(rt::abs_f(t), AnyTensor::I32)?,
        AnyTensor::I64(t) => lift(rt::abs_f(t), AnyTensor::I64)?,
        AnyTensor::U8(t) => lift(rt::abs_f(t), AnyTensor::U8)?,
        AnyTensor::U16(t) => lift(rt::abs_f(t), AnyTensor::U16)?,
        AnyTensor::U32(t) => lift(rt::abs_f(t), AnyTensor::U32)?,
        AnyTensor::U64(t) => lift(rt::abs_f(t), AnyTensor::U64)?,
        AnyTensor::F32(t) => lift(rt::abs_f(t), AnyTensor::F32)?,
        AnyTensor::F64(t) => lift(rt::abs_f(t), AnyTensor::F64)?,
        AnyTensor::C32(t) => lift(rt::abs_f(t), AnyTensor::F32)?,
        AnyTensor::C64(t) => lift(rt::abs_f(t), AnyTensor::F64)?,
    };
    Ok(NativeArray { t })
}
