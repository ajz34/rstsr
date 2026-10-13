//! Predicates, whole-array reductions, index reductions, select and clip.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use core::mem::MaybeUninit;
use num::complex::ComplexFloat;
use num::Complex;
use num::{Float, FromPrimitive, Zero};
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::operators::exports::Op_MutA_RefB_API;
use rstsr_core::operators::reduction::{
    OpAllAPI, OpAnyAPI, OpArgMaxAPI, OpArgMinAPI, OpCountNonZeroAPI, OpCumProdAPI, OpCumSumAPI, OpMaxAPI, OpMeanAPI,
    OpMinAPI, OpProdAPI, OpStdAPI, OpSumAPI, OpVarAPI,
};
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};
use rstsr_core::tensor::operators::exports::TensorNotEqualAPI;
use rstsr_dtype_traits::DTypePromoteAPI;

use rstsr_faer_py_core::dim_from;
use rstsr_faer_py_core::{
    device_faer, dispatch_t_bool, dispatch_t_float_complex_same, dispatch_t_index_ord, dispatch_t_index_zero,
    dispatch_t_numeric_same, dispatch_t_real_numeric_same, dispatch_where, idx_lift, lift, type_err, AnyTensor,
    FTensor, NativeArray,
};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

/// Boolean reduction over axes rebuilt from rstsr's bool-typed `all`/`any`
/// (OpAllAPI): truthiness is derived as `x != 0` first, so the reduction is
/// value-exact on every dtype (gap G-017); axes/keepdims come from
/// `ReduceArgs` (fixes G-041). `AxesIndex::None` yields a 0-d result.
fn op_all_axes<T>(t: &FTensor<T>, axes: Option<Vec<isize>>, keepdims: bool) -> rt::Result<FTensor<bool>>
where
    T: Default + PartialEq + Clone,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAllAPI<bool, IxD, TOut = bool>,
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    // 0-d zero: broadcasts against any shape without promoting the input
    // (a (1,) zero would lift a 0-d input to shape (1,))
    let zero: FTensor<T> = rt::asarray_f((vec![T::default()], dim_from(&[]), device_faer()))?;
    let truthy = rt::not_equal_f(t, &zero)?;
    rt::all_with_args_f(&truthy, reduce_args(axes, keepdims))
}

fn op_any_axes<T>(t: &FTensor<T>, axes: Option<Vec<isize>>, keepdims: bool) -> rt::Result<FTensor<bool>>
where
    T: Default + PartialEq + Clone,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAnyAPI<bool, IxD, TOut = bool>,
    for<'x> &'x FTensor<T>: TensorNotEqualAPI<&'x FTensor<T>, Output = FTensor<bool>>,
{
    let zero: FTensor<T> = rt::asarray_f((vec![T::default()], dim_from(&[]), device_faer()))?;
    let truthy = rt::not_equal_f(t, &zero)?;
    rt::any_with_args_f(&truthy, reduce_args(axes, keepdims))
}

// isnan/isfinite/isinf: rstsr's is_nan_f/is_finite_f/is_inf_f exist only for
// float/complex tensors (TensorIsNanAPI etc. have no int/bool impls). The
// standard defines them on every dtype — ints/bool yield constant arrays
// (false/true/false). Float/complex go through rstsr's own ops (G-017).

fn const_bool(n: usize, v: bool) -> rt::Result<FTensor<bool>> {
    rt::asarray_f((vec![v; n], device_faer()))
}

// --------------------------------------------------- predicates & logicals --

#[pyfunction]
pub fn all(x: &NativeArray, axis: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t_bool!(x.t, op_all_axes(axis.clone(), keepdims))? })
}

#[pyfunction]
pub fn any(x: &NativeArray, axis: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t_bool!(x.t, op_any_axes(axis.clone(), keepdims))? })
}

#[pyfunction]
pub fn isnan(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::F32(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        AnyTensor::F64(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        AnyTensor::C32(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        AnyTensor::C64(v) => lift(rt::is_nan_f(v), AnyTensor::Bool)?,
        other => lift(const_bool(other.size(), false), AnyTensor::Bool)?,
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn isfinite(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::F32(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        AnyTensor::F64(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        AnyTensor::C32(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        AnyTensor::C64(v) => lift(rt::is_finite_f(v), AnyTensor::Bool)?,
        other => lift(const_bool(other.size(), true), AnyTensor::Bool)?,
    };
    Ok(NativeArray { t })
}

#[pyfunction]
pub fn isinf(x: &NativeArray) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::F32(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        AnyTensor::F64(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        AnyTensor::C32(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        AnyTensor::C64(v) => lift(rt::is_inf_f(v), AnyTensor::Bool)?,
        other => lift(const_bool(other.size(), false), AnyTensor::Bool)?,
    };
    Ok(NativeArray { t })
}

// --------------------------------------------------- statistical (W3) ------
//
// Axes reductions over rstsr's `*_with_args` families (ReduceArgs: axes +
// keepdims; VarArgs: + correction). The array-api accumulation rule (integer
// inputs widen to the default integer dtype, `dtype=` selects the
// accumulator) is served Python-side by casting with the existing astype
// path BEFORE the same-dtype reduction — the order the standard itself
// recommends ("the input array should be cast to the specified data type
// before computing the sum"). `AxesIndex::None` (axis=None) reduces all
// axes to a 0-d result; an empty axes list reduces nothing.

fn reduce_args(axes: Option<Vec<isize>>, keepdims: bool) -> ReduceArgs {
    ReduceArgs { axes: axes.map(AxesIndex::Vec).unwrap_or(AxesIndex::None), keepdims }
}

macro_rules! reduce_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(t: &FTensor<T>, axes: Option<Vec<isize>>, keepdims: bool) -> rt::Result<FTensor<T>>
        where
            DeviceFaer: DeviceRawAPI<T, Raw = Vec<T>> + $Trait<T, IxD, TOut = T>,
        {
            rt::$rt(t, reduce_args(axes, keepdims))
        }
    };
}

reduce_wrapper!(op_sum_axes, sum_with_args_f, OpSumAPI);
reduce_wrapper!(op_prod_axes, prod_with_args_f, OpProdAPI);
reduce_wrapper!(op_max_axes, max_with_args_f, OpMaxAPI);
reduce_wrapper!(op_min_axes, min_with_args_f, OpMinAPI);
reduce_wrapper!(op_mean_axes, mean_with_args_f, OpMeanAPI);

/// Variance over axes; rstsr's `OpVarAPI::TOut` is the component float type
/// (`T::Real`), so a complex input yields a real-dtype result.
fn op_var_axes<T>(
    t: &FTensor<T>,
    axes: Option<Vec<isize>>,
    keepdims: bool,
    correction: Option<f64>,
) -> rt::Result<FTensor<T::Real>>
where
    T: ComplexFloat + FromPrimitive + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Send + Sync + 'static,
    DeviceFaer: OpVarAPI<T, IxD, TOut = T::Real>
        + DeviceCreationAnyAPI<T::Real>
        + Op_MutA_RefB_API<
            T::Real,
            T::Real,
            IxD,
            dyn for<'x, 'y> Fn(&'x mut MaybeUninit<T::Real>, &'y T::Real) + Send + Sync,
        >,
{
    let args = VarArgs { axes: reduce_args(axes, keepdims).axes, keepdims, correction };
    rt::var_with_args_f(t, args)
}

fn op_std_axes<T>(
    t: &FTensor<T>,
    axes: Option<Vec<isize>>,
    keepdims: bool,
    correction: Option<f64>,
) -> rt::Result<FTensor<T::Real>>
where
    T: ComplexFloat + FromPrimitive + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Send + Sync + 'static,
    DeviceFaer: OpStdAPI<T, IxD, TOut = T::Real>
        + DeviceCreationAnyAPI<T::Real>
        + Op_MutA_RefB_API<
            T::Real,
            T::Real,
            IxD,
            dyn for<'x, 'y> Fn(&'x mut MaybeUninit<T::Real>, &'y T::Real) + Send + Sync,
        >,
{
    let args = VarArgs { axes: reduce_args(axes, keepdims).axes, keepdims, correction };
    rt::std_with_args_f(t, args)
}

/// Cumulative scan; axis=None is valid for 1-D input only (the rust-side
/// contract mirrors the standard), include_initial grows the axis to M+1.
macro_rules! cumulative_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(t: &FTensor<T>, axis: Option<isize>, include_initial: bool) -> rt::Result<FTensor<T>>
        where
            DeviceFaer: DeviceRawAPI<T, Raw = Vec<T>> + $Trait<T, IxD, TOut = T>,
        {
            rt::$rt(t, CumulativeArgs { axis, include_initial })
        }
    };
}

cumulative_wrapper!(op_cumulative_sum, cumulative_sum_f, OpCumSumAPI);
cumulative_wrapper!(op_cumulative_prod, cumulative_prod_f, OpCumProdAPI);

macro_rules! py_reduce_numeric {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray, axis: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_t_numeric_same!(x.t, stringify!($pyname), $wrapper(axis.clone(), keepdims))?,
                })
            }
        )*
    };
}

py_reduce_numeric!(sum => op_sum_axes, prod => op_prod_axes);

/// max/min: real numeric dtypes only (`ExtReal` kernels; complex ordering is
/// unspecified in the standard and unimplemented in rstsr).
macro_rules! py_reduce_real_numeric {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray, axis: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_t_real_numeric_same!(x.t, stringify!($pyname), $wrapper(axis.clone(), keepdims))?,
                })
            }
        )*
    };
}

py_reduce_real_numeric!(max => op_max_axes, min => op_min_axes);

/// mean: float/complex dtypes (rstsr's `ComplexFloat` kernel; integer inputs
/// are cast to the default float dtype by the Python layer per spec).
#[pyfunction]
pub fn mean(x: &NativeArray, axis: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t_float_complex_same!(x.t, "mean", op_mean_axes(axis.clone(), keepdims))? })
}

/// var/std over real and complex floating dtypes; complex inputs produce
/// real-dtype results (component-float semantics).
macro_rules! py_reduce_varstd {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray, axis: Option<Vec<isize>>, correction: Option<f64>, keepdims: bool) -> PyResult<NativeArray> {
                let t: AnyTensor = match &x.t {
                    AnyTensor::F32(v) => lift($wrapper(v, axis.clone(), keepdims, correction), AnyTensor::F32)?,
                    AnyTensor::F64(v) => lift($wrapper(v, axis.clone(), keepdims, correction), AnyTensor::F64)?,
                    AnyTensor::C32(v) => lift($wrapper(v, axis.clone(), keepdims, correction), AnyTensor::F32)?,
                    AnyTensor::C64(v) => lift($wrapper(v, axis.clone(), keepdims, correction), AnyTensor::F64)?,
                    _ => type_err(format!("{}: real or complex floating dtypes only", stringify!($pyname)))?,
                };
                Ok(NativeArray { t })
            }
        )*
    };
}

py_reduce_varstd!(var => op_var_axes, std => op_std_axes);

macro_rules! py_cumulative {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray, axis: Option<isize>, include_initial: bool) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_t_numeric_same!(x.t, stringify!($pyname), $wrapper(axis, include_initial))?,
                })
            }
        )*
    };
}

py_cumulative!(cumulative_sum => op_cumulative_sum, cumulative_prod => op_cumulative_prod);

// ---------------------------------------------- index reductions (W5) ------
//
// argmax / argmin / count_nonzero. rstsr's index-reduction kernels return
// `usize` tensors; the standard requires the default index dtype (int64), so
// every result is re-materialized as int64 — a lossless element cast of the
// same kind the astype path performs (G-007/G-008), not an algorithm.

fn op_argmax<T>(t: &FTensor<T>, args: ReduceArgs) -> rt::Result<FTensor<usize>>
where
    T: Clone + PartialOrd + Send + Sync + 'static,
    DeviceFaer: OpArgMaxAPI<T, IxD, TOut = usize> + DeviceCreationAnyAPI<usize>,
{
    rt::argmax_with_args_f(t, args)
}

fn op_argmin<T>(t: &FTensor<T>, args: ReduceArgs) -> rt::Result<FTensor<usize>>
where
    T: Clone + PartialOrd + Send + Sync + 'static,
    DeviceFaer: OpArgMinAPI<T, IxD, TOut = usize> + DeviceCreationAnyAPI<usize>,
{
    rt::argmin_with_args_f(t, args)
}

fn op_count_nonzero<T>(t: &FTensor<T>, args: ReduceArgs) -> rt::Result<FTensor<usize>>
where
    T: Clone + PartialEq + Zero + Send + Sync + 'static,
    DeviceFaer: OpCountNonZeroAPI<T, IxD, TOut = usize> + DeviceCreationAnyAPI<usize>,
{
    rt::count_nonzero_with_args_f(t, args)
}

macro_rules! py_index_reduce {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x: &NativeArray, axis: Option<isize>, keepdims: bool) -> PyResult<NativeArray> {
                dispatch_t_index_ord!(x.t, $wrapper(reduce_args(axis.map(|a| vec![a]), keepdims)))
            }
        )*
    };
}

py_index_reduce!(argmax => op_argmax, argmin => op_argmin);

#[pyfunction]
pub fn count_nonzero(x: &NativeArray, axes: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
    dispatch_t_index_zero!(x.t, op_count_nonzero(reduce_args(axes.clone(), keepdims)))
}

/// `count_nonzero` on bool tensors — **the one documented exception** in the
/// index family: rstsr's generic `count_nonzero` kernel is bound on `Zero`
/// (no bool impl), but rstsr serves bool counting through its bool-specialized
/// sum, `TensorSumBoolAPI::sum_with_args_f` (`OpSumBoolAPI`, `TOut = usize`).
/// Counting `True`s is the 0/1 sum by definition, so the Python layer routes
/// bool inputs here instead of casting — a rust-backed kernel, not a shim-side
/// fallback. See `api.py::count_nonzero`, which carries the same note for the
/// Python reader.
#[pyfunction]
pub fn sum_bool(x: &NativeArray, axes: Option<Vec<isize>>, keepdims: bool) -> PyResult<NativeArray> {
    match &x.t {
        AnyTensor::Bool(t) => idx_lift(t.sum_with_args_f(reduce_args(axes.clone(), keepdims))),
        _ => type_err("sum_bool: only boolean arrays are allowed (count_nonzero's bool path)"),
    }
}

// --------------------------------------------------- element-wise select --
//
// The `where` slice of the searching surface (register G-037).
//
// `where(cond, x, y)`: the condition must be a boolean tensor (spec: "should
// have a boolean data type" — declined, never truthiness-cast); x/y may be any
// dtype pair in rstsr's promotion matrix. The promoted result type comes from
// rstsr's own `DTypePromoteAPI`, so no dtype table is duplicated in the shim.

fn op_where<TX, TY>(
    cond: &FTensor<bool>,
    x: &FTensor<TX>,
    y: &FTensor<TY>,
) -> rt::Result<FTensor<<TX as DTypePromoteAPI<TY>>::Res>>
where
    TX: DTypePromoteAPI<TY>,
    for<'x> &'x FTensor<bool>:
        TensorWhereAPI<&'x FTensor<TX>, &'x FTensor<TY>, Output = FTensor<<TX as DTypePromoteAPI<TY>>::Res>>,
{
    rt::where_f(cond, x, y)
}

/// Rust keyword `where` forces the `where_` item name; the Python-visible name
/// is restored by the pyfunction attribute.
#[pyfunction(name = "where")]
pub fn where_(cond: &NativeArray, x: &NativeArray, y: &NativeArray) -> PyResult<NativeArray> {
    match &cond.t {
        AnyTensor::Bool(c) => Ok(NativeArray { t: dispatch_where!(c, x.t, y.t, op_where)? }),
        _ => type_err("where: condition must have a boolean data type"),
    }
}

// ------------------------------------------------------------------ clip -----

/// Same-dtype clip wrapper. The Python shim casts both bounds to `x.dtype`
/// (the array API's clip keeps `x`'s dtype; bounds do not promote), so the
/// native op sees three same-dtype operands; an absent bound is `None`.
fn op_clip<T>(x: &FTensor<T>, lo: Option<&FTensor<T>>, hi: Option<&FTensor<T>>) -> rt::Result<FTensor<T>>
where
    for<'x> &'x FTensor<T>: TensorClipAPI<TensorView<'x, T>, TensorView<'x, T>, Output = FTensor<T>>,
    for<'x> &'x FTensor<T>: TensorClipAPI<TensorView<'x, T>, Option<T>, Output = FTensor<T>>,
    for<'x> &'x FTensor<T>: TensorClipAPI<Option<T>, TensorView<'x, T>, Output = FTensor<T>>,
    for<'x> &'x FTensor<T>: TensorClipAPI<Option<T>, Option<T>, Output = FTensor<T>>,
{
    match (lo, hi) {
        (Some(l), Some(h)) => rt::clip_f(x, (l.view(), h.view())),
        (Some(l), None) => rt::clip_f(x, (l.view(), Option::<T>::None)),
        (None, Some(h)) => rt::clip_f(x, (Option::<T>::None, h.view())),
        (None, None) => rt::clip_f(x, (Option::<T>::None, Option::<T>::None)),
    }
}

macro_rules! clip_arm {
    ($v:expr, $variant:ident, $ty:ty, $ctor:path, $min:expr, $max:expr) => {{
        let lo = match $min {
            None => None,
            Some(na) => match &na.t {
                AnyTensor::$variant(t) => Some(t),
                _ => return type_err("clip: min dtype must match the input dtype"),
            },
        };
        let hi = match $max {
            None => None,
            Some(na) => match &na.t {
                AnyTensor::$variant(t) => Some(t),
                _ => return type_err("clip: max dtype must match the input dtype"),
            },
        };
        lift(op_clip::<$ty>($v, lo, hi), $ctor)
    }};
}

/// Element-wise clip; bounds are the same dtype as `x` (the Python shim casts
/// them), and `min`/`max` are `None` when the caller omitted that bound. With
/// both absent the result is a copy of `x` (array-API semantics).
#[pyfunction]
pub fn clip(x: &NativeArray, min: Option<&NativeArray>, max: Option<&NativeArray>) -> PyResult<NativeArray> {
    let t: AnyTensor = match &x.t {
        AnyTensor::I8(v) => clip_arm!(v, I8, i8, AnyTensor::I8, min, max)?,
        AnyTensor::I16(v) => clip_arm!(v, I16, i16, AnyTensor::I16, min, max)?,
        AnyTensor::I32(v) => clip_arm!(v, I32, i32, AnyTensor::I32, min, max)?,
        AnyTensor::I64(v) => clip_arm!(v, I64, i64, AnyTensor::I64, min, max)?,
        AnyTensor::U8(v) => clip_arm!(v, U8, u8, AnyTensor::U8, min, max)?,
        AnyTensor::U16(v) => clip_arm!(v, U16, u16, AnyTensor::U16, min, max)?,
        AnyTensor::U32(v) => clip_arm!(v, U32, u32, AnyTensor::U32, min, max)?,
        AnyTensor::U64(v) => clip_arm!(v, U64, u64, AnyTensor::U64, min, max)?,
        AnyTensor::F32(v) => clip_arm!(v, F32, f32, AnyTensor::F32, min, max)?,
        AnyTensor::F64(v) => clip_arm!(v, F64, f64, AnyTensor::F64, min, max)?,
        _ => return type_err("clip: only real dtypes are supported"),
    };
    Ok(NativeArray { t })
}
