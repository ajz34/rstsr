//! Manipulation surface (W4 + W6): the array-API manipulation functions
//! rstsr-core provides — joins, splits, axis moves, broadcasts, and the
//! repeat/roll/tile family.
//!
//! Every entry point is a thin wrapper: marshalling up, one `rt::` call,
//! `.into_owned()` where rstsr returns a view (copy semantics, register
//! G-036). Joins (`concat`/`stack`/`meshgrid`) are same-dtype only: rstsr's
//! kernels carry one dtype parameter, so cross-dtype joins need promotion,
//! which the shim does not implement (register G-009).

use core::mem::MaybeUninit;
use num::Complex;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_dtype_traits::ExtZero;

use rstsr_core::operators::assignment::{OpAssignAPI, OpAssignArbitaryAPI};
use rstsr_core::operators::ops::OpSubAPI;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};

use crate::any_tensor::{
    device_faer, dispatch_name, dispatch_name_many, dispatch_t, err_py, lift, lift_vec, liftp, type_err, AnyTensor,
    AnyTensorRef, FTensor, NativeArray,
};

// AnyTensorRef (typed_part/tensor_ref) drives diff's same-dtype side args.

/* #region homogeneous-part helpers */

/// Typed borrow of one operand of a same-dtype join; a mismatched variant is
/// declined with the G-009 register reference (never promoted shim-side).
fn typed_part<'a, T: 'static>(part: &'a NativeArray, op: &str) -> PyResult<&'a FTensor<T>>
where
    AnyTensor: AnyTensorRef<T>,
{
    part.t.tensor_ref().ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{op}: all arrays must share one dtype — cross-dtype joins need type promotion, \
             which rstsr does not provide (register G-009); cast with astype() first"
        ))
    })
}

fn collect_parts<'b, T: 'static>(parts: &[&'b NativeArray], op: &str) -> PyResult<Vec<&'b FTensor<T>>>
where
    AnyTensor: AnyTensorRef<T>,
{
    parts.iter().map(|p| typed_part::<T>(p, op)).collect()
}

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

/// Unwrap the pyo3 argument list into plain handle references.
fn handles<'a, 'py>(parts: &'a [PyRef<'py, NativeArray>]) -> Vec<&'a NativeArray> {
    parts.iter().map(|p| &**p).collect()
}

/* #endregion */

/* #region single-tensor layout ops */

// expand_dims / squeeze / flip / moveaxis touch the layout only; rstsr returns
// views, materialized here (no shared storage in the handle model, G-036).

fn op_expand_dims<T>(t: &FTensor<T>, axes: AxesIndex<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    Ok(rt::expand_dims_f(t, axes)?.into_owned())
}

fn op_squeeze<T>(t: &FTensor<T>, axes: AxesIndex<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    Ok(rt::squeeze_f(t, axes)?.into_owned())
}

fn op_flip<T>(t: &FTensor<T>, axes: AxesIndex<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    Ok(rt::flip_f(t, axes)?.into_owned())
}

fn op_moveaxis<T>(t: &FTensor<T>, source: AxesIndex<isize>, destination: AxesIndex<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    Ok(rt::moveaxis_f(t, source, destination)?.into_owned())
}

/// `axis=None` (all axes) travels as `AxesIndex::None`; an explicit tuple as
/// `AxesIndex::Vec` (an empty tuple is a valid no-op per the spec).
fn axes_arg(axes: Option<Vec<isize>>) -> AxesIndex<isize> {
    match axes {
        Some(v) => AxesIndex::Vec(v),
        None => AxesIndex::None,
    }
}

#[pyfunction]
pub fn expand_dims(x: &NativeArray, axes: Vec<isize>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_expand_dims(AxesIndex::Vec(axes.clone())))? })
}

#[pyfunction]
pub fn squeeze(x: &NativeArray, axes: Option<Vec<isize>>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_squeeze(axes_arg(axes.clone())))? })
}

#[pyfunction]
pub fn flip(x: &NativeArray, axes: Option<Vec<isize>>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_flip(axes_arg(axes.clone())))? })
}

#[pyfunction]
pub fn moveaxis(x: &NativeArray, source: Vec<isize>, destination: Vec<isize>) -> PyResult<NativeArray> {
    let t = dispatch_t!(x.t, op_moveaxis(AxesIndex::Vec(source.clone()), AxesIndex::Vec(destination.clone())))?;
    Ok(NativeArray { t })
}

/* #endregion */

/* #region broadcast_shapes */

/// Pure shape function over rstsr's own broadcasting rule; the device default
/// order is the only order the shim exposes.
#[pyfunction]
pub fn broadcast_shapes(shapes: Vec<Vec<usize>>) -> PyResult<Vec<usize>> {
    // IxD is Vec<usize>, so the Python shapes pass straight into rstsr
    err_py(rt::broadcast_shapes_f(&shapes, device_faer().default_order()))
}

/* #endregion */

/* #region joins */

fn op_concat<T>(parts: &[&NativeArray], axis: isize) -> PyResult<FTensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    AnyTensor: AnyTensorRef<T>,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let refs = collect_parts::<T>(parts, "concat")?;
    err_py(rt::concat_f((refs, axis)))
}

fn op_stack<T>(parts: &[&NativeArray], axis: isize) -> PyResult<FTensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    AnyTensor: AnyTensorRef<T>,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let refs = collect_parts::<T>(parts, "stack")?;
    err_py(rt::stack_f((refs, axis)))
}

fn op_meshgrid<T>(parts: &[&NativeArray], indexing: &str) -> PyResult<Vec<FTensor<T>>>
where
    T: Clone + Send + Sync + 'static,
    AnyTensor: AnyTensorRef<T>,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    let refs = collect_parts::<T>(parts, "meshgrid")?;
    // copy=true: fresh owned grids (grids share the input's elements, and the
    // handle model has no shared storage — G-036).
    let grids = err_py(rt::meshgrid_f((refs, indexing, true)))?;
    Ok(grids.into_iter().map(|g| g.into_owned()).collect())
}

#[pyfunction]
pub fn concat<'py>(parts: Vec<PyRef<'py, NativeArray>>, axis: isize) -> PyResult<NativeArray> {
    let refs = handles(&parts);
    if refs.is_empty() {
        return value_err("concat: at least one array is required");
    }
    let name = refs[0].t.dtype_name();
    Ok(NativeArray { t: dispatch_name!(name, op_concat(&refs, axis))? })
}

#[pyfunction]
pub fn stack<'py>(parts: Vec<PyRef<'py, NativeArray>>, axis: isize) -> PyResult<NativeArray> {
    let refs = handles(&parts);
    if refs.is_empty() {
        return value_err("stack: at least one array is required");
    }
    let name = refs[0].t.dtype_name();
    Ok(NativeArray { t: dispatch_name!(name, op_stack(&refs, axis))? })
}

#[pyfunction]
pub fn meshgrid<'py>(parts: Vec<PyRef<'py, NativeArray>>, indexing: &str) -> PyResult<Vec<NativeArray>> {
    let refs = handles(&parts);
    // spec: zero vectors is a legal input and yields an empty tuple
    if refs.is_empty() {
        return Ok(Vec::new());
    }
    let name = refs[0].t.dtype_name();
    let out: Vec<AnyTensor> = dispatch_name_many!(name, op_meshgrid(&refs, indexing))?;
    Ok(out.into_iter().map(|t| NativeArray { t }).collect())
}

/* #endregion */

/* #region unstack */

fn op_unstack<T>(x: &NativeArray, axis: isize) -> PyResult<Vec<FTensor<T>>>
where
    T: Clone + Default + Send + Sync + 'static,
    AnyTensor: AnyTensorRef<T>,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let t = typed_part::<T>(x, "unstack")?;
    let views = err_py(rt::unstack_f((t, axis)))?;
    Ok(views.into_iter().map(|v| v.into_owned()).collect())
}

#[pyfunction]
pub fn unstack(x: &NativeArray, axis: isize) -> PyResult<Vec<NativeArray>> {
    let out: Vec<AnyTensor> = dispatch_name_many!(x.t.dtype_name(), op_unstack(x, axis))?;
    Ok(out.into_iter().map(|t| NativeArray { t }).collect())
}

/* #endregion */

/* #region repeat / roll / tile (W6) */

fn op_repeat<T>(t: &FTensor<T>, repeats: &RepeatArg, axis: AxesIndex<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    rt::repeat_f(t, (repeats.clone(), axis))
}

/// `repeats` travels as int-or-list (the Python layer flattens an index array
/// through `tolist`, the `take` precedent); `RepeatArg::All` for the scalar
/// and a length-1 list, `Elems` otherwise (rstsr's length-1 shortcut).
#[pyfunction]
pub fn repeat(
    x: &NativeArray,
    repeats: Option<Vec<usize>>,
    all_count: Option<usize>,
    axis: Option<isize>,
) -> PyResult<NativeArray> {
    let arg = match (all_count, repeats) {
        (Some(n), _) => RepeatArg::All(n),
        (None, Some(v)) if v.len() == 1 => RepeatArg::All(v[0]),
        (None, Some(v)) => RepeatArg::Elems(v),
        (None, None) => return value_err("repeat: repeats is required"),
    };
    // at most one axis (AxesIndex::Vec would be rejected rust-side)
    let axis = match axis {
        Some(a) => AxesIndex::Val(a),
        None => AxesIndex::None,
    };
    Ok(NativeArray { t: dispatch_t!(x.t, op_repeat(&arg, axis))? })
}

/// `axis=None` flattens (AxesIndex::None); an int is a single axis.
#[pyfunction]
pub fn roll(x: &NativeArray, shift: Vec<isize>, axis: Option<Vec<isize>>) -> PyResult<NativeArray> {
    let shift_idx = match shift.len() {
        1 => AxesIndex::Val(shift[0]),
        _ => AxesIndex::Vec(shift),
    };
    let axis_idx = match axis {
        Some(v) if v.len() == 1 => AxesIndex::Val(v[0]),
        Some(v) => AxesIndex::Vec(v),
        None => AxesIndex::None,
    };
    Ok(NativeArray { t: dispatch_t!(x.t, op_roll(shift_idx.clone(), axis_idx.clone()))? })
}

fn op_roll<T>(t: &FTensor<T>, shift: AxesIndex<isize>, axis: AxesIndex<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    rt::roll_f(t, (shift, axis))
}

fn op_tile<T>(t: &FTensor<T>, repetitions: AxesIndex<usize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpAssignArbitaryAPI<T, IxD, IxD>,
{
    rt::tile_f(t, repetitions)
}

/// `repetitions` is int-or-tuple at the spec level; the Python layer always
/// sends a list (a bare int becomes a length-1 list).
#[pyfunction]
pub fn tile(x: &NativeArray, repetitions: Vec<usize>) -> PyResult<NativeArray> {
    let reps = match repetitions.len() {
        1 => AxesIndex::Val(repetitions[0]),
        _ => AxesIndex::Vec(repetitions),
    };
    Ok(NativeArray { t: dispatch_t!(x.t, op_tile(reps.clone()))? })
}

/* #endregion */

/* #region diff (W7) */

fn op_diff<T>(
    t: &FTensor<T>,
    axis: isize,
    n: usize,
    prepend: Option<&FTensor<T>>,
    append: Option<&FTensor<T>>,
) -> rt::Result<FTensor<T>>
where
    T: Clone + Default + PartialEq + ExtZero + core::ops::Sub<Output = T> + Send + Sync + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD> + OpSubAPI<T, T, T, IxD>,
{
    rt::diff_f(t, axis, n, prepend, append)
}

/// prepend/append are same-dtype arrays (cross-dtype sides need promotion,
/// G-009); `n` and axis validation happen rust-side. Bool has no `Sub`, so
/// its arm is declined (numeric dtypes per the spec).
#[pyfunction]
pub fn diff(
    x: &NativeArray,
    axis: isize,
    n: usize,
    prepend: Option<&NativeArray>,
    append: Option<&NativeArray>,
) -> PyResult<NativeArray> {
    macro_rules! arms {
        ($($dv:ident);* $(;)?) => {
            match &x.t {
                $(AnyTensor::$dv(t) => {
                    let pre = prepend.and_then(|p| p.t.tensor_ref());
                    let app = append.and_then(|a| a.t.tensor_ref());
                    if (prepend.is_some() && pre.is_none()) || (append.is_some() && app.is_none()) {
                        return type_err(
                            "diff: prepend/append must have the same dtype as x (cross-dtype sides \
                             need promotion, rstsr gap G-009); cast first",
                        );
                    }
                    lift(op_diff(t, axis, n, pre, app), AnyTensor::$dv).map(|t| NativeArray { t })
                },)*
                AnyTensor::Bool(_) => type_err(
                    "diff: not defined for bool dtype (rstsr's difference kernel is bound on Sub)",
                ),
            }
        };
    }
    arms!(I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
}

/* #endregion */
