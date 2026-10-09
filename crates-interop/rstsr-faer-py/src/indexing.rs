//! Indexing surface: spec `__getitem__`/`__setitem__` over DeviceFaer.
//!
//! Basic keys (int/slice/newaxis/ellipsis) ride rstsr's layout slicing
//! (`i_f`/`i_mut_f`) and its assign/fill; whole-tensor boolean-mask keys ride
//! `rt::mask_select` / `rt::mask_fill` (rust-side, G-038); keys containing an
//! integer array or a boolean mask ride `rt::array_index` (rust-side array
//! indexing, G-039), and may freely mix basic indexers with them. Results are
//! fresh owned tensors — the handle model has no shared storage (register G-036).

use core::mem::MaybeUninit;
use num::Complex;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyEllipsis, PyNone, PySlice, PyTuple};
use pyo3::Bound;
use rstsr::prelude::rt;
use rstsr::prelude::*;

use rstsr_common::layout::exports::{Indexer, SliceI};
use rstsr_core::operators::adv_indexing::{
    DeviceArrayIndexAPI, DeviceIndexSelectAPI, DeviceMaskIndexAPI, DeviceTakeAlongAxisAPI,
};
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::operators::searching::OpNonzeroAPI;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};

use crate::any_tensor::{dispatch_t, err_py, lift, parse_leaf, type_err, AnyTensor, FTensor, NativeArray, PyScalar};
use crate::creation::{dim_from, ScalarCastTarget};

/* #region key parsing */

enum KeyItem {
    Select(isize),
    Slice(Option<isize>, Option<isize>, Option<isize>),
    NewAxis,
    Ellipsis,
    /// An integer index array (same device as the indexed tensor).
    Array(FTensor<isize>),
    /// A boolean mask (same device as the indexed tensor), consuming one axis
    /// per mask axis.
    Mask(FTensor<bool>),
}

fn sl(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Indexer {
    Indexer::Slice(SliceI::new(start, stop, step))
}

/// Convert an integer index array (any integer dtype) into the `isize` index
/// dtype of rstsr's array indexing. Boolean arrays are routed to
/// [`KeyItem::Mask`] by `parse_key` (and a lone one to [`getitem_mask`] by the
/// Python layer), so only integer dtypes reach here.
fn index_array_to_isize(x: &NativeArray) -> PyResult<FTensor<isize>> {
    match &x.t {
        AnyTensor::I8(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::I16(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::I32(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::I64(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::U8(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::U16(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::U32(t) => Ok(t.mapv(|v| v as isize)),
        AnyTensor::U64(t) => Ok(t.mapv(|v| v as isize)),
        _ => type_err("integer-array (fancy) indexing requires an integer index array"),
    }
}

/// Duck-typed key parse: int / slice / None / Ellipsis / integer array /
/// boolean mask. A lone whole-tensor boolean mask is routed to
/// [`getitem_mask`] / [`setitem_mask`] by the Python layer.
fn parse_key<'py>(key: &Bound<'py, PyTuple>) -> PyResult<Vec<KeyItem>> {
    let mut items = Vec::new();
    for item in key.iter() {
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("boolean scalar indices are not supported"));
        }
        if let Ok(i) = item.extract::<i64>() {
            items.push(KeyItem::Select(i as isize));
        } else if let Ok(s) = item.cast::<PySlice>() {
            let g = |v: Bound<'_, PyAny>| -> PyResult<Option<isize>> { v.extract() };
            let (a, b, c) = (g(s.getattr("start")?)?, g(s.getattr("stop")?)?, g(s.getattr("step")?)?);
            items.push(KeyItem::Slice(a, b, c));
        } else if item.is_instance_of::<PyNone>() {
            items.push(KeyItem::NewAxis);
        } else if item.is_instance_of::<PyEllipsis>() {
            items.push(KeyItem::Ellipsis);
        } else if let Ok(handle) = item.extract::<PyRef<'py, NativeArray>>() {
            match &handle.t {
                AnyTensor::Bool(t) => items.push(KeyItem::Mask(t.clone())),
                _ => items.push(KeyItem::Array(index_array_to_isize(&handle)?)),
            }
        } else {
            return Err(PyTypeError::new_err(format!("invalid index element of type {}", item.get_type().name()?)));
        }
    }
    Ok(items)
}

fn to_indexers(items: &[KeyItem]) -> PyResult<Vec<Indexer>> {
    items
        .iter()
        .map(|it| match it {
            KeyItem::Select(i) => Ok(Indexer::Select(*i)),
            KeyItem::Slice(a, b, c) => Ok(sl(*a, *b, *c)),
            KeyItem::NewAxis => Ok(Indexer::Insert),
            KeyItem::Ellipsis => Ok(Indexer::Ellipsis),
            // integer-array keys are served by `op_getitem`; item assignment
            // with an index array is not implemented (G-039)
            KeyItem::Array(_) => type_err(
                "integer-array (fancy) item assignment is not implemented; only boolean-mask \
                 assignment (x[mask] = value) is supported",
            ),
            KeyItem::Mask(_) => type_err(
                "boolean-mask item assignment is only supported when the mask is the whole key \
                 (x[mask] = value)",
            ),
        })
        .collect::<PyResult<Vec<Indexer>>>()
}

/* #endregion */

/* #region basic */

/// Every `__getitem__` key rides `rt::array_index`: with integer index arrays
/// or boolean masks that is array indexing, and without them it degenerates to
/// basic slicing (a view). Riding one path also means a bad key raises the same
/// error kind in both spellings (NumPy raises `IndexError` either way).
fn op_getitem<T>(t: &FTensor<T>, items: Vec<KeyItem>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<isize>
        + DeviceAPI<bool>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + DeviceCreationAnyAPI<usize>
        + DeviceArrayIndexAPI<T>
        + OpNonzeroAPI<bool, IxD>,
{
    let indexers = items
        .into_iter()
        .map(|it| match it {
            KeyItem::Select(i) => ArrayIndexer::Basic(Indexer::Select(i)),
            KeyItem::Slice(a, b, c) => ArrayIndexer::Basic(sl(a, b, c)),
            KeyItem::NewAxis => ArrayIndexer::Basic(Indexer::Insert),
            KeyItem::Ellipsis => ArrayIndexer::Basic(Indexer::Ellipsis),
            // moved, not cloned: `parse_key` owns the handle's tensor
            KeyItem::Array(idx) => ArrayIndexer::ArrayIndex(idx),
            KeyItem::Mask(m) => ArrayIndexer::ArrayBool(m),
        })
        .collect::<Vec<_>>();
    Ok(rt::array_index_f(t, ArrayIndexArgs::new(indexers))?.into_owned())
}

#[pyfunction]
pub fn getitem_basic(x: &NativeArray, key: &Bound<'_, PyTuple>) -> PyResult<NativeArray> {
    let items = parse_key(key)?;
    Ok(NativeArray { t: dispatch_t!(x.t, op_getitem(items))? })
}

/* #endregion */

/* #region setitem */

fn op_setitem_basic<T>(t: &mut FTensor<T>, src: &FTensor<T>, idx: &[Indexer]) -> rt::Result<()>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let mut view = t.i_mut_f(idx)?;
    view.assign_f(src.view())
}

impl NativeArray {
    /// Basic-key assignment; `value` is an array of the same dtype.
    pub fn setitem_basic_arr(&mut self, key: &Bound<'_, PyTuple>, value: &NativeArray) -> PyResult<()> {
        let items = parse_key(key)?;
        let idx = to_indexers(&items)?;
        macro_rules! arms {
            ($($dv:ident);* $(;)?) => {
                match (&mut self.t, &value.t) {
                    $((AnyTensor::$dv(ref mut a), AnyTensor::$dv(b)) =>
                        err_py(op_setitem_basic(a, b, &idx)),)*
                    _ => type_err(
                        "cross-dtype item assignment is not provided (gap); use matching dtypes",
                    ),
                }
            };
        }
        arms!(Bool; I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
    }

    /// Basic-key assignment with a Python scalar value.
    pub fn setitem_basic_scalar(&mut self, key: &Bound<'_, PyTuple>, value: PyScalar) -> PyResult<()> {
        let items = parse_key(key)?;
        let idx = to_indexers(&items)?;
        match &mut self.t {
            AnyTensor::Bool(ref mut a) => {
                let v = <bool as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I8(ref mut a) => {
                let v = <i8 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I16(ref mut a) => {
                let v = <i16 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I32(ref mut a) => {
                let v = <i32 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::I64(ref mut a) => {
                let v = <i64 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U8(ref mut a) => {
                let v = <u8 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U16(ref mut a) => {
                let v = <u16 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U32(ref mut a) => {
                let v = <u32 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::U64(ref mut a) => {
                let v = <u64 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::F32(ref mut a) => {
                let v = <f32 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::F64(ref mut a) => {
                let v = <f64 as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::C32(ref mut a) => {
                let v = <Complex<f32> as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
            AnyTensor::C64(ref mut a) => {
                let v = <Complex<f64> as ScalarCastTarget>::from_scalar(value)?;
                let mut view = err_py(a.i_mut_f(idx.as_slice()))?;
                err_py(view.fill_f(v))
            },
        }
    }
}

/* #endregion */

/* #region module-level mutation entry points */

#[pyfunction]
pub fn setitem_basic(x: &mut NativeArray, key: &Bound<'_, PyTuple>, value: &NativeArray) -> PyResult<()> {
    x.setitem_basic_arr(key, value)
}

#[pyfunction]
pub fn setitem_scalar(x: &mut NativeArray, key: &Bound<'_, PyTuple>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let s = parse_leaf(value)?;
    x.setitem_basic_scalar(key, s)
}

/* #endregion */

/* #region mask (G-038) */

/// Extract the boolean mask from a key array; integer-array keys stay declined
/// (G-039) and are routed away by the Python layer.
fn as_bool_mask(m: &NativeArray) -> PyResult<&FTensor<bool>> {
    match &m.t {
        AnyTensor::Bool(t) => Ok(t),
        _ => Err(PyTypeError::new_err("boolean-mask indexing requires a bool array")),
    }
}

fn op_getitem_mask<T>(t: &FTensor<T>, m: &FTensor<bool>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<bool>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<T>
        + OpNonzeroAPI<bool, IxD>
        + DeviceMaskIndexAPI<T, IxD, IxD>,
{
    rt::mask_select_f(t, m)
}

#[pyfunction]
pub fn getitem_mask(x: &NativeArray, mask: &NativeArray) -> PyResult<NativeArray> {
    let m = as_bool_mask(mask)?;
    Ok(NativeArray { t: dispatch_t!(x.t, op_getitem_mask(m))? })
}

fn op_setitem_mask<T>(t: &mut FTensor<T>, m: &FTensor<bool>, v: T) -> rt::Result<()>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceAPI<bool> + DeviceRawAPI<MaybeUninit<T>> + DeviceMaskIndexAPI<T, IxD, IxD>,
{
    rt::mask_fill_f(t, m, v)
}

/// `x[mask] = value` with an array value (scalar or size-1, same dtype).
#[pyfunction]
pub fn setitem_mask(x: &mut NativeArray, mask: &NativeArray, value: &NativeArray) -> PyResult<()> {
    let m = as_bool_mask(mask)?;
    macro_rules! arms {
        ($($dv:ident : $ty:ty);* $(;)?) => {
            match (&mut x.t, &value.t) {
                $((AnyTensor::$dv(ref mut a), AnyTensor::$dv(v)) => {
                    let val: $ty = err_py(v.to_scalar_f())?;
                    err_py(op_setitem_mask(a, m, val))
                }),*
                _ => type_err("boolean-mask assignment requires the value dtype to match the array dtype"),
            }
        };
    }
    arms!(Bool: bool; I8: i8; I16: i16; I32: i32; I64: i64; U8: u8; U16: u16; U32: u32; U64: u64; F32: f32; F64: f64; C32: Complex<f32>; C64: Complex<f64>)
}

/// `x[mask] = value` with a Python scalar value.
#[pyfunction]
pub fn setitem_mask_scalar(x: &mut NativeArray, mask: &NativeArray, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let m = as_bool_mask(mask)?;
    let s = parse_leaf(value)?;
    macro_rules! arms {
        ($($dv:ident : $ty:ty);* $(;)?) => {
            match &mut x.t {
                $(AnyTensor::$dv(ref mut a) => {
                    let v = <$ty as ScalarCastTarget>::from_scalar(s)?;
                    err_py(op_setitem_mask(a, m, v))
                }),*
            }
        };
    }
    arms!(Bool: bool; I8: i8; I16: i16; I32: i32; I64: i64; U8: u8; U16: u16; U32: u32; U64: u64; F32: f32; F64: f64; C32: Complex<f32>; C64: Complex<f64>)
}

/* #endregion */

/* #region take */

/// `take`: indices travel as a Python int list (the Python layer flattens the
/// index array through `tolist`), so the shim carries no index-dtype dispatch
/// of its own; rstsr resolves negative indices and checks the bounds.
fn op_take<T>(t: &FTensor<T>, indices: Vec<isize>, axis: isize) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + DeviceIndexSelectAPI<T, IxD>,
{
    rt::take_f(t, indices, axis)
}

#[pyfunction]
pub fn take(x: &NativeArray, indices: Vec<isize>, axis: isize) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_take(indices.clone(), axis))? })
}

/* #endregion */

/* #region take_along_axis (W6) */

/// Indices travel as a Python nested int list (the Python layer flattens the
/// index array through `tolist`), rebuilt as an `isize` tensor of the same
/// shape; rstsr resolves negatives and enforces the same-ndim,
/// broadcast-compatible (outside `axis`) shape contract.
fn op_take_along_axis<T>(
    t: &FTensor<T>,
    indices: Vec<isize>,
    idx_shape: Vec<usize>,
    axis: isize,
) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + DeviceTakeAlongAxisAPI<T, IxD, IxD>,
{
    let idx: FTensor<isize> = rt::asarray_f((indices, dim_from(&idx_shape), t.device()))?;
    rt::take_along_axis_f(t, &idx, axis)
}

#[pyfunction]
pub fn take_along_axis(
    x: &NativeArray,
    indices: Vec<isize>,
    idx_shape: Vec<usize>,
    axis: isize,
) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_take_along_axis(indices.clone(), idx_shape.clone(), axis))? })
}

/* #endregion */
