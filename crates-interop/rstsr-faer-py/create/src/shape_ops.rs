//! Shape manipulation for the rstsr-faer-py array-API namespace: reshape,
//! transpose, integer indexing on axis 0 and broadcast_to.

use core::mem::MaybeUninit;
use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_common::layout::exports::Indexer;
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::storage::exports::{DeviceCreationAnyAPI, DeviceRawAPI};

use rstsr_faer_py_core::{dispatch_t, lift, AnyTensor, FTensor, NativeArray};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

fn op_reshape<T>(t: &FTensor<T>, shape: Vec<isize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let cow = rt::reshape_f(t, shape)?;
    Ok(cow.into_owned())
}

fn op_transpose<T>(t: &FTensor<T>, axes: Option<Vec<isize>>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    // axes=None: full axis reversal (spec `.T` semantics)
    let reversed: Vec<isize> = (0..t.ndim()).rev().map(|i| i as isize).collect();
    let view = rt::transpose_f(t, axes.unwrap_or(reversed))?;
    Ok(view.into_owned())
}

// ------------------------------------------------------------ manipulation --

#[pyfunction]
pub fn reshape(x: &NativeArray, shape: Vec<isize>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_reshape(shape.clone()))? })
}

#[pyfunction]
pub fn transpose(x: &NativeArray, axes: Option<Vec<isize>>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_transpose(axes.clone()))? })
}

/// Integer index on axis 0, spec semantics (drops the axis). Copy, not a
/// view — aliasing/sharing is unimplemented (register G-036).
fn op_index<T>(t: &FTensor<T>, idx: usize) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    let view = t.i(Indexer::from(idx));
    Ok(view.into_owned())
}

#[pyfunction]
pub fn getitem_int(x: &NativeArray, idx: usize) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_index(idx))? })
}

fn op_broadcast_to<T>(t: &FTensor<T>, shape: Vec<usize>) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync,
    DeviceFaer: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
{
    let v = rt::broadcast_to_f(t, shape)?;
    Ok(v.into_owned())
}

#[pyfunction]
pub fn broadcast_to(x: &NativeArray, shape: Vec<usize>) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_t!(x.t, op_broadcast_to(shape.clone()))? })
}
