//! DLPack capsule bridge over `rt::dlpack` (rstsr-cpu-dlpack), driven by `dlpark`'s
//! pyo3 protocol layer.
//!
//! Export: deep copy into a `dltensor_versioned` capsule (IS_COPIED); the capsule
//! destructor runs the producer deleter exactly once, and only for an unconsumed capsule.
//! Import: copy-only — dlpark discovers the producer (DLPack C Exchange API,
//! `__dlpack_device__`, or a legacy `__dlpack__`) and takes the capsule, then
//! rstsr-cpu-dlpack's read-only view adopts it and is gathered into an owned tensor.

use std::mem::MaybeUninit;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;

use rstsr::prelude::rt::dlpack;
use rstsr::prelude::*;

use rstsr_core::operators::exports::OpAssignAPI;
use rstsr_core::storage::exports::DeviceCreationAnyAPI;

use dlpark::ffi::{DLDataTypeCode, DLManagedTensorVersioned};
use dlpark::python::ImportedDlpack;
use dlpark::Managed;

use crate::any_tensor::{err_py, lift, type_err, AnyTensor, FTensor, NativeArray};

// ---------------------------------------------------------------- export ---

/// Copy export from the bridge: the capsule owns the copy (`IS_COPIED`).
fn export_managed<T>(t: &FTensor<T>) -> rt::Result<Managed<DLManagedTensorVersioned>>
where
    T: dlpack::DlpackDtype + Send,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    Ok(dlpack::to_dlpack_copy_f(t)?.into_managed())
}

/// Build the versioned capsule for `__dlpack__`.
///
/// dlpark's `IntoPyObject for Managed<DLManagedTensorVersioned>` wraps the export in a
/// `dltensor_versioned` capsule whose destructor frees exactly an unconsumed capsule.
#[pyfunction]
pub fn dlpack_export<'py>(py: Python<'py>, t: &NativeArray) -> PyResult<Bound<'py, PyAny>> {
    let managed = match &t.t {
        AnyTensor::Bool(t) => err_py(export_managed(t))?,
        AnyTensor::I8(t) => err_py(export_managed(t))?,
        AnyTensor::I16(t) => err_py(export_managed(t))?,
        AnyTensor::I32(t) => err_py(export_managed(t))?,
        AnyTensor::I64(t) => err_py(export_managed(t))?,
        AnyTensor::U8(t) => err_py(export_managed(t))?,
        AnyTensor::U16(t) => err_py(export_managed(t))?,
        AnyTensor::U32(t) => err_py(export_managed(t))?,
        AnyTensor::U64(t) => err_py(export_managed(t))?,
        AnyTensor::F32(t) => err_py(export_managed(t))?,
        AnyTensor::F64(t) => err_py(export_managed(t))?,
        AnyTensor::C32(t) => err_py(export_managed(t))?,
        AnyTensor::C64(t) => err_py(export_managed(t))?,
    };
    managed.into_pyobject(py)
}

// ---------------------------------------------------------------- import ---

fn import_owned<T>(imported: ImportedDlpack) -> rt::Result<FTensor<T>>
where
    T: dlpack::DlpackDtype + Send,
    DeviceFaer:
        DeviceAPI<T, Raw = Vec<T>> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, IxD>,
{
    // dlpark's consumer already owns the producer's tensor; `into_raw` hands the pointer to
    // the bridge's import path, which re-validates it and re-takes ownership — the capsule
    // destructor was neutralized, so this is the only handle. A rejected re-validation frees
    // nothing (the producer keeps its tensor), which is the bridge's documented contract.
    let td = match imported {
        ImportedDlpack::Versioned(m) => unsafe { dlpack::from_dlpack_versioned_f::<T, DeviceFaer, IxD>(m.into_raw())? },
        ImportedDlpack::Legacy(m) => unsafe { dlpack::from_dlpack_legacy_f::<T, DeviceFaer, IxD>(m.into_raw())? },
    };
    // Gathering into an owned tensor drops the view, which runs the producer deleter once.
    Ok(td.to_owned())
}

#[pyfunction]
pub fn dlpack_import(obj: &Bound<'_, PyAny>) -> PyResult<NativeArray> {
    let imported = dlpark::python::from_dlpack(obj.as_borrowed(), None, None)?;
    let dtype =
        imported.validate().map_err(|err| PyValueError::new_err(format!("from_dlpack: DLPack tensor: {err}")))?.dtype();
    if dtype.lanes != 1 {
        return type_err(format!("from_dlpack: dtype lanes = {} is not supported (only scalar lanes)", dtype.lanes));
    }
    let k = |c: DLDataTypeCode| c.0;
    let (code, bits) = (dtype.code.0, dtype.bits);
    let t = match (code, bits) {
        (c, 8) if c == k(DLDataTypeCode::BOOL) => lift(import_owned::<bool>(imported), AnyTensor::Bool),
        (c, 8) if c == k(DLDataTypeCode::INT) => lift(import_owned::<i8>(imported), AnyTensor::I8),
        (c, 16) if c == k(DLDataTypeCode::INT) => lift(import_owned::<i16>(imported), AnyTensor::I16),
        (c, 32) if c == k(DLDataTypeCode::INT) => lift(import_owned::<i32>(imported), AnyTensor::I32),
        (c, 64) if c == k(DLDataTypeCode::INT) => lift(import_owned::<i64>(imported), AnyTensor::I64),
        (c, 8) if c == k(DLDataTypeCode::UINT) => lift(import_owned::<u8>(imported), AnyTensor::U8),
        (c, 16) if c == k(DLDataTypeCode::UINT) => lift(import_owned::<u16>(imported), AnyTensor::U16),
        (c, 32) if c == k(DLDataTypeCode::UINT) => lift(import_owned::<u32>(imported), AnyTensor::U32),
        (c, 64) if c == k(DLDataTypeCode::UINT) => lift(import_owned::<u64>(imported), AnyTensor::U64),
        (c, 16) if c == k(DLDataTypeCode::FLOAT) => {
            return type_err("from_dlpack: float16 is not represented in this shim (G-033)")
        },
        (c, 32) if c == k(DLDataTypeCode::FLOAT) => lift(import_owned::<f32>(imported), AnyTensor::F32),
        (c, 64) if c == k(DLDataTypeCode::FLOAT) => lift(import_owned::<f64>(imported), AnyTensor::F64),
        (c, 64) if c == k(DLDataTypeCode::COMPLEX) => lift(import_owned::<num::Complex<f32>>(imported), AnyTensor::C32),
        (c, 128) if c == k(DLDataTypeCode::COMPLEX) => {
            lift(import_owned::<num::Complex<f64>>(imported), AnyTensor::C64)
        },
        (c, b) => return type_err(format!("from_dlpack: unsupported DLPack dtype (code {c}, bits {b})")),
    };
    Ok(NativeArray { t: t? })
}
