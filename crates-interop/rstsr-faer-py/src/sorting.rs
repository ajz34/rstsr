//! Sorting surface: `sort`/`argsort` over rstsr's `SortArgs` machinery.
//!
//! Real dtypes only per the spec; complex inputs reach the rstsr tensor
//! layer, which declines them (`UnImplemented`, surfaced as ValueError by
//! `err_py`) — the shim adds no gate of its own. `argsort` returns `usize`
//! rust-side, lifted to the namespace's default index dtype (int64).

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::operators::sorting::{OpArgSortAPI, OpSortAPI};
use rstsr_core::storage::exports::DeviceCreationAnyAPI;

use crate::any_tensor::{dispatch_t, lift, AnyTensor, FTensor, NativeArray};
use crate::ops::idx_lift;

fn op_sort<T>(t: &FTensor<T>, args: SortArgs) -> rt::Result<FTensor<T>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer: DeviceCreationAnyAPI<T> + OpSortAPI<T, IxD>,
{
    rt::sort_f(t, args)
}

fn op_argsort<T>(t: &FTensor<T>, args: SortArgs) -> rt::Result<FTensor<usize>>
where
    T: Clone + Send + Sync + 'static,
    DeviceFaer: DeviceCreationAnyAPI<usize> + OpArgSortAPI<T, IxD>,
{
    rt::argsort_f(t, args)
}

/// `argsort` dispatch: all 13 dtypes (complex declines rust-side at runtime),
/// every arm's `usize` result lifted to int64.
macro_rules! dispatch_argsort {
    ($scrut:expr, $f:ident ( $($arg:expr),* )) => {
        match &$scrut {
            AnyTensor::Bool(t) => idx_lift(($f::<bool>)(&t, $($arg),*)),
            AnyTensor::I8(t) => idx_lift(($f::<i8>)(&t, $($arg),*)),
            AnyTensor::I16(t) => idx_lift(($f::<i16>)(&t, $($arg),*)),
            AnyTensor::I32(t) => idx_lift(($f::<i32>)(&t, $($arg),*)),
            AnyTensor::I64(t) => idx_lift(($f::<i64>)(&t, $($arg),*)),
            AnyTensor::U8(t) => idx_lift(($f::<u8>)(&t, $($arg),*)),
            AnyTensor::U16(t) => idx_lift(($f::<u16>)(&t, $($arg),*)),
            AnyTensor::U32(t) => idx_lift(($f::<u32>)(&t, $($arg),*)),
            AnyTensor::U64(t) => idx_lift(($f::<u64>)(&t, $($arg),*)),
            AnyTensor::F32(t) => idx_lift(($f::<f32>)(&t, $($arg),*)),
            AnyTensor::F64(t) => idx_lift(($f::<f64>)(&t, $($arg),*)),
            AnyTensor::C32(t) => idx_lift(($f::<Complex<f32>>)(&t, $($arg),*)),
            AnyTensor::C64(t) => idx_lift(($f::<Complex<f64>>)(&t, $($arg),*)),
        }
    };
}

#[pyfunction]
pub fn sort(x: &NativeArray, axis: isize, descending: bool, stable: bool) -> PyResult<NativeArray> {
    let args = SortArgs { axis, descending, stable };
    Ok(NativeArray { t: dispatch_t!(x.t, op_sort(args.clone()))? })
}

#[pyfunction]
pub fn argsort(x: &NativeArray, axis: isize, descending: bool, stable: bool) -> PyResult<NativeArray> {
    let args = SortArgs { axis, descending, stable };
    dispatch_argsort!(x.t, op_argsort(args.clone()))
}
