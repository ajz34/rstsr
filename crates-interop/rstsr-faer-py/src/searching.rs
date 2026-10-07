//! Searching surface (W6): `searchsorted` and `nonzero` over the rstsr
//! searching kernels.
//!
//! `searchsorted`: x1 must be 1-D and x2 the same dtype (the Python layer
//! pre-casts scalars to x1's dtype as a 0-d array); `side` arrives validated
//! as a bool (true = right), `sorter` as an optional int list. Output is
//! `usize` rust-side, lifted to int64.
//!
//! `nonzero`: all dtypes; returns one int64 index tensor per dimension
//! (strict C order), the Python layer builds the tuple.

use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::operators::searching::{OpNonzeroAPI, OpSearchSortedAPI};
use rstsr_core::storage::exports::DeviceCreationAnyAPI;
use rstsr_dtype_traits::{ExtSortCmp, ExtZero};

use crate::any_tensor::{err_py, AnyTensor, FTensor, NativeArray};
use crate::ops::idx_lift_tensor;

fn op_searchsorted<T>(
    x1: &FTensor<T>,
    x2: &FTensor<T>,
    side_right: bool,
    sorter: Option<Vec<usize>>,
) -> rt::Result<FTensor<usize>>
where
    T: Clone + ExtSortCmp + Send + Sync + 'static,
    DeviceFaer: DeviceCreationAnyAPI<usize> + OpSearchSortedAPI<T, T, IxD>,
{
    rt::searchsorted_f(x1, x2, SearchSortedArgs { side: bool_to_side(side_right), sorter })
}

fn bool_to_side(right: bool) -> SearchSide {
    if right {
        SearchSide::Right
    } else {
        SearchSide::Left
    }
}

/// Same-dtype x1/x2 dispatch; output usize -> int64.
#[pyfunction]
pub fn searchsorted(
    x1: &NativeArray,
    x2: &NativeArray,
    side_right: bool,
    sorter: Option<Vec<usize>>,
) -> PyResult<NativeArray> {
    macro_rules! arms {
        ($($dv:ident);* $(;)?) => {
            match (&x1.t, &x2.t) {
                $((AnyTensor::$dv(a), AnyTensor::$dv(b)) =>
                    idx_lift_tensor_of(op_searchsorted(a, b, side_right, sorter.clone())),)*
                _ => crate::any_tensor::type_err(
                    "searchsorted: x1 and x2 must share one dtype (mixed dtypes need promotion, \
                     rstsr gap G-009); cast first",
                ),
            }
        };
    }
    arms!(Bool; I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
}

/// `idx_lift_tensor` over an `rt::Result` (mirrors `ops::idx_lift`).
fn idx_lift_tensor_of(r: rt::Result<FTensor<usize>>) -> PyResult<NativeArray> {
    idx_lift_tensor(err_py(r)?)
}

fn op_nonzero<T>(t: &FTensor<T>) -> rt::Result<Vec<FTensor<usize>>>
where
    T: ExtZero + PartialEq + Send + Sync,
    DeviceFaer: DeviceCreationAnyAPI<usize> + OpNonzeroAPI<T, IxD>,
{
    rt::nonzero_f(t)
}

/// Returns one index tensor per dimension (int64), C order; 0-d input errors
/// rust-side (spec-aligned deviation).
#[pyfunction]
pub fn nonzero(x: &NativeArray) -> PyResult<Vec<NativeArray>> {
    macro_rules! arms {
        ($($dv:ident);* $(;)?) => {
            match &x.t {
                $(AnyTensor::$dv(t) => {
                    let coords = err_py(op_nonzero(t))?;
                    coords.into_iter().map(idx_lift_tensor).collect()
                },)*
            }
        };
    }
    arms!(Bool; I8; I16; I32; I64; U8; U16; U32; U64; F32; F64; C32; C64)
}
