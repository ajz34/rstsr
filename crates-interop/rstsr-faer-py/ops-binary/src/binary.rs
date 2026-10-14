//! Promoted binary kernels whose result keeps the promoted dtype.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr_core::tensor::operators::exports::{
    TensorATan2API, TensorFloorDivideAPI, TensorMaximumAPI, TensorMinimumAPI,
};
use rstsr_dtype_traits::{DTypeIntoFloatAPI, DTypePromoteAPI};

use rstsr_faer_py_core::{dispatch_bin_promote, lift, type_err, AnyTensor, FTensor, NativeArray};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

// --------------------------------------------------- binary (W2) ------------

/// Mixed-dtype wrappers for kernels whose promoted result keeps the promoted
/// dtype (`maximum`-family) ...
macro_rules! bin_promote_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T, U>(a: &FTensor<T>, b: &FTensor<U>) -> rt::Result<FTensor<<T as DTypePromoteAPI<U>>::Res>>
        where
            T: DTypePromoteAPI<U>,
            for<'x> &'x FTensor<T>: $Trait<&'x FTensor<U>, Output = FTensor<<T as DTypePromoteAPI<U>>::Res>>,
        {
            rt::$rt(a, b)
        }
    };
}

/// ... and for kernels that promote first and then map to the float type
/// (`atan2`-family: `TOut = Res::FloatType`).
macro_rules! bin_promote_float_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T, U>(
            a: &FTensor<T>,
            b: &FTensor<U>,
        ) -> rt::Result<FTensor<<<T as DTypePromoteAPI<U>>::Res as DTypeIntoFloatAPI>::FloatType>>
        where
            T: DTypePromoteAPI<U>,
            <T as DTypePromoteAPI<U>>::Res: DTypeIntoFloatAPI,
            for<'x> &'x FTensor<T>: $Trait<
                &'x FTensor<U>,
                Output = FTensor<<<T as DTypePromoteAPI<U>>::Res as DTypeIntoFloatAPI>::FloatType>,
            >,
        {
            rt::$rt(a, b)
        }
    };
}

bin_promote_wrapper!(op_maximum, maximum_f, TensorMaximumAPI);
bin_promote_wrapper!(op_minimum, minimum_f, TensorMinimumAPI);
bin_promote_wrapper!(op_floor_divide, floor_divide_f, TensorFloorDivideAPI);
bin_promote_float_wrapper!(op_atan2, atan2_f, TensorATan2API);

macro_rules! py_bin_promote {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_promote!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_promote!(
    maximum => op_maximum,
    minimum => op_minimum,
    floor_divide => op_floor_divide,
    atan2 => op_atan2,
);
