//! Promoted binary kernels that map to the promoted float type.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr_core::tensor::operators::exports::{
    TensorCopySignAPI, TensorHypotAPI, TensorLogAddExpAPI, TensorNextAfterAPI,
};
use rstsr_dtype_traits::{DTypeIntoFloatAPI, DTypePromoteAPI};

use rstsr_faer_py_core::{dispatch_bin_promote, lift, type_err, AnyTensor, FTensor, NativeArray};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

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

bin_promote_float_wrapper!(op_copysign, copysign_f, TensorCopySignAPI);
bin_promote_float_wrapper!(op_hypot, hypot_f, TensorHypotAPI);
bin_promote_float_wrapper!(op_nextafter, nextafter_f, TensorNextAfterAPI);
bin_promote_float_wrapper!(op_logaddexp, log_add_exp_f, TensorLogAddExpAPI);

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
    copysign => op_copysign,
    hypot => op_hypot,
    nextafter => op_nextafter,
    logaddexp => op_logaddexp,
);
