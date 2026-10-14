//! Integer/bitwise/shift/zeta binary kernels.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr::prelude::*;
use rstsr_core::tensor::operators::exports::{
    TensorExtBitAndAPI, TensorExtBitOrAPI, TensorExtBitXorAPI, TensorExtShlAPI, TensorExtShrAPI, TensorRemAPI,
};
use rstsr_dtype_traits::DTypePromoteAPI;

use rstsr_faer_py_core::{
    dispatch_bin_bool_self, dispatch_bin_numeric_self, dispatch_bin_pow, dispatch_bin_promote,
    dispatch_bin_promote_bitwise, dispatch_bin_promote_int, lift, type_err, AnyTensor, FTensor, NativeArray,
};

// Complex referenced by generated turbofish instantiations in macros.
#[allow(unused)]
fn _complex_used(_c: Complex<f64>) {}

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

/// Same-dtype integer/bitwise binary ops (`remainder`, `bitwise_*`, shifts).
macro_rules! bin_self_wrapper {
    ($wrapper:ident, $rt:ident, $Trait:ident) => {
        fn $wrapper<T>(a: &FTensor<T>, b: &FTensor<T>) -> rt::Result<FTensor<T>>
        where
            for<'x> &'x FTensor<T>: $Trait<&'x FTensor<T>, Output = FTensor<T>>,
        {
            rt::$rt(a, b)
        }
    };
}

bin_self_wrapper!(op_remainder, rem_f, TensorRemAPI);
// promoted bitwise / shifts (mixed int/uint operands promote to a common dtype)
bin_promote_wrapper!(op_bitwise_and, ext_bitand_f, TensorExtBitAndAPI);
bin_promote_wrapper!(op_bitwise_or, ext_bitor_f, TensorExtBitOrAPI);
bin_promote_wrapper!(op_bitwise_xor, ext_bitxor_f, TensorExtBitXorAPI);
bin_promote_wrapper!(op_bitwise_left_shift, ext_shl_f, TensorExtShlAPI);
bin_promote_wrapper!(op_bitwise_right_shift, ext_shr_f, TensorExtShrAPI);

macro_rules! py_bin_self {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_numeric_self!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_self!(remainder => op_remainder);

macro_rules! py_bin_int_bool {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_promote_bitwise!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_int_bool!(
    bitwise_and => op_bitwise_and,
    bitwise_or => op_bitwise_or,
    bitwise_xor => op_bitwise_xor,
);

macro_rules! py_bin_int {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_promote_int!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_int!(
    bitwise_left_shift => op_bitwise_left_shift,
    bitwise_right_shift => op_bitwise_right_shift,
);

bin_promote_wrapper!(op_pow, pow_f, TensorPowAPI);

/// `pow`: promoted result dtype over integer, real and complex operands.
#[pyfunction]
pub fn pow(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
    Ok(NativeArray { t: dispatch_bin_pow!(x1.t, x2.t, "pow", op_pow)? })
}

/// `logical_and/or/xor`: boolean inputs only, boolean output.
macro_rules! py_logical {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_bool_self!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_logical!(
    logical_and => op_bitwise_and,
    logical_or => op_bitwise_or,
    logical_xor => op_bitwise_xor,
);
