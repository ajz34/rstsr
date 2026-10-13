//! Promoted arithmetic: mixed-dtype operands promote to their common dtype.
//!
//! One of the domain sub-crates of `rstsr-faer-py`, split out so that no single
//! `rustc` invocation monomorphizes the whole dtype-dispatch surface.

use num::Complex;
use pyo3::prelude::*;
use rstsr::prelude::rt;
use rstsr_core::tensor::operators::exports::{TensorExtAddAPI, TensorExtDivAPI, TensorExtMulAPI, TensorExtSubAPI};
use rstsr_dtype_traits::DTypePromoteAPI;

use rstsr_faer_py_core::{
    dispatch_bin_promote, dispatch_bin_promote_arith, lift, type_err, AnyTensor, FTensor, NativeArray,
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

bin_promote_wrapper!(op_ext_add, ext_add_f, TensorExtAddAPI);
bin_promote_wrapper!(op_ext_sub, ext_sub_f, TensorExtSubAPI);
bin_promote_wrapper!(op_ext_mul, ext_mul_f, TensorExtMulAPI);
bin_promote_wrapper!(op_ext_div, ext_div_f, TensorExtDivAPI);

// ------------------------------------------------------------ arithmetic ----

// Promoted arithmetic: mixed-dtype operands promote to their common dtype
// (rstsr `ext_add` family); bool is excluded, complex is served same/promoted.
macro_rules! py_bin_promote_arith {
    ($($pyname:ident => $wrapper:ident),* $(,)?) => {
        $(
            #[pyfunction]
            pub fn $pyname(x1: &NativeArray, x2: &NativeArray) -> PyResult<NativeArray> {
                Ok(NativeArray {
                    t: dispatch_bin_promote_arith!(x1.t, x2.t, stringify!($pyname), $wrapper)?,
                })
            }
        )*
    };
}

py_bin_promote_arith!(
    add => op_ext_add,
    subtract => op_ext_sub,
    multiply => op_ext_mul,
    divide => op_ext_div,
);
