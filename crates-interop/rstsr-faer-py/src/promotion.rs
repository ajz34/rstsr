//! Runtime dtype promotion for the Python surface.
//!
//! # Why this exists (and why Rust does not need it)
//!
//! In Rust, dtype promotion is a *compile-time* relation. `DTypePromoteAPI<T>`
//! binds every ordered pair of element types to an associated `Res` type, and a
//! binary operation only type-checks when that bound is satisfied:
//!
//! ```text
//! <f32 as DTypePromoteAPI<i64>>::Res = f64   // the compiler knows this
//! ```
//!
//! There is no runtime "promotion value" to inspect — if a promotion does not
//! exist, the offending code simply does not compile. This is the ordinary
//! guarantee a compiled, statically typed language gives, and it is why the
//! Rust tensor API never needs a `can_cast` or `result_type` function.
//!
//! Python has no such static layer. The array-API namespace must answer
//! promotion questions (`result_type`, `can_cast`) about dtype *tokens* handed
//! to it at runtime — typically typed interactively at a REPL, where no
//! compiler is watching. The spec therefore *mandates* functions for what Rust
//! gets for free from the type system.
//!
//! This module bridges the two: `dtype_promote` re-projects the very same
//! `DTypePromoteAPI` lattice onto a `(&str, &str) -> &str` map, with every arm
//! generated from the associated `Res` type. The Python-side answer is thus
//! produced from the identical trait impls the Rust ops use — not a
//! hand-maintained copy that could drift out of step with them.

use num::Complex;
use pyo3::prelude::*;
use rstsr_dtype_traits::DTypePromoteAPI;

use crate::any_tensor::{type_err, DtypeName};

/// Generate `promote_name` from a table of `a => [b, ...]` rows: each arm reads
/// the promoted type straight off the trait, so the result is never written by
/// hand and cannot drift from the impls the ops use.
macro_rules! define_promote_name {
    ( $( $an:literal $A:ty => [ $($bn:literal $B:ty),* $(,)? ] ),* $(,)? ) => {
        fn promote_name(a: &str, b: &str) -> Option<&'static str> {
            match (a, b) {
                $( $(
                    ($an, $bn) => Some(<<$A as DTypePromoteAPI<$B>>::Res as DtypeName>::NAME),
                )* )*
                _ => None,
            }
        }
    };
}

define_promote_name!(
    "bool" bool => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "int8" i8 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "int16" i16 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "int32" i32 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "int64" i64 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "uint8" u8 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "uint16" u16 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "uint32" u32 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "uint64" u64 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "float32" f32 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "float64" f64 => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "complex64" Complex<f32> => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
    "complex128" Complex<f64> => ["bool" bool, "int8" i8, "int16" i16, "int32" i32, "int64" i64, "uint8" u8, "uint16" u16, "uint32" u32, "uint64" u64, "float32" f32, "float64" f64, "complex64" Complex<f32>, "complex128" Complex<f64>],
);

/// Promoted dtype name of two canonical dtype names.
///
/// Internal to the shim: it is the runtime mirror of the `DTypePromoteAPI`
/// bound (see the module docs), consumed by the Python `result_type` and
/// `can_cast` wrappers. It is deliberately not part of the array-API
/// namespace.
#[pyfunction]
pub fn dtype_promote(a: &str, b: &str) -> PyResult<&'static str> {
    match promote_name(a, b) {
        Some(name) => Ok(name),
        None => type_err(format!("dtype_promote: unknown dtype {a:?} or {b:?}")),
    }
}
