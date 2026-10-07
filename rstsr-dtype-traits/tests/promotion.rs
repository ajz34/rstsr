//! Integration tests for the `promotion` module.

#![allow(non_camel_case_types)] // mirrors the module under test

use num::Complex;
use rstsr_dtype_traits::*;

type c32 = Complex<f32>;
type c64 = Complex<f64>;

/// Compile-time completeness check: the pair promotes.
fn promotes<A, B>()
where
    A: DTypePromoteAPI<B>,
{
}

/// Compile-time check with a known promoted result type.
fn promotes_to<A, B, R>()
where
    A: DTypePromoteAPI<B, Res = R>,
{
}

/// The full ordered matrix over the canonical dtypes must promote. This
/// guards the `i8 x i16` pair (and its `DTypeCastAPI<i16> for i8`): on
/// 2025-09-29 a section comment merged with the following impl line and
/// silently dropped it, so every `DTypePromoteAPI`-bound op declined the
/// pair.
#[test]
fn promotion_matrix_is_complete() {
    macro_rules! all_pairs {
        ($($a:ty => $($b:ty),+ ;)+) => {
            $( $( promotes::<$a, $b>(); )+ )+
        };
    }
    all_pairs! {
        bool => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        i8 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        i16 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        i32 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        i64 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        u8 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        u16 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        u32 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        u64 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        f32 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        f64 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        c32 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
        c64 => bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, c32, c64;
    }
}

#[test]
fn promotion_results() {
    promotes_to::<i8, i16, i16>();
    promotes_to::<i8, u8, i16>();
    promotes_to::<i8, u64, f64>();
    promotes_to::<i32, f32, f64>();
    promotes_to::<u64, i64, f64>();
    promotes_to::<f32, f64, f64>();
    promotes_to::<c32, f64, c64>();
    promotes_to::<c32, c64, c64>();
    promotes_to::<bool, i8, i8>();
}

/// Values of the restored `i8 x i16` row (mirror of `i16 x i8`).
#[test]
fn promote_and_cast_i8_i16() {
    assert_eq!(<i8 as DTypePromoteAPI<i16>>::promote_other(300i16), 300i16);
    assert_eq!(<i8 as DTypePromoteAPI<i16>>::promote_self(-5i8), -5i16);
    assert_eq!(<i8 as DTypeCastAPI<i16>>::into_cast(7i8), 7i16);
}
