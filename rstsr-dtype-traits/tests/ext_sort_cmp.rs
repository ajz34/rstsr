//! Integration tests for the `ext_sort_cmp` module.

use core::cmp::Ordering;
use num::Complex;
use rstsr_dtype_traits::*;

#[test]
fn test_int_ordering() {
    assert_eq!(1.ext_total_cmp(&2), Ordering::Less);
    assert_eq!(true.ext_total_cmp(&false), Ordering::Greater);
}

#[test]
fn test_float_nan_last() {
    assert_eq!((-0.0f64).ext_total_cmp(&0.0), Ordering::Equal);
    assert_eq!(f64::NAN.ext_total_cmp(&f64::INFINITY), Ordering::Greater);
    assert_eq!(f64::NAN.ext_total_cmp(&f64::NAN), Ordering::Equal);
    assert_eq!(1.0f64.ext_total_cmp(&f64::NAN), Ordering::Less);
}

#[cfg(feature = "half")]
#[test]
fn test_half_nan_last() {
    let nan = half::f16::NAN;
    let one = half::f16::ONE;
    assert_eq!(nan.ext_total_cmp(&one), Ordering::Greater);
    assert_eq!(nan.ext_total_cmp(&nan), Ordering::Equal);
}

#[test]
fn test_complex_lexicographic() {
    let a = Complex::new(1.0f64, 2.0);
    let b = Complex::new(1.0, 3.0);
    assert_eq!(a.ext_total_cmp(&b), Ordering::Less);
}

#[test]
fn test_complex_nan_numpy_order() {
    // NumPy ordering (arraytypes.c.src C@TYPE@_compare): finite values
    // first (whatever their parts), NaN-bearing values last, part-wise
    // lexicographic among NaN-bearing values (NaN part after finite).
    let f_inf = Complex::new(f64::INFINITY, 0.0);
    let nan_re = Complex::new(f64::NAN, 0.0);
    let nan_im = Complex::new(0.0f64, f64::NAN);
    let nan_both = Complex::new(f64::NAN, f64::NAN);
    let neg_inf_im = Complex::new(f64::NEG_INFINITY, f64::INFINITY);
    // every finite value sorts before every NaN-bearing one
    assert_eq!(f_inf.ext_total_cmp(&nan_re), Ordering::Less); // (inf,0) < (nan,0)
    assert_eq!(f_inf.ext_total_cmp(&nan_im), Ordering::Less); // (inf,0) < (0,nan)
    assert_eq!(f_inf.ext_total_cmp(&nan_both), Ordering::Less);
    assert_eq!(neg_inf_im.ext_total_cmp(&nan_im), Ordering::Less); // (-inf,inf) < (0,nan)
                                                                   // a NaN-bearing value never
                                                                   // sorts before a finite one,
                                                                   // whatever the parts
    assert_eq!(nan_im.ext_total_cmp(&f_inf), Ordering::Greater); // (0,nan) > (inf,0)
    let two = Complex::new(2.0f64, 0.0);
    assert_eq!(nan_im.ext_total_cmp(&two), Ordering::Greater); // (0,nan) > (2,0)
                                                               // among NaN-bearing: part-wise
                                                               // lexicographic, NaN part after
                                                               // finite
    assert_eq!(nan_im.ext_total_cmp(&nan_re), Ordering::Less); // (0,nan) < (nan,0)
    assert_eq!(nan_re.ext_total_cmp(&nan_both), Ordering::Less); // (nan,0) < (nan,nan)
    assert_eq!(nan_im.ext_total_cmp(&nan_both), Ordering::Less);
    let m1_nan = Complex::new(-1.0f64, f64::NAN);
    assert_eq!(m1_nan.ext_total_cmp(&nan_im), Ordering::Less); // (-1,nan) < (0,nan)
}

#[test]
fn test_complex_nan_equal_parts() {
    // identical NaN placement with equal finite parts compare Equal
    let a = Complex::new(1.0f64, f64::NAN);
    let b = Complex::new(1.0, f64::NAN);
    assert_eq!(a.ext_total_cmp(&b), Ordering::Equal);
}
