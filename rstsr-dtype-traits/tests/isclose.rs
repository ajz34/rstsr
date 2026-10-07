//! Integration tests for the `isclose` module.

use rstsr_dtype_traits::*;
#[test]
fn test_isclose_f64() {
    let a = 1.00001_f64;
    let b = 1.00002_f64;
    let args = None.into();
    assert!(isclose(&a, &b, &args));
    let args = IsCloseArgsBuilder::default().rtol(1.0e-6).atol(1.0e-9).equal_nan(false).build().unwrap();
    assert!(!isclose(&a, &b, &args));
}

#[test]
fn test_isclose_usize() {
    let a: usize = 100;
    let b: usize = 102;
    let args = None.into();
    assert!(!isclose(&a, &b, &args));
}

#[test]
fn test_isclose_usize_c32() {
    use num::Complex;
    let a: usize = 100;
    let b: Complex<f32> = Complex::new(100.0, 0.0);
    let args = None.into();
    assert!(isclose(&a, &b, &args));
    let c: Complex<f32> = Complex::new(100.01, 0.0);
    assert!(!isclose(&a, &c, &args));
}
