//! Integration tests for the `ext_zero` module.

use rstsr_dtype_traits::*;

#[test]
fn test_ext_zero() {
    assert!(!bool::ext_zero());
    assert_eq!(0, i32::ext_zero());
    assert_eq!(0.0, f64::ext_zero());
    assert_eq!(num::Complex::new(0.0, 0.0), <num::Complex<f64> as ExtZero>::ext_zero());
}
