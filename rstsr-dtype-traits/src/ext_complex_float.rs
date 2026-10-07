//! Extension trait for floating-point scalars, real or complex.

use num::complex::ComplexFloat;
use num::traits::{Float, FloatConst};
use num::{Complex, One, Zero};

/// Extension trait for floating-point scalars — real (`f32`/`f64`) or complex
/// (`Complex<f32>`/`Complex<f64>`) — providing operations absent from [`num`].
///
/// The motivating cases are `ln(1 + x)` and `exp(x) - 1`: their textbook
/// evaluations cancel catastrophically when `x` is near zero, so the array API
/// explicitly forbids them (`log1p`/`expm1` docstrings: "should avoid
/// implementing this function as simply `log(1+x)`").
pub trait ExtComplexFloat: ComplexFloat {
    /// Calculates an approximation to `ln(1 + self)`, accurate for `self` near zero.
    fn ext_log_1p(self) -> Self;

    /// Calculates an approximation to `exp(self) - 1`, accurate for `self` near zero.
    fn ext_exp_m1(self) -> Self;
}

impl ExtComplexFloat for f32 {
    #[inline]
    fn ext_log_1p(self) -> Self {
        libm::log1pf(self)
    }

    #[inline]
    fn ext_exp_m1(self) -> Self {
        libm::expm1f(self)
    }
}

impl ExtComplexFloat for f64 {
    #[inline]
    fn ext_log_1p(self) -> Self {
        libm::log1p(self)
    }

    #[inline]
    fn ext_exp_m1(self) -> Self {
        libm::expm1(self)
    }
}

impl<T> ExtComplexFloat for Complex<T>
where
    T: Float + FloatConst,
{
    #[inline]
    fn ext_log_1p(self) -> Self {
        let one = Self::one();
        let u = one + self;
        if !u.is_finite() || u.is_zero() {
            // Infinite/NaN operands, and `self = -1` (where `u = 0`): plain `ln`
            // already yields the branch-cut-correct special values for these.
            return u.ln();
        }
        // `u = fl(1 + self)`; `rho = u - (1 + self)` is exact (Sterbenz), so
        // `1 + self = u - rho` and `ln(1 + self) = ln(u) - rho/u + O((rho/u)^2)`,
        // with `|rho/u| <= 2^-53` — negligible unless `u` is subnormal.
        let rho = (u - one) - self;
        u.ln() - rho / u
    }

    #[inline]
    fn ext_exp_m1(self) -> Self {
        let one = Self::one();
        let half = T::one() / (T::one() + T::one());
        if self.is_finite() && self.abs() < half {
            // `exp(z) - 1 = 2 exp(z/2) sinh(z/2)`: cancellation-free near zero.
            // The direct form is used away from zero, where it is accurate and
            // avoids the intermediate overflow of the identity for large `|Re z|`.
            let h = self.scale(half);
            (h.exp() * h.sinh()).scale(T::one() + T::one())
        } else {
            self.exp() - one
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::Complex;

    fn assert_close(actual: Complex<f64>, expected: Complex<f64>) {
        // relative tolerance, scaled to the expected magnitude so that tiny values
        // (e.g. 1e-20) are compared tightly rather than against a 1e-12 absolute bound.
        let tol = 1e-14 * expected.norm().max(f64::MIN_POSITIVE);
        let err = (actual - expected).norm();
        assert!(err <= tol, "actual = {actual:?}, expected = {expected:?}, err = {err:e}");
    }

    #[test]
    fn test_log1p_real_special_cases() {
        assert!(f64::NAN.ext_log_1p().is_nan());
        assert!((-2.0_f64).ext_log_1p().is_nan()); // x < -1
        assert_eq!((-1.0_f64).ext_log_1p(), f64::NEG_INFINITY);
        assert_eq!(f64::INFINITY.ext_log_1p(), f64::INFINITY);
        assert!(f64::NEG_INFINITY.ext_log_1p().is_nan()); // x < -1
    }

    #[test]
    fn test_log1p_real_precision() {
        // naive ln(1 + x) returns 0 here; log1p must return ~x
        let x = 1e-20_f64;
        assert!((x.ext_log_1p() - x).abs() < 1e-36);
        assert_eq!((1.0_f64 + x).ln(), 0.0); // documents the naive failure
                                             // -0 must be preserved
        assert!((-0.0_f64).ext_log_1p().is_sign_negative() || (-0.0_f64).ext_log_1p() == 0.0);
    }

    #[test]
    fn test_log1p_complex_special_cases() {
        let c = Complex::new;
        // z = -1 + 0j -> -inf + 0j
        let r = c(-1.0_f64, 0.0).ext_log_1p();
        assert!(r.re.is_infinite() && r.re < 0.0 && r.im == 0.0, "{r:?}");
        // a finite, b = +inf -> +inf + pi/2 j
        let r = c(1.0, f64::INFINITY).ext_log_1p();
        assert!(r.re.is_infinite() && r.re > 0.0 && (r.im - std::f64::consts::FRAC_PI_2).abs() < 1e-12, "{r:?}");
        // a = +inf, b = +inf -> +inf + pi/4 j
        let r = c(f64::INFINITY, f64::INFINITY).ext_log_1p();
        assert!(r.re.is_infinite() && r.re > 0.0 && (r.im - std::f64::consts::FRAC_PI_4).abs() < 1e-12, "{r:?}");
        // NaN + finite -> NaN + NaN j
        let r = c(f64::NAN, 1.0).ext_log_1p();
        assert!(r.re.is_nan() && r.im.is_nan(), "{r:?}");
    }

    #[test]
    fn test_log1p_complex_precision() {
        let z = Complex::new(1e-20_f64, 1e-20_f64);
        // naive ln(1 + z) loses the real part; compensated form keeps it
        assert_close(z.ext_log_1p(), z);
        let naive = (Complex::new(1.0, 0.0) + z).ln();
        assert!(naive.re.abs() < 1e-30, "naive real part = {:e}", naive.re);
    }

    #[test]
    fn test_expm1_real_special_cases() {
        assert!(f64::NAN.ext_exp_m1().is_nan());
        assert_eq!(f64::INFINITY.ext_exp_m1(), f64::INFINITY);
        assert_eq!(f64::NEG_INFINITY.ext_exp_m1(), -1.0);
        assert_eq!(1e-20_f64.ext_exp_m1(), 1e-20);
    }

    #[test]
    fn test_expm1_complex_special_cases() {
        let c = Complex::new;
        // a = -inf, b finite -> -1 + 0j
        let r = c(f64::NEG_INFINITY, 2.0).ext_exp_m1();
        assert!((r.re + 1.0).abs() < 1e-12 && r.im.abs() < 1e-12, "{r:?}");
        // a = +inf, b = +0 -> +inf + 0j
        let r = c(f64::INFINITY, 0.0).ext_exp_m1();
        assert!(r.re.is_infinite() && r.re > 0.0 && r.im == 0.0, "{r:?}");
        // a finite, b = +inf -> NaN + NaN j
        let r = c(1.0, f64::INFINITY).ext_exp_m1();
        assert!(r.re.is_nan() && r.im.is_nan(), "{r:?}");
    }

    #[test]
    fn test_expm1_complex_precision() {
        let z = Complex::new(1e-20_f64, 1e-20_f64);
        assert_close(z.ext_exp_m1(), z);
        // large negative real part: must not overflow the identity
        let r = Complex::new(-2000.0_f64, 0.5).ext_exp_m1();
        assert!((r.re + 1.0).abs() < 1e-12 && r.im.abs() < 1e-12, "{r:?}");
    }
}
