//! Extension trait for floating-point scalars, real or complex.

use num::complex::ComplexFloat;
use num::traits::{Float, FloatConst};
use num::{Complex, One, Zero};

use crate::c99_complex;

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

    // The operations below are the complex elementary functions whose
    // `num-complex` formulas do not follow C99 Annex G: they return `NaN`
    // components where the standard prescribes a specific `±inf` / `±0` / `π`
    // combination, and `num-complex` offers nothing better (its `FIXME #1284`).
    // For real scalars these delegate to the C99-conformant libm routines, so
    // only complex inputs take the Annex G code path.

    /// Principal square root, with C99 branch-cut and special-value rules.
    fn ext_sqrt(self) -> Self;

    /// Hyperbolic cosine, with C99 special-value rules.
    fn ext_cosh(self) -> Self;

    /// Hyperbolic sine, with C99 special-value rules.
    fn ext_sinh(self) -> Self;

    /// Hyperbolic tangent, with C99 special-value rules.
    fn ext_tanh(self) -> Self;

    /// Tangent, with C99 special-value rules.
    fn ext_tan(self) -> Self;

    /// Principal arc cosine, with C99 branch-cut and special-value rules.
    fn ext_acos(self) -> Self;

    /// Principal arc sine, with C99 branch-cut and special-value rules.
    fn ext_asin(self) -> Self;

    /// Principal inverse hyperbolic cosine, with C99 branch-cut and special-value rules.
    fn ext_acosh(self) -> Self;

    /// Principal inverse hyperbolic sine, with C99 branch-cut and special-value rules.
    fn ext_asinh(self) -> Self;

    /// Principal inverse hyperbolic tangent, with C99 branch-cut and special-value rules.
    fn ext_atanh(self) -> Self;
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

    #[inline]
    fn ext_sqrt(self) -> Self {
        libm::sqrtf(self)
    }

    #[inline]
    fn ext_cosh(self) -> Self {
        libm::coshf(self)
    }

    #[inline]
    fn ext_sinh(self) -> Self {
        libm::sinhf(self)
    }

    #[inline]
    fn ext_tanh(self) -> Self {
        libm::tanhf(self)
    }

    #[inline]
    fn ext_tan(self) -> Self {
        libm::tanf(self)
    }

    #[inline]
    fn ext_acos(self) -> Self {
        libm::acosf(self)
    }

    #[inline]
    fn ext_asin(self) -> Self {
        libm::asinf(self)
    }

    #[inline]
    fn ext_acosh(self) -> Self {
        libm::acoshf(self)
    }

    #[inline]
    fn ext_asinh(self) -> Self {
        libm::asinhf(self)
    }

    #[inline]
    fn ext_atanh(self) -> Self {
        libm::atanhf(self)
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

    #[inline]
    fn ext_sqrt(self) -> Self {
        libm::sqrt(self)
    }

    #[inline]
    fn ext_cosh(self) -> Self {
        libm::cosh(self)
    }

    #[inline]
    fn ext_sinh(self) -> Self {
        libm::sinh(self)
    }

    #[inline]
    fn ext_tanh(self) -> Self {
        libm::tanh(self)
    }

    #[inline]
    fn ext_tan(self) -> Self {
        libm::tan(self)
    }

    #[inline]
    fn ext_acos(self) -> Self {
        libm::acos(self)
    }

    #[inline]
    fn ext_asin(self) -> Self {
        libm::asin(self)
    }

    #[inline]
    fn ext_acosh(self) -> Self {
        libm::acosh(self)
    }

    #[inline]
    fn ext_asinh(self) -> Self {
        libm::asinh(self)
    }

    #[inline]
    fn ext_atanh(self) -> Self {
        libm::atanh(self)
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
        // expm1(±0 ± 0i) = 0 + 0i: the standard fixes both components at +0,
        // while the identity below would return the sign of the input zeros.
        if self.re == T::zero() && self.im == T::zero() {
            return Self::new(T::zero(), T::zero());
        }
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

    #[inline]
    fn ext_sqrt(self) -> Self {
        let (re, im) = c99_complex::csqrt(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_cosh(self) -> Self {
        let (re, im) = c99_complex::ccosh(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_sinh(self) -> Self {
        let (re, im) = c99_complex::csinh(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_tanh(self) -> Self {
        let (re, im) = c99_complex::ctanh(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_tan(self) -> Self {
        let (re, im) = c99_complex::ctan(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_acos(self) -> Self {
        let (re, im) = c99_complex::cacos(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_asin(self) -> Self {
        let (re, im) = c99_complex::casin(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_acosh(self) -> Self {
        let (re, im) = c99_complex::cacosh(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_asinh(self) -> Self {
        let (re, im) = c99_complex::casinh(self.re, self.im);
        Self::new(re, im)
    }

    #[inline]
    fn ext_atanh(self) -> Self {
        let (re, im) = c99_complex::catanh(self.re, self.im);
        Self::new(re, im)
    }
}
