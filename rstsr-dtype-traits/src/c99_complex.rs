//! C99 (Annex G) conformant complex elementary functions.
//!
//! Ported from NumPy's `npy_math_complex.c.src` (itself taken from FreeBSD's
//! `msun`), which implements the branch cuts and special-value rules of C99
//! Annex G. `num-complex` deliberately leaves these out — see its
//! `FIXME #1284` — so the naive formulas it provides return `nan + nanj` where
//! the standard prescribes a specific `±inf` / `±0` / `π` combination.
//!
//! Every routine works on `(real, imaginary)` component pairs rather than
//! `Complex<T>`, so that the generic bound stays on the scalar `T`.

// `x - x`, `b - b`, ... are deliberate NaN generators in the C99 reference (they
// also quiet a signaling-NaN operand), not a typo.
#![allow(clippy::eq_op)]

use num::traits::{Float, FloatConst};

#[inline]
fn cast<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

/// `true` for `f32`. A few thresholds are precision-specific in the reference;
/// this selects between the `float` and `double` values.
#[inline]
fn is_single<T: Float>() -> bool {
    core::mem::size_of::<T>() == core::mem::size_of::<f32>()
}

/// `f(a, b) = (hypot(a, b) - b) / 2`, with `hypot(a, b)` passed in.
///
/// Rewritten as `a * a / (hypot(a, b) + b) / 2` when `b > 0` so that the
/// subtraction `hypot - b` cannot cancel. See Hull et al. (1997), pp. 308-309.
#[inline]
fn f_hypot_minus<T: Float>(a: T, b: T, hypot_ab: T) -> T {
    if b < T::zero() {
        (hypot_ab - b) / cast::<T>(2.0)
    } else if b == T::zero() {
        a / cast::<T>(2.0)
    } else {
        a * a / (hypot_ab + b) / cast::<T>(2.0)
    }
}

/// `sum_squares(x, y) = x*x + y*y`, or just `x*x` when `y*y` would underflow.
#[inline]
fn sum_squares<T: Float>(x: T, y: T) -> T {
    if y < T::min_positive_value().sqrt() {
        x * x
    } else {
        x * x + y * y
    }
}

/// `Re(1 / (x + iy)) = x / (x*x + y*y)`, guarding the `inf/inf` cases an
/// infinite component would otherwise produce.
#[inline]
fn real_part_reciprocal<T: Float>(x: T, y: T) -> T {
    if x.is_infinite() {
        return T::one() / x; // ±0
    }
    if y.is_infinite() {
        return x / y / y; // ±0
    }
    x / (x * x + y * y)
}

/// Optimized `log` of a finite `z` with `|z|` above `~1/eps`; returns `(Re, Im)`.
fn clog_for_large_values<T: Float + FloatConst>(x: T, y: T) -> (T, T) {
    let one = T::one();
    let two = cast::<T>(2.0);
    let mut ax = x.abs();
    let mut ay = y.abs();
    if ax < ay {
        core::mem::swap(&mut ax, &mut ay);
    }
    let sqrt_min = T::min_positive_value().sqrt();

    // Divide by `e` before `hypot` to avoid overflow when both are huge
    // (`e > sqrt(2)`, so adding 1 to the log compensates exactly).
    let rr = if ax > T::max_value() / two {
        (x / T::E()).hypot(y / T::E()).ln() + one
    } else if ax > one / sqrt_min || ay < sqrt_min {
        x.hypot(y).ln()
    } else {
        (ax * ax + ay * ay).ln() / two
    };
    (rr, y.atan2(x))
}

/// Outputs of [`do_hard_work`] (Hull et al., "Implementing the complex arcsine
/// and arccosine functions using exception handling").
struct HardWork<T> {
    rx: T,
    b_is_usable: bool,
    b: T,
    sqrt_a2my2: T,
    new_y: T,
}

/// Shared core of `casinh` and `cacos`. Assumes `x, y >= 0` and
/// `x, y < 1/eps`.
fn do_hard_work<T: Float>(x: T, y: T) -> HardWork<T> {
    let one = T::one();
    let two = cast::<T>(2.0);
    let four = cast::<T>(4.0);
    let eps = T::epsilon();
    let a_crossover = cast::<T>(10.0);
    let b_crossover = cast::<T>(0.6417);
    let four_sqrt_min = four * T::min_positive_value().sqrt();

    let r = x.hypot(y + one); // |z + i|
    let s = x.hypot(y - one); // |z - i|
    let mut a = (r + s) / two;
    if a < one {
        a = one; // guard against rounding pushing A below its exact lower bound
    }

    // rx = Re(casinh(z)) = -Im(cacos(y + ix)) = log(A + sqrt(A*A - 1)).
    let rx = if a < a_crossover {
        if y == one && x < eps * eps / cast::<T>(128.0) {
            x.sqrt()
        } else if x >= eps * (y - one).abs() {
            let am1 = f_hypot_minus(x, one + y, r) + f_hypot_minus(x, one - y, s);
            (am1 + (am1 * (a + one)).sqrt()).ln_1p()
        } else if y < one {
            x / ((one - y) * (one + y)).sqrt()
        } else {
            ((y - one) + ((y - one) * (y + one)).sqrt()).ln_1p()
        }
    } else {
        (a + (a * a - one).sqrt()).ln()
    };

    let mut new_y = y;
    if y < four_sqrt_min {
        // `y/A` would underflow; hand a rescaled `y` to the caller's `atan2`.
        let scale = two / eps;
        return HardWork { rx, b_is_usable: false, b: T::zero(), sqrt_a2my2: a * scale, new_y: y * scale };
    }

    let b = y / a;
    let mut b_is_usable = true;
    let mut sqrt_a2my2 = T::zero();
    if b > b_crossover {
        // `asin(B)` would lose precision; use sqrt(A*A - y*y) via `atan2` instead.
        b_is_usable = false;
        if y == one && x < eps / cast::<T>(128.0) {
            sqrt_a2my2 = x.sqrt() * ((a + y) / two).sqrt();
        } else if x >= eps * (y - one).abs() {
            let amy = f_hypot_minus(x, y + one, r) + f_hypot_minus(x, y - one, s);
            sqrt_a2my2 = (amy * (a + y)).sqrt();
        } else if y > one {
            let scale = four / eps / eps;
            sqrt_a2my2 = x * scale * y / ((y + one) * (y - one)).sqrt();
            new_y = y * scale;
        } else {
            sqrt_a2my2 = ((one - y) * (one + y)).sqrt();
        }
    }
    HardWork { rx, b_is_usable, b, sqrt_a2my2, new_y }
}

/// Principal square root of `a + bi`.
pub(super) fn csqrt<T: Float + FloatConst>(a: T, b: T) -> (T, T) {
    let zero = T::zero();
    let two = cast::<T>(2.0);

    if a == zero && b == zero {
        // csqrt(±0 ± 0i) = ±0 ± 0i, sign of the real part preserved.
        return (zero, b);
    }
    if b.is_infinite() {
        // csqrt(x + ∞i) = ∞ + ∞i regardless of finite x.
        return (T::infinity(), b);
    }
    if a.is_nan() {
        return (a, (b - b) / (b - b)); // NaN + NaN i
    }
    if a.is_infinite() {
        return if a.is_sign_negative() {
            // csqrt(-∞ + yi) = 0 + ∞i (sign of y); with y NaN: NaN + ∞i.
            ((b - b).abs(), a.copysign(b))
        } else {
            // csqrt(+∞ + yi) = ∞ + 0i (sign of y); with y NaN: ∞ + NaN i.
            (a, (b - b).copysign(b))
        };
    }

    // Scale to avoid overflow in `a*a` / `b*b` (Algorithm 312, CACM 1967).
    let thresh = T::max_value() / (T::one() + T::SQRT_2());
    let scale = a.abs() >= thresh || b.abs() >= thresh;
    let (a, b) = if scale { (a * cast::<T>(0.25), b * cast::<T>(0.25)) } else { (a, b) };

    let (re, im) = if a >= zero {
        let t = ((a + a.hypot(b)) * cast::<T>(0.5)).sqrt();
        (t, b / (two * t))
    } else {
        let t = ((-a + a.hypot(b)) * cast::<T>(0.5)).sqrt();
        (b.abs() / (two * t), t.copysign(b))
    };
    if scale {
        (re * two, im)
    } else {
        (re, im)
    }
}

/// Hyperbolic cosine of `x + yi`.
pub(super) fn ccosh<T: Float>(x: T, y: T) -> (T, T) {
    let zero = T::zero();
    let half = cast::<T>(0.5);
    // `exp(|x|)/2` is exact enough once `cosh(x)` and `exp(|x|)/2` differ by
    // less than a rounding step; both `f32` and `f64` are safe from 22 on.
    let big = cast::<T>(22.0);
    let xfinite = x.is_finite();
    let yfinite = y.is_finite();

    if xfinite && yfinite {
        if y == zero {
            return (x.cosh(), x * y); // preserves the sign of the zero imaginary
        }
        if x.abs() < big {
            return (x.cosh() * y.cos(), x.sinh() * y.sin());
        }
        let h = x.abs().exp() * half;
        return (h * y.cos(), h.copysign(x) * y.sin());
    }
    if x == zero && !yfinite {
        return (y - y, zero.copysign(x * (y - y)));
    }
    if y == zero && !xfinite {
        return (x * x, zero.copysign(x) * y);
    }
    if xfinite && !yfinite {
        return (y - y, x * (y - y));
    }
    if x.is_infinite() {
        if !yfinite {
            return (x * x, x * (y - y));
        }
        return ((x * x) * y.cos(), x * y.sin());
    }
    ((x * x) * (y - y), (x + x) * (y - y))
}

/// Hyperbolic sine of `x + yi`.
pub(super) fn csinh<T: Float>(x: T, y: T) -> (T, T) {
    let zero = T::zero();
    let half = cast::<T>(0.5);
    let big = cast::<T>(22.0);
    let xfinite = x.is_finite();
    let yfinite = y.is_finite();

    if xfinite && yfinite {
        if y == zero {
            return (x.sinh(), y);
        }
        if x.abs() < big {
            return (x.sinh() * y.cos(), x.cosh() * y.sin());
        }
        let h = x.abs().exp() * half;
        return (h.copysign(x) * y.cos(), h * y.sin());
    }
    if x == zero && !yfinite {
        return (zero.copysign(x * (y - y)), y - y);
    }
    if y == zero && !xfinite {
        if x.is_nan() {
            return (x, y);
        }
        return (x, zero.copysign(y));
    }
    if xfinite && !yfinite {
        return (y - y, x * (y - y));
    }
    if !xfinite && !x.is_nan() {
        if !yfinite {
            return (x * x, x * (y - y));
        }
        return (x * y.cos(), T::infinity() * y.sin());
    }
    ((x * x) * (y - y), (x + x) * (y - y))
}

/// Hyperbolic tangent of `x + yi`, via Kahan's algorithm.
pub(super) fn ctanh<T: Float>(x: T, y: T) -> (T, T) {
    let zero = T::zero();
    let one = T::one();
    let four = cast::<T>(4.0);

    if !x.is_finite() {
        if x.is_nan() {
            return (x, if y == zero { y } else { x * y });
        }
        // ctanh(±∞ + iy) = ±1 + i·0 for finite y. The standard fixes this zero
        // as +0 (only the b = ∞ / NaN variants leave the sign open), so take it
        // from `y` rather than the rounding-sensitive `sin(y)*cos(y)`.
        return (one.copysign(x), zero.copysign(y));
    }
    if !y.is_finite() {
        // ctanh(+0 + i∞) = +0 + NaN i, but for nonzero finite `x` the limit
        // does not exist: ctanh(x + i∞) = NaN + NaN i.
        if x == zero {
            return (x, y - y);
        }
        return (y - y, y - y);
    }

    // |x| >= huge: use sinh²(x) ≈ exp(2|x|)/4 and avoid the overflow that
    // `sinh(x)*sinh(x)` would hit in the general denominator.
    let huge = cast::<T>(if is_single::<T>() { 11.0 } else { 22.0 });
    if x.abs() >= huge {
        let e = (-x.abs()).exp();
        return (one.copysign(x), four * y.sin() * y.cos() * e * e);
    }

    let t = y.tan();
    let beta = one + t * t; // 1 / cos²(y)
    let s = x.sinh();
    let rho = (one + s * s).sqrt(); // cosh(x)
    let denom = one + beta * s * s;
    ((beta * rho * s) / denom, t / denom)
}

/// Tangent of `x + yi`, from `ctan(z) = -i·ctanh(iz)`.
pub(super) fn ctan<T: Float>(x: T, y: T) -> (T, T) {
    let (re, im) = ctanh(-y, x);
    (im, -re)
}

/// Principal arc cosine of `x + yi`.
pub(super) fn cacos<T: Float + FloatConst>(x: T, y: T) -> (T, T) {
    let zero = T::zero();
    let one = T::one();
    let four = cast::<T>(4.0);
    let eps = T::epsilon();
    let recip_epsilon = one / eps;
    let sqrt_6_eps = (cast::<T>(6.0) * eps).sqrt();
    let pio2 = T::FRAC_PI_2();
    let sx = x.is_sign_negative();
    let sy = y.is_sign_negative();
    let ax = x.abs();
    let ay = y.abs();

    if x.is_nan() || y.is_nan() {
        if x.is_infinite() {
            return (y + y, T::neg_infinity()); // NaN - ∞i
        }
        if y.is_infinite() {
            return (x + x, -y); // NaN ∓ ∞i
        }
        if x == zero {
            return (pio2, y + y); // π/2 + NaN i
        }
        return (T::nan(), T::nan());
    }

    if ax > recip_epsilon || ay > recip_epsilon {
        let (wx, wy) = clog_for_large_values(x, y);
        let rx = wy.abs();
        let ry = if sy { wx + T::LN_2() } else { -(wx + T::LN_2()) };
        return (rx, ry);
    }

    // z = 1 exactly; keep the sign of the zero imaginary part.
    if x == one && y == zero {
        return (zero, -y);
    }

    if ax < sqrt_6_eps / four && ay < sqrt_6_eps / four {
        return (pio2 - x, -y);
    }

    let hw = do_hard_work(ay, ax);
    let rx = if hw.b_is_usable {
        if sx {
            (-hw.b).acos()
        } else {
            hw.b.acos()
        }
    } else if sx {
        hw.sqrt_a2my2.atan2(-hw.new_y)
    } else {
        hw.sqrt_a2my2.atan2(hw.new_y)
    };
    let ry = if sy { hw.rx } else { -hw.rx };
    (rx, ry)
}

/// Principal arc sine of `x + yi`, from `casin(z) = i·conj(casinh(i·conj(z)))`.
pub(super) fn casin<T: Float + FloatConst>(x: T, y: T) -> (T, T) {
    let (re, im) = casinh(y, x);
    (im, re)
}

/// Principal inverse hyperbolic cosine of `x + yi`, as `±i·cacos(z)` with
/// `Re >= 0`.
pub(super) fn cacosh<T: Float + FloatConst>(x: T, y: T) -> (T, T) {
    let (rx, ry) = cacos(x, y);
    if rx.is_nan() && ry.is_nan() {
        return (ry, rx);
    }
    if rx.is_nan() {
        return (ry.abs(), rx);
    }
    if ry.is_nan() {
        // cacosh(+0 + NaN i) = NaN ± πi/2: the real part is NaN, the imaginary
        // part keeps the finite `Re(cacos)`.
        return (ry, rx);
    }
    (ry.abs(), rx.copysign(y))
}

/// Principal inverse hyperbolic sine of `x + yi`.
pub(super) fn casinh<T: Float + FloatConst>(x: T, y: T) -> (T, T) {
    let zero = T::zero();
    let one = T::one();
    let four = cast::<T>(4.0);
    let eps = T::epsilon();
    let recip_epsilon = one / eps;
    let sqrt_6_eps = (cast::<T>(6.0) * eps).sqrt();
    let ax = x.abs();
    let ay = y.abs();

    if x.is_nan() || y.is_nan() {
        if x.is_infinite() {
            return (x, y + y); // ±∞ + NaN i
        }
        if y.is_infinite() {
            return (y, x + x); // ±∞ + NaN i
        }
        if y == zero {
            return (x + x, y); // NaN + 0i
        }
        return (T::nan(), T::nan());
    }

    if ax > recip_epsilon || ay > recip_epsilon {
        // casinh(z) = sign(x)·clog(sign(x)·z) + O(1/z²).
        let (wx, wy) = if x.is_sign_negative() { clog_for_large_values(-x, -y) } else { clog_for_large_values(x, y) };
        let wx = wx + T::LN_2();
        return (wx.copysign(x), wy.copysign(y));
    }

    if x == zero && y == zero {
        return (x, y);
    }
    if ax < sqrt_6_eps / four && ay < sqrt_6_eps / four {
        return (x, y);
    }

    let hw = do_hard_work(ax, ay);
    let ry = if hw.b_is_usable { hw.b.asin() } else { hw.new_y.atan2(hw.sqrt_a2my2) };
    (hw.rx.copysign(x), ry.copysign(y))
}

/// Principal inverse hyperbolic tangent of `x + yi`.
pub(super) fn catanh<T: Float + FloatConst>(x: T, y: T) -> (T, T) {
    let zero = T::zero();
    let one = T::one();
    let two = cast::<T>(2.0);
    let four = cast::<T>(4.0);
    let eps = T::epsilon();
    let recips = one / eps;
    let sqrt_3_eps = (cast::<T>(3.0) * eps).sqrt();
    let pio2 = T::FRAC_PI_2();
    let ax = x.abs();
    let ay = y.abs();

    if y == zero && ax <= one {
        return (x.atanh(), y); // real axis inside the branch points
    }
    if x == zero {
        return (x, y.atan()); // pure imaginary axis
    }

    if x.is_nan() || y.is_nan() {
        if x.is_infinite() {
            return (zero.copysign(x), y + y); // ±0 + NaN i
        }
        if y.is_infinite() {
            return (zero.copysign(x), pio2.copysign(y)); // ±0 ± π/2 i
        }
        return (T::nan(), T::nan());
    }

    if ax > recips || ay > recips {
        return (real_part_reciprocal(x, y), pio2.copysign(y));
    }

    if ax < sqrt_3_eps / two && ay < sqrt_3_eps / two {
        return (x, y); // casinh(z) = z + O(z³) near the origin
    }

    let rx = if ax == one && ay < eps {
        (T::LN_2() - ay.ln()) / two
    } else {
        (four * ax / sum_squares(ax - one, ay)).ln_1p() / four
    };

    let ry = if ax == one {
        two.atan2(-ay) / two
    } else if ay < eps {
        (two * ay).atan2((one - ax) * (one + ax)) / two
    } else {
        (two * ay).atan2((one - ax) * (one + ax) - ay * ay) / two
    };

    (rx.copysign(x), ry.copysign(y))
}
