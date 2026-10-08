use duplicate::duplicate_item;
use num::{complex::ComplexFloat, Complex};

// Extension trait for numerical ([`num::Num`]) types.
pub trait ExtNum: Clone {
    /* #region abs */

    /// The output type of the absolute value operation.
    type AbsOut: Clone + ExtNum<AbsOut = Self::AbsOut>;

    /// Whether the absolute value operation does not change the value.
    const ABS_UNCHANGED: bool;

    /// Whether the output type of the absolute value operation is the same as the input type.
    const ABS_SAME_TYPE: bool;

    /// Computes the absolute value of the number.
    fn ext_abs(self) -> Self::AbsOut;

    /// Computes the absolute difference between two numbers.
    ///
    /// For most cases, this is equivalent to `(self - other).ext_abs()`.
    /// However, for unsigned integer types, this avoids potential underflow.
    fn ext_abs_diff(self, other: Self) -> Self::AbsOut;

    /* #endregion */

    /* #region sign */

    /// Computes the sign of the number, with the same output type as the input.
    ///
    /// This follows NumPy's `np.sign` conventions:
    /// - signed integers: `-1`, `0` or `1` (no overflow at the type's minimum);
    /// - unsigned integers: `0` or `1`;
    /// - floats: `-1.0`, `0.0` or `1.0`; infinities map to `±1.0`, NaN maps to NaN;
    /// - complex: `z / |z|`, and complex zero maps to zero.
    fn ext_sign(self) -> Self;

    /* #endregion */

    /* #region remainder */

    /// Computes the floored remainder — NumPy's `remainder` / Python's `%`
    /// (the array-API `remainder`).
    ///
    /// Unlike Rust's `%`, the result takes the sign of `other`, so that
    /// `ext_floor_divide(x, y) * y + ext_rem(x, y) == x`. For floats this also
    /// covers the array-API special cases: a zero dividend and an infinite
    /// divisor keep `other`'s sign / value. Complex operands have no floored
    /// form and delegate to the Gaussian `%` of [`num::Complex`].
    fn ext_rem(self, other: Self) -> Self;

    /* #endregion */

    /* #region pow */

    /// Raises the number to the power `other` — the array-API `pow`.
    ///
    /// Returns `None` when `other` falls outside this type's integer-power
    /// domain, i.e. a negative exponent for an integer type (the array API
    /// leaves `int ** int` with a negative exponent unspecified; NumPy
    /// rejects it). Float and complex accept any exponent; complex uses the
    /// principal branch `exp(other * ln(self))`.
    fn ext_pow(self, other: Self) -> Option<Self>;

    /* #endregion */

    /* #region round */

    /// Rounds to the nearest integral value, with halfway cases to the even
    /// neighbor — the array-API `round`.
    ///
    /// Identity for integer types. Complex rounds the real and imaginary parts
    /// independently (array-API 2024.12).
    fn ext_round(self) -> Self;

    /* #endregion */

    /* #region real-imag */

    /// Returns the real part of the number.
    fn ext_real(self) -> Self::AbsOut;

    /// Returns the imaginary part of the number.
    fn ext_imag(self) -> Self::AbsOut;

    fn ext_conj(self) -> Self {
        self
    }

    /* #endregion */

    /* #region utilities */

    #[inline]
    fn is_nan(&self) -> bool {
        false
    }

    /* #endregion */
}

#[duplicate_item(T; [u8]; [u16]; [u32]; [u64]; [u128]; [usize];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = true;
    const ABS_SAME_TYPE: bool = true;
    #[inline]
    fn ext_abs(self) -> Self {
        self
    }
    #[inline]
    fn ext_abs_diff(self, other: Self) -> Self {
        self.abs_diff(other)
    }
    /* #endregion */

    /* #region sign */
    #[inline]
    fn ext_sign(self) -> Self {
        if self == 0 {
            0
        } else {
            1
        }
    }
    /* #endregion */

    /* #region remainder */
    #[inline]
    fn ext_rem(self, other: Self) -> Self {
        // no negative values: C and floored remainders coincide
        if other == 0 {
            0
        } else {
            self % other
        }
    }
    /* #endregion */

    /* #region pow */
    #[inline]
    #[allow(clippy::unnecessary_cast)] // identity cast for u32 itself
    fn ext_pow(self, other: Self) -> Option<Self> {
        Some(self.pow(other as u32))
    }
    /* #endregion */

    /* #region round */
    #[inline]
    fn ext_round(self) -> Self {
        self
    }
    /* #endregion */

    /* #region real-imag */
    #[inline]
    fn ext_real(self) -> Self {
        self
    }
    #[inline]
    fn ext_imag(self) -> Self {
        0 as Self
    }
}

#[duplicate_item(T; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = true;
    #[inline]
    fn ext_abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn ext_abs_diff(self, other: Self) -> Self {
        if self >= other {
            self - other
        } else {
            other - self
        }
    }
    /* #endregion */

    /* #region sign */
    #[inline]
    fn ext_sign(self) -> Self {
        // comparison form avoids overflow at the type's minimum (e.g. i8::MIN)
        if self > 0 {
            1
        } else if self < 0 {
            -1
        } else {
            0
        }
    }
    /* #endregion */

    /* #region remainder */
    #[inline]
    fn ext_rem(self, other: Self) -> Self {
        if other == 0 {
            return 0;
        }
        // `% -1` overflows at the type's minimum in debug builds; the quotient is
        // exact, so the remainder is 0.
        if other == -1 {
            return 0;
        }
        let r = self % other;
        // `r + other` cannot overflow: `r` and `other` have opposite signs.
        if r != 0 && (r < 0) != (other < 0) {
            r + other
        } else {
            r
        }
    }
    /* #endregion */

    /* #region pow */
    #[inline]
    fn ext_pow(self, other: Self) -> Option<Self> {
        if other < 0 {
            // the array API leaves this unspecified and NumPy rejects it
            None
        } else {
            Some(self.pow(other as u32))
        }
    }
    /* #endregion */

    /* #region round */
    #[inline]
    fn ext_round(self) -> Self {
        self
    }
    /* #endregion */

    /* #region real-imag */
    #[inline]
    fn ext_real(self) -> Self {
        self
    }
    #[inline]
    fn ext_imag(self) -> Self {
        0 as Self
    }
    /* #endregion */
}

#[duplicate_item(
    T       roundeven;
    [f32]   [libm::roundevenf];
    [f64]   [libm::roundeven];
)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = true;
    #[inline]
    fn ext_abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn ext_abs_diff(self, other: Self) -> Self {
        (self - other).abs()
    }
    /* #endregion */

    /* #region sign */
    #[inline]
    fn ext_sign(self) -> Self {
        // NaN maps to NaN; infinities map to ±1; both zeros map to 0 (NumPy convention)
        if self.is_nan() {
            self
        } else if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    /* #endregion */

    /* #region remainder */
    fn ext_rem(self, other: Self) -> Self {
        let zero = 0.0 as Self;
        if self.is_nan() || other.is_nan() {
            return Self::NAN;
        }
        // infinite dividend (finite divisor) is NaN
        if self.is_infinite() {
            return Self::NAN;
        }
        if other.is_infinite() {
            // finite dividend, infinite divisor: `other` when the signs differ,
            // else `self`; a zero dividend keeps the divisor's sign
            if self == zero {
                return zero.copysign(other);
            }
            return if self.is_sign_positive() == other.is_sign_positive() { self } else { other };
        }
        if other == zero {
            return Self::NAN;
        }
        let r = self % other; // sign of the dividend
        if r == zero {
            // exact multiple (including a ±0 dividend): carry `other`'s sign
            return zero.copysign(other);
        }
        if r.is_sign_positive() != other.is_sign_positive() {
            r + other
        } else {
            r
        }
    }
    /* #endregion */

    /* #region pow */
    #[inline]
    fn ext_pow(self, other: Self) -> Option<Self> {
        Some(self.powf(other))
    }
    /* #endregion */

    /* #region round */
    #[inline]
    fn ext_round(self) -> Self {
        // ties-to-even, exactly (libm is a `no_std` dependency of this crate)
        roundeven(self)
    }
    /* #endregion */

    /* #region real-imag */
    #[inline]
    fn ext_real(self) -> Self {
        self
    }
    #[inline]
    fn ext_imag(self) -> Self {
        0 as Self
    }
    /* #endregion */

    /* #region utilities */
    fn is_nan(&self) -> bool {
        use num::Float;
        Float::is_nan(*self)
    }
    /* #endregion */
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = true;
    #[inline]
    fn ext_abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn ext_abs_diff(self, other: Self) -> Self {
        (self - other).abs()
    }
    /* #endregion */

    /* #region sign */
    #[inline]
    fn ext_sign(self) -> Self {
        // NaN maps to NaN; infinities map to ±1; both zeros map to 0 (NumPy convention)
        if self.is_nan() {
            self
        } else if self > Self::ZERO {
            Self::ONE
        } else if self < Self::ZERO {
            Self::NEG_ONE
        } else {
            Self::ZERO
        }
    }
    /* #endregion */

    /* #region remainder */
    #[inline]
    fn ext_rem(self, other: Self) -> Self {
        // round-trip through f32: exact in, one rounding out
        Self::from_f32(f32::from(self).ext_rem(f32::from(other)))
    }
    /* #endregion */

    /* #region pow */
    #[inline]
    fn ext_pow(self, other: Self) -> Option<Self> {
        Some(Self::from_f32(f32::from(self).powf(f32::from(other))))
    }
    /* #endregion */

    /* #region round */
    #[inline]
    fn ext_round(self) -> Self {
        // round through f32: the result is integral and f16/bf16-exact
        Self::from_f32(libm::roundevenf(f32::from(self)))
    }
    /* #endregion */

    /* #region real-imag */
    #[inline]
    fn ext_real(self) -> Self {
        self
    }
    #[inline]
    fn ext_imag(self) -> Self {
        Self::ZERO
    }
    /* #endregion */

    /* #region utilities */
    #[inline]
    fn is_nan(&self) -> bool {
        use num::Float;
        Float::is_nan(*self)
    }
    /* #endregion */
}

#[duplicate_item(
    T                roundeven;
    [Complex<f32>]   [libm::roundevenf];
    [Complex<f64>]   [libm::roundeven];
)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = <T as ComplexFloat>::Real;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = false;
    #[inline]
    fn ext_abs(self) -> Self::AbsOut {
        self.norm()
    }
    #[inline]
    fn ext_abs_diff(self, other: Self) -> Self::AbsOut {
        (self - other).norm()
    }
    /* #endregion */

    /* #region sign */
    #[inline]
    fn ext_sign(self) -> Self {
        // sign(z) = z / |z|, but 0 / 0 yields NaN, so a zero magnitude maps to
        // 0 (matching NumPy)
        let abs = self.norm();
        if abs == 0.0 {
            Self::ZERO
        } else {
            self / abs
        }
    }
    /* #endregion */

    /* #region remainder */
    #[inline]
    fn ext_rem(self, other: Self) -> Self {
        // complex has no floored remainder; keep num-complex's Gaussian `%`
        self % other
    }
    /* #endregion */

    /* #region pow */
    #[inline]
    fn ext_pow(self, other: Self) -> Option<Self> {
        // principal branch: exp(other * ln(self))
        Some(self.powc(other))
    }
    /* #endregion */

    /* #region round */
    #[inline]
    fn ext_round(self) -> Self {
        // real and imaginary parts round independently (array-API 2024.12)
        Self::new(roundeven(self.re), roundeven(self.im))
    }
    /* #endregion */

    /* #region real-imag */
    #[inline]
    fn ext_real(self) -> Self::AbsOut {
        self.re
    }
    #[inline]
    fn ext_imag(self) -> Self::AbsOut {
        self.im
    }
    #[inline]
    fn ext_conj(self) -> Self {
        self.conj()
    }
    /* #endregion */

    /* #region utilities */
    #[inline]
    fn is_nan(&self) -> bool {
        use num::complex::ComplexFloat;
        ComplexFloat::is_nan(*self)
    }
    /* #endregion */
}
