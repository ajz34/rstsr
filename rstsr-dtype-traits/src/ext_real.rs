use duplicate::duplicate_item;
use num::{Float, Integer};

/// Extension trait for real types (floats and integers included).
pub trait ExtReal: Clone {
    /// Computes the floor division of two numbers.
    fn ext_floor_divide(self, other: Self) -> Self;

    /// Rounds toward negative infinity to an integral value — the array-API `floor`.
    ///
    /// Identity for integer types (already integral).
    fn ext_floor(self) -> Self;

    /// Rounds toward positive infinity to an integral value — the array-API `ceil`.
    ///
    /// Identity for integer types.
    fn ext_ceil(self) -> Self;

    /// Rounds toward zero to an integral value — the array-API `trunc`.
    ///
    /// Identity for integer types.
    fn ext_trunc(self) -> Self;

    /// Returns the minimum of two numbers.
    ///
    /// # Note
    ///
    /// For floats, NaN propagates (IEEE 754-2019 `minimum`, as the array-API requires): if either
    /// operand is NaN, the result is NaN.
    fn ext_min(self, other: Self) -> Self;

    /// The minimum value that can be represented by this type.
    fn ext_min_value() -> Self;

    /// Returns the maximum of two numbers.
    ///
    /// # Note
    ///
    /// For floats, NaN propagates (IEEE 754-2019 `maximum`, as the array-API requires): if either
    /// operand is NaN, the result is NaN.
    fn ext_max(self, other: Self) -> Self;

    /// The maximum value that can be represented by this type.
    fn ext_max_value() -> Self;

    /// Determines whether the sign bit is set — the array-API `signbit`.
    ///
    /// Follows IEEE 754: true for negative values, `-0.0` and negatively
    /// signed NaN; always false for unsigned integers.
    fn ext_signbit(self) -> bool;
}

#[duplicate_item(T; [u8]; [u16]; [u32]; [u64]; [u128]; [usize];)]
impl ExtReal for T {
    fn ext_signbit(self) -> bool {
        false
    }
    fn ext_floor_divide(self, other: Self) -> Self {
        Integer::div_floor(&self, &other)
    }
    fn ext_floor(self) -> Self {
        self
    }
    fn ext_ceil(self) -> Self {
        self
    }
    fn ext_trunc(self) -> Self {
        self
    }
    fn ext_min(self, other: Self) -> Self {
        Ord::min(self, other)
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        Ord::max(self, other)
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}

#[duplicate_item(T; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtReal for T {
    fn ext_signbit(self) -> bool {
        self < 0
    }
    fn ext_floor_divide(self, other: Self) -> Self {
        Integer::div_floor(&self, &other)
    }
    fn ext_floor(self) -> Self {
        self
    }
    fn ext_ceil(self) -> Self {
        self
    }
    fn ext_trunc(self) -> Self {
        self
    }
    fn ext_min(self, other: Self) -> Self {
        Ord::min(self, other)
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        Ord::max(self, other)
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}

#[duplicate_item(T; [f32]; [f64];)]
impl ExtReal for T {
    fn ext_signbit(self) -> bool {
        // sign bit, not `< 0`: true for -0.0 and negatively-signed NaN
        Float::is_sign_negative(self)
    }
    fn ext_floor_divide(self, other: Self) -> Self {
        Float::floor(self / other)
    }
    fn ext_floor(self) -> Self {
        Float::floor(self)
    }
    fn ext_ceil(self) -> Self {
        Float::ceil(self)
    }
    fn ext_trunc(self) -> Self {
        Float::trunc(self)
    }
    fn ext_min(self, other: Self) -> Self {
        if self.is_nan() {
            self
        } else if other.is_nan() {
            other
        } else {
            T::min(self, other)
        }
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        if self.is_nan() {
            self
        } else if other.is_nan() {
            other
        } else {
            T::max(self, other)
        }
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtReal for T {
    fn ext_signbit(self) -> bool {
        Float::is_sign_negative(self)
    }
    fn ext_floor_divide(self, other: Self) -> Self {
        Float::floor(self / other)
    }
    fn ext_floor(self) -> Self {
        Float::floor(self)
    }
    fn ext_ceil(self) -> Self {
        Float::ceil(self)
    }
    fn ext_trunc(self) -> Self {
        Float::trunc(self)
    }
    fn ext_min(self, other: Self) -> Self {
        if self.is_nan() {
            self
        } else if other.is_nan() {
            other
        } else {
            T::min(self, other)
        }
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        if self.is_nan() {
            self
        } else if other.is_nan() {
            other
        } else {
            T::max(self, other)
        }
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}
