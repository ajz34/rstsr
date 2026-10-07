use duplicate::duplicate_item;

/// Zero value accessor for all rstsr dtypes (num-`Zero`-like, but also
/// covering `bool`, whose zero is `false`).
pub trait ExtZero: Clone {
    /// The additive identity of this dtype (`0`, `0.0`, `false`, `0+0i`).
    fn ext_zero() -> Self;
}

#[duplicate_item(T; [bool]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtZero for T {
    fn ext_zero() -> Self {
        Self::default()
    }
}

#[duplicate_item(T; [f32]; [f64];)]
impl ExtZero for T {
    fn ext_zero() -> Self {
        0.0
    }
}

impl ExtZero for num::Complex<f32> {
    fn ext_zero() -> Self {
        num::Complex::new(0.0, 0.0)
    }
}

impl ExtZero for num::Complex<f64> {
    fn ext_zero() -> Self {
        num::Complex::new(0.0, 0.0)
    }
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtZero for T {
    fn ext_zero() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ext_zero() {
        assert!(!bool::ext_zero());
        assert_eq!(0, i32::ext_zero());
        assert_eq!(0.0, f64::ext_zero());
        assert_eq!(num::Complex::new(0.0, 0.0), <num::Complex<f64> as ExtZero>::ext_zero());
    }
}
