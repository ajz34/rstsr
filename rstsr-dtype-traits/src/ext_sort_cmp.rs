use core::cmp::Ordering;

use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;

/// Total-order comparison for sortable dtypes: integers, booleans, real
/// floats, and complex floats.
///
/// Unlike [`ExtReal::ext_min`](crate::ExtReal::ext_min) (IEEE NaN-skipping
/// semantics), this trait provides a consistent total order usable by sorting:
///
/// - values equal under `==` (including `-0.0` and `0.0`) compare [`Ordering::Equal`];
/// - NaN (or a NaN component, for complex) is ordered greater than everything.
pub trait ExtSortCmp: Clone {
    /// Total-order comparison of `self` against `other`.
    fn ext_total_cmp(&self, other: &Self) -> Ordering;
}

#[duplicate_item(T; [bool]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtSortCmp for T {
    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self, other)
    }
}

#[duplicate_item(T; [f32]; [f64];)]
impl ExtSortCmp for T {
    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            // NaN ordered greater than everything; NaN == NaN for sorting
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // `partial_cmp` already treats -0.0 == 0.0 as Equal
            (false, false) => self.partial_cmp(other).unwrap_or(Ordering::Equal),
        }
    }
}

#[duplicate_item(T; [Complex<f32>]; [Complex<f64>];)]
impl ExtSortCmp for T {
    /// Lexicographic order (real part first, then imaginary); a NaN in either
    /// component orders the value after every non-NaN value.
    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        let self_nan = self.is_nan();
        let other_nan = other.is_nan();
        match (self_nan, other_nan) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => match self.re.partial_cmp(&other.re) {
                Some(Ordering::Equal) => self.im.partial_cmp(&other.im).unwrap_or(Ordering::Equal),
                other => other.unwrap_or(Ordering::Equal),
            },
        }
    }
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtSortCmp for T {
    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self.partial_cmp(other).unwrap_or(Ordering::Equal),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_complex_lexicographic() {
        let a = Complex::new(1.0f64, 2.0);
        let b = Complex::new(1.0, 3.0);
        assert_eq!(a.ext_total_cmp(&b), Ordering::Less);
        let nan = Complex::new(0.0f64, f64::NAN);
        assert_eq!(nan.ext_total_cmp(&Complex::new(f64::INFINITY, 0.0)), Ordering::Greater);
        assert_eq!(nan.ext_total_cmp(&nan), Ordering::Equal);
    }
}
