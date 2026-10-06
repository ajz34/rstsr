use core::cmp::Ordering;

use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;

/// Total-order comparison for sortable dtypes: integers, booleans, real
/// floats, and complex floats.
///
/// Unlike [`ExtReal::ext_min`](crate::ExtReal::ext_min) (IEEE NaN-skipping
/// semantics), this trait provides a consistent total order usable by sorting,
/// following NumPy's ordering semantics:
///
/// - values equal under `==` (including `-0.0` and `0.0`) compare [`Ordering::Equal`];
/// - NaN (or a NaN component, for complex) is ordered greater than everything, including in
///   descending sorts;
/// - complex NaN-bearing values order by their finite parts before the NaN group (NumPy
///   `numpy_tag.h` complex comparator).
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
    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        // NumPy orders NaN-bearing complexes lexicographically by their
        // finite parts (NaN component sorted last within each part), before
        // every non-NaN value; see NumPy `numpy_tag.h` complex comparator.
        fn part_cmp<F: num::Float>(a: &F, b: &F) -> Ordering {
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            }
        }
        let (a_re, a_im) = (self.re, self.im);
        let (b_re, b_im) = (other.re, other.im);
        match part_cmp(&a_re, &b_re) {
            Ordering::Equal => part_cmp(&a_im, &b_im),
            ord => ord,
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
    fn test_complex_nan_by_finite_parts() {
        // NumPy ordering: NaN-bearing complexes compare by finite parts,
        // NaN part last; all NaN-bearing values come after finite ones
        let finite = Complex::new(f64::INFINITY, 0.0);
        let nan_re = Complex::new(f64::NAN, 0.0);
        let nan_im = Complex::new(0.0f64, f64::NAN);
        let nan_both = Complex::new(f64::NAN, f64::NAN);
        // finite parts decide among NaN-bearing values
        assert_eq!(nan_im.ext_total_cmp(&nan_re), Ordering::Less); // (0, nan) < (nan, 0)
        assert_eq!(nan_re.ext_total_cmp(&nan_both), Ordering::Less); // (nan, 0) < (nan, nan)
        assert_eq!(nan_im.ext_total_cmp(&nan_both), Ordering::Less);
        // any NaN-bearing value sorts after every finite one
        assert_eq!(finite.ext_total_cmp(&nan_re), Ordering::Less);
        assert_eq!(nan_both.ext_total_cmp(&finite), Ordering::Greater);
    }

    #[test]
    fn test_complex_nan_equal_parts() {
        // identical finite parts with NaN in the same slot compare Equal
        let a = Complex::new(1.0f64, f64::NAN);
        let b = Complex::new(1.0, f64::NAN);
        assert_eq!(a.ext_total_cmp(&b), Ordering::Equal);
    }
}
