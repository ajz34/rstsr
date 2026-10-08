use core::cmp::Ordering;

use duplicate::duplicate_item;
use num::Complex;

/// Total-order comparison for sortable dtypes: integers, booleans, real
/// floats, and complex floats.
///
/// Unlike [`ExtReal::ext_min`](crate::ExtReal::ext_min) (which propagates
/// NaN), this trait provides a consistent total order usable by sorting,
/// following NumPy's ordering semantics:
///
/// - values equal under `==` (including `-0.0` and `0.0`) compare [`Ordering::Equal`];
/// - NaN is ordered greater than everything, including in descending sorts;
/// - complex: every finite value orders before every NaN-bearing value; among NaN-bearing values
///   the order is part-wise lexicographic with the NaN part after a finite part (NumPy
///   `arraytypes.c.src` `C@TYPE@_compare`).
pub trait ExtSortCmp: Clone {
    /// Total-order comparison of `self` against `other`.
    fn ext_total_cmp(&self, other: &Self) -> Ordering;

    /// Whether this value is NaN (or NaN-bearing, for complex); such values
    /// order after all others regardless of sort direction. `false` for all
    /// non-float dtypes.
    fn ext_is_nan(&self) -> bool {
        false
    }
}

#[duplicate_item(T; [bool]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtSortCmp for T {
    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(self, other)
    }
}

#[duplicate_item(T; [f32]; [f64];)]
impl ExtSortCmp for T {
    fn ext_is_nan(&self) -> bool {
        self.is_nan()
    }

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
    fn ext_is_nan(&self) -> bool {
        self.is_nan()
    }

    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        // NumPy `C@TYPE@_compare` (numpy/_core/src/multiarray/arraytypes.c.src):
        // every finite value orders before every NaN-bearing value; among
        // NaN-bearing values the order is part-wise lexicographic with the
        // NaN part ordering after a finite part.
        let (ar, ai, br, bi) = (self.re, self.im, other.re, other.im);
        if ar < br {
            return if ai.is_nan() && !bi.is_nan() { Ordering::Greater } else { Ordering::Less };
        }
        if br < ar {
            return if bi.is_nan() && !ai.is_nan() { Ordering::Less } else { Ordering::Greater };
        }
        if ar == br || (ar.is_nan() && br.is_nan()) {
            if ai < bi || (bi.is_nan() && !ai.is_nan()) {
                return Ordering::Less;
            }
            if bi < ai || (ai.is_nan() && !bi.is_nan()) {
                return Ordering::Greater;
            }
            return Ordering::Equal;
        }
        // real parts incomparable with exactly one NaN: that value is greater
        if !ar.is_nan() {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtSortCmp for T {
    fn ext_is_nan(&self) -> bool {
        self.is_nan()
    }

    fn ext_total_cmp(&self, other: &Self) -> Ordering {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self.partial_cmp(other).unwrap_or(Ordering::Equal),
        }
    }
}
