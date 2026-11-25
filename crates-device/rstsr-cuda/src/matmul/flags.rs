//! The driver enable set of the matmul chain.

use core::fmt;
use std::ops::{BitOr, BitOrAssign, Sub, SubAssign};

use super::drivers::GemmDriver;

/// The matmul drivers a device may use ([`CudaRuntimeFlags::matmul_drivers`]).
///
/// [`CudaRuntimeFlags::matmul_drivers`]: crate::flags::CudaRuntimeFlags::matmul_drivers
///
/// Selection is the static candidate table of the type triple, filtered by this set. Forcing one
/// driver is enabling only its bit; an empty set, or a set none of whose members can serve the
/// triple, is an error naming the drivers and the triple.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct MatMulDrivers(u8);

impl Default for MatMulDrivers {
    /// Every driver.
    fn default() -> Self {
        Self::all()
    }
}

impl MatMulDrivers {
    pub const CUBLAS_LT: Self = GemmDriver::CublasLt.flag();
    pub const CUBLAS_EX: Self = GemmDriver::CublasEx.flag();
    pub const CUBLAS: Self = GemmDriver::Cublas.flag();
    pub const TILED: Self = GemmDriver::Tiled.flag();
    pub const NAIVE: Self = GemmDriver::Naive.flag();

    /// The set of one raw bit pattern ([`GemmDriver::flag`] builds the named constants).
    pub(crate) const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Every driver.
    pub const fn all() -> Self {
        Self(0b11111)
    }

    /// No driver.
    pub const fn none() -> Self {
        Self(0)
    }

    /// Whether every driver of `other` is enabled here.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Every driver of `self` that is not in `other` (`all().minus(X)` disables `X`).
    pub const fn minus(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Whether no driver is enabled.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for MatMulDrivers {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl BitOrAssign for MatMulDrivers {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl Sub for MatMulDrivers {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.minus(rhs)
    }
}

impl SubAssign for MatMulDrivers {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 &= !rhs.0;
    }
}

impl fmt::Debug for MatMulDrivers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = GemmDriver::ALL.iter().filter(|d| self.contains(d.flag())).map(|d| d.name()).collect();
        if names.is_empty() {
            return write!(f, "MatMulDrivers(none)");
        }
        write!(f, "MatMulDrivers({})", names.join("|"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five drivers own one bit each, and `all` is their union.
    #[test]
    fn one_bit_per_driver() {
        let mut bits = Vec::new();
        for d in GemmDriver::ALL {
            assert_eq!(d.flag().0.count_ones(), 1, "{} must own exactly one bit", d.name());
            assert!(!bits.contains(&d.flag().0), "{} shares a bit", d.name());
            bits.push(d.flag().0);
        }
        let mut union = MatMulDrivers::none();
        for d in GemmDriver::ALL {
            union |= d.flag();
        }
        assert_eq!(union, MatMulDrivers::all());
    }

    /// `contains` is a subset test, so a forced driver is contained in its own set only.
    #[test]
    fn contains_is_a_subset_test() {
        let tiled = MatMulDrivers::TILED;
        assert!(MatMulDrivers::all().contains(tiled));
        assert!(tiled.contains(tiled));
        assert!(!tiled.contains(MatMulDrivers::NAIVE));
        let without = MatMulDrivers::all() - MatMulDrivers::CUBLAS;
        assert!(!without.contains(MatMulDrivers::CUBLAS));
        assert!(without.contains(MatMulDrivers::TILED));
    }

    /// The default enables every driver; a forced set holds one name.
    #[test]
    fn default_and_display() {
        assert_eq!(MatMulDrivers::default(), MatMulDrivers::all());
        assert_eq!(format!("{:?}", MatMulDrivers::TILED | MatMulDrivers::NAIVE), "MatMulDrivers(tiled|naive)");
        assert_eq!(format!("{:?}", MatMulDrivers::default()), "MatMulDrivers(cublasLt|cublasEx|cublas|tiled|naive)");
        assert_eq!(format!("{:?}", MatMulDrivers::none()), "MatMulDrivers(none)");
    }
}
