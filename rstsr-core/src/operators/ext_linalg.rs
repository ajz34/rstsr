//! Extended (array-API promotion-compatible) linear algebra operations.
//!
//! The device traits here are the promotion-compatible forms of the linalg
//! operations: the operands may have different dtypes, each pair promoted to its
//! common dtype inside the kernel. The same-dtype device traits live in
//! [`crate::operators::matmul`] and [`crate::operators::linalg`].

use crate::prelude_dev::*;

/// Matrix multiplication with Array-API dtype promotion.
///
/// The operands may have different dtypes: each pair is promoted to its common
/// dtype ([`DTypePromoteAPI`]) inside the kernel and the product is accumulated
/// in that type. This is the device op behind [`ext_matmul`]; the same-dtype
/// [`DeviceMatMulAPI`] is left unchanged.
pub trait DeviceExtMatMulAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    /// Matrix multiplication into an **uninitialized** output: writes
    /// `c = alpha * (a @ b)` and never reads `c`.
    ///
    /// The same contract as [`DeviceMatMulAPI::matmul_uninit`], except that `a`
    /// and `b` may have different dtypes; `TC` must be their promoted common
    /// type. Implementations must initialize every element of `lc`.
    fn ext_matmul_uninit(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        alpha: TC,
    ) -> Result<()>
    where
        Self: DeviceRawAPI<MaybeUninit<TC>>;
}
