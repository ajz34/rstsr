//! Rem, bitwise, shift, and compound-assign device-op impls (B1–B3).
//!
//! Split out of `operators.rs` for file size; same impl conventions.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaArith, CudaDType, CudaInt};
use crate::kernels::elementwise;
use crate::prelude_dev::*;

// /* #region rem + bitwise + shifts (B1/B2) */
/// `rem`: int `%` with zero-divisor guard, NumPy float remainder
/// (sign-of-divisor), Gaussian complex remainder — `exprs::rem`
/// carries the dispatch. Bounds follow the CPU's `Rem` table: all
/// `CudaArith` dtypes (complex included; num-complex has `Rem`).
impl<TA, TB, TC, D> OpRemAPI<TA, TB, TC, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: CudaDType + CudaArith,
    TA: CudaDType + CudaArith,
    TB: CudaDType + CudaArith,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::rem(TA::CNAME, TB::CNAME, TC::CNAME, "a[ia]", "b[ib]");
        elementwise::binary_launch::<TC, TA, TB>(self, c, lc, a, la, b, lb, &expr)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::rem(TA::CNAME, TB::CNAME, TC::CNAME, "t[it]", "v");
        elementwise::scalar_launch::<TC, TA, TB>(self, c, lc, a, la, &b, &expr)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::rem(TA::CNAME, TB::CNAME, TC::CNAME, "v", "t[it]");
        elementwise::scalar_launch::<TC, TB, TA>(self, c, lc, b, lb, &a, &expr)
    }
}

/// Bitwise or/and/xor on `CudaInt` (bool bitwise is served through the
/// same C operators; C promotes bool to int and demotes the store —
/// the generated `(a[ia] | b[ib])` expression re-casts to bool on
/// store, matching Rust's bool BitOr).
macro_rules! impl_bitwise_cuda {
    ($OpAPI:ident, $opchar:expr) => {
        impl<TA, TB, TC, D> $OpAPI<TA, TB, TC, D> for DeviceCuda
        where
            Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
                + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
                + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
            TC: CudaDType + CudaInt,
            TA: CudaDType + CudaInt,
            TB: CudaDType + CudaInt,
            D: DimAPI,
        {
            fn op_mutc_refa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<TC>>,
                lc: &Layout<D>,
                a: &CudaBuffer<TA>,
                la: &Layout<D>,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = format!("(a[ia] {} b[ib])", $opchar);
                elementwise::binary_launch::<TC, TA, TB>(self, c, lc, a, la, b, lb, &expr)
            }

            fn op_mutc_refa_numb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<TC>>,
                lc: &Layout<D>,
                a: &CudaBuffer<TA>,
                la: &Layout<D>,
                b: TB,
            ) -> Result<()> {
                let expr = format!("(t[it] {} v)", $opchar);
                elementwise::scalar_launch::<TC, TA, TB>(self, c, lc, a, la, &b, &expr)
            }

            fn op_mutc_numa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<TC>>,
                lc: &Layout<D>,
                a: TA,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = format!("(v {} t[it])", $opchar);
                elementwise::scalar_launch::<TC, TB, TA>(self, c, lc, b, lb, &a, &expr)
            }
        }
    };
}

impl_bitwise_cuda!(OpBitOrAPI, "|");
impl_bitwise_cuda!(OpBitAndAPI, "&");
impl_bitwise_cuda!(OpBitXorAPI, "^");

/// Shifts: mask the shift count to the width (NumPy masking; Rust
/// release-mode x86 semantics) — documented divergence from Rust's
/// overflow panic in debug builds.
macro_rules! impl_shift_cuda {
    ($OpAPI:ident, $opchar:expr) => {
        impl<TA, TB, TC, D> $OpAPI<TA, TB, TC, D> for DeviceCuda
        where
            Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
                + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
                + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
            TC: CudaDType + CudaInt,
            TA: CudaDType + CudaInt,
            TB: CudaDType + CudaInt,
            D: DimAPI,
        {
            fn op_mutc_refa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<TC>>,
                lc: &Layout<D>,
                a: &CudaBuffer<TA>,
                la: &Layout<D>,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let bits = core::mem::size_of::<TC>() * 8;
                let expr = format!("(a[ia] {} (b[ib] & {}))", $opchar, bits - 1);
                elementwise::binary_launch::<TC, TA, TB>(self, c, lc, a, la, b, lb, &expr)
            }

            fn op_mutc_refa_numb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<TC>>,
                lc: &Layout<D>,
                a: &CudaBuffer<TA>,
                la: &Layout<D>,
                b: TB,
            ) -> Result<()> {
                let bits = core::mem::size_of::<TC>() * 8;
                let expr = format!("(t[it] {} (v & {}))", $opchar, bits - 1);
                elementwise::scalar_launch::<TC, TA, TB>(self, c, lc, a, la, &b, &expr)
            }

            fn op_mutc_numa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<TC>>,
                lc: &Layout<D>,
                a: TA,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let bits = core::mem::size_of::<TC>() * 8;
                let expr = format!("(v {} (t[it] & {}))", $opchar, bits - 1);
                elementwise::scalar_launch::<TC, TB, TA>(self, c, lc, b, lb, &a, &expr)
            }
        }
    };
}

impl_shift_cuda!(OpShlAPI, "<<");
impl_shift_cuda!(OpShrAPI, ">>");

// /* #endregion */
// /* #region compound assigns (B3) */
/// Arithmetic compound assigns (float-capable: add/sub/mul/div).
macro_rules! impl_assign_arith_cuda {
    ($OpAssignAPI:ident, $opchar:expr) => {
        impl<TA, TB, D> $OpAssignAPI<TA, TB, D> for DeviceCuda
        where
            Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>> + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>,
            TA: CudaDType + CudaArith,
            TB: CudaDType + CudaArith,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                a: &mut CudaBuffer<TA>,
                la: &Layout<D>,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = format!("(t[it] {} b[ib])", $opchar);
                elementwise::binary_assign_launch::<TA, TB>(self, a, la, b, lb, &expr)
            }

            fn op_muta_numb(&self, a: &mut CudaBuffer<TA>, la: &Layout<D>, b: TB) -> Result<()> {
                let expr = format!("(t[it] {} v)", $opchar);
                elementwise::scalar_assign_launch::<TA, TB>(self, a, la, &b, &expr)
            }
        }
    };
}

impl_assign_arith_cuda!(OpAddAssignAPI, '+');
impl_assign_arith_cuda!(OpSubAssignAPI, '-');
impl_assign_arith_cuda!(OpMulAssignAPI, '*');
impl_assign_arith_cuda!(OpDivAssignAPI, '/');

/// Bitwise compound assigns (int-only on CPU).
macro_rules! impl_assign_int_cuda {
    ($OpAssignAPI:ident, $opchar:expr) => {
        impl<TA, TB, D> $OpAssignAPI<TA, TB, D> for DeviceCuda
        where
            Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>> + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>,
            TA: CudaDType + CudaInt,
            TB: CudaDType + CudaInt,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                a: &mut CudaBuffer<TA>,
                la: &Layout<D>,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = format!("(t[it] {} b[ib])", $opchar);
                elementwise::binary_assign_launch::<TA, TB>(self, a, la, b, lb, &expr)
            }

            fn op_muta_numb(&self, a: &mut CudaBuffer<TA>, la: &Layout<D>, b: TB) -> Result<()> {
                let expr = format!("(t[it] {} v)", $opchar);
                elementwise::scalar_assign_launch::<TA, TB>(self, a, la, &b, &expr)
            }
        }
    };
}

impl_assign_int_cuda!(OpBitOrAssignAPI, '|');
impl_assign_int_cuda!(OpBitAndAssignAPI, '&');
impl_assign_int_cuda!(OpBitXorAssignAPI, '^');

/// `rem_assign`: float-capable (the expression builder dispatches int
/// `%` vs NumPy float remainder vs Gaussian complex remainder).
impl<TA, TB, D> OpRemAssignAPI<TA, TB, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>> + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>,
    TA: CudaDType + CudaArith,
    TB: CudaDType + CudaArith,
    D: DimAPI,
{
    fn op_muta_refb(&self, a: &mut CudaBuffer<TA>, la: &Layout<D>, b: &CudaBuffer<TB>, lb: &Layout<D>) -> Result<()> {
        let expr = crate::kernels::exprs::rem(TA::CNAME, TB::CNAME, TA::CNAME, "t[it]", "b[ib]");
        elementwise::binary_assign_launch::<TA, TB>(self, a, la, b, lb, &expr)
    }

    fn op_muta_numb(&self, a: &mut CudaBuffer<TA>, la: &Layout<D>, b: TB) -> Result<()> {
        let expr = crate::kernels::exprs::rem(TA::CNAME, TB::CNAME, TA::CNAME, "t[it]", "v");
        elementwise::scalar_assign_launch::<TA, TB>(self, a, la, &b, &expr)
    }
}

/// Shift assigns with the width-masked count.
macro_rules! impl_assign_shift_cuda {
    ($OpAssignAPI:ident, $opchar:expr) => {
        impl<TA, TB, D> $OpAssignAPI<TA, TB, D> for DeviceCuda
        where
            Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>> + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>,
            TA: CudaDType + CudaInt,
            TB: CudaDType + CudaInt,
            D: DimAPI,
        {
            fn op_muta_refb(
                &self,
                a: &mut CudaBuffer<TA>,
                la: &Layout<D>,
                b: &CudaBuffer<TB>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let bits = core::mem::size_of::<TA>() * 8;
                let expr = format!("(t[it] {} (b[ib] & {}))", $opchar, bits - 1);
                elementwise::binary_assign_launch::<TA, TB>(self, a, la, b, lb, &expr)
            }

            fn op_muta_numb(&self, a: &mut CudaBuffer<TA>, la: &Layout<D>, b: TB) -> Result<()> {
                let bits = core::mem::size_of::<TA>() * 8;
                let expr = format!("(t[it] {} (v & {}))", $opchar, bits - 1);
                elementwise::scalar_assign_launch::<TA, TB>(self, a, la, &b, &expr)
            }
        }
    };
}

impl_assign_shift_cuda!(OpShlAssignAPI, "<<");
impl_assign_shift_cuda!(OpShrAssignAPI, ">>");

// /* #endregion */
