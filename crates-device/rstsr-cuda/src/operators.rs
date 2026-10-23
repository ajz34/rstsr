//! Elementwise device-op trait impls for `DeviceCuda`: arithmetic,
//! comparison, and unary float families. Element bounds live on these
//! device impls (`CudaDType` sub-traits), not on the tensor tier.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaArith, CudaCmp, CudaDType, CudaFloat};
use crate::kernels_ops;
use crate::prelude_dev::*;

/// CUDA C call expression `name<suffix>(arg)`.
fn concat_expr(name: &str, suffix: &str, arg: &str) -> String {
    format!("{name}{suffix}({arg})")
}

// /* #region binary arithmetic */

#[duplicate_item(
    OpAPI   opchar;
    [OpAddAPI] ['+'];
    [OpSubAPI] ['-'];
    [OpMulAPI] ['*'];
    [OpDivAPI] ['/'];
)]
impl<TA, TB, TC, D> OpAPI<TA, TB, TC, D> for DeviceCuda
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
        kernels_ops::binary_arith::<TC, TA, TB>(self, c, lc, a, la, b, lb, opchar)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        kernels_ops::binary_arith_scalar::<TC, TA, TB>(self, c, lc, a, la, &b, opchar, false)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        kernels_ops::binary_arith_scalar::<TC, TB, TA>(self, c, lc, b, lb, &a, opchar, true)
    }
}

// /* #endregion */

// /* #region comparisons */

#[duplicate_item(
    OpAPI        opstr;
    [OpEqualAPI        ] ["=="];
    [OpNotEqualAPI     ] ["!="];
    [OpGreaterAPI      ] [">" ];
    [OpGreaterEqualAPI ] [">="];
    [OpLessAPI         ] ["<" ];
    [OpLessEqualAPI    ] ["<="];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
    TA: CudaDType + CudaCmp,
    TB: CudaDType + CudaCmp,
    D: DimAPI,
{
    type TOut = bool;

    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        kernels_ops::binary_cmp::<TA, TB>(self, c, lc, a, la, b, lb, opstr)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        kernels_ops::binary_cmp_scalar::<TA, TB>(self, c, lc, a, la, &b, opstr, false)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        kernels_ops::binary_cmp_scalar::<TB, TA>(self, c, lc, b, lb, &a, opstr, true)
    }
}

// /* #endregion */

// /* #region unary float functions */

/// `TOut = T` float unaries; CUDA C function names take an `f`-suffixed
/// form for `float` — the generator picks via the `{F}()` marker.
#[duplicate_item(
    OpAPI       fn_base  is_recip;
    [OpSinAPI     ] ["sin"]   [false];
    [OpCosAPI] ["cos"]         [false];
    [OpTanAPI] ["tan"]         [false];
    [OpAsinAPI] ["asin"]        [false];
    [OpAcosAPI] ["acos"]        [false];
    [OpAtanAPI] ["atan"]        [false];
    [OpSinhAPI] ["sinh"]        [false];
    [OpCoshAPI] ["cosh"]        [false];
    [OpTanhAPI] ["tanh"]        [false];
    [OpAsinhAPI] ["asinh"]       [false];
    [OpAcoshAPI] ["acosh"]       [false];
    [OpAtanhAPI] ["atanh"]       [false];
    [OpExpAPI] ["exp"]         [false];
    [OpExpm1API   ] ["expm1"]  [false];
    [OpLogAPI] ["log"]         [false];
    [OpLog1pAPI   ] ["log1p"]  [false];
    [OpLog2API    ] ["log2"]  [false];
    [OpLog10API   ] ["log10"]  [false];
    [OpSqrtAPI] ["sqrt"]        [false];
    [OpInvAPI     ] ["inv"]  [true];
    [OpReciprocalAPI] ["recip"] [true];
)]
impl<T, D> OpAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaFloat,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        // {F} resolves to the C float-suffix form per dtype at generation
        let cname = T::CNAME;
        let call = if cname == "float" {
            concat_expr(fn_base, "f", "b[ib]")
        } else if cname == "double" {
            concat_expr(fn_base, "", "b[ib]")
        } else {
            rstsr_raise!(UnImplemented, "unary float kernel dtype {cname} not covered")?
        };
        let expr: &str = if is_recip { "(1.0 / b[ib])" } else { &call };
        kernels_ops::unary_float::<T>(self, a, la, b, lb, expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        // each thread reads and writes its own element, so the same buffer
        // is both operand and destination
        let cname = T::CNAME;
        let call = if cname == "float" {
            concat_expr(fn_base, "f", "a[io]")
        } else if cname == "double" {
            concat_expr(fn_base, "", "a[io]")
        } else {
            rstsr_raise!(UnImplemented, "unary float kernel dtype {cname} not covered")?
        };
        let expr: &str = if is_recip { "(1.0 / a[io])" } else { &call };
        // SAFETY: MaybeUninit<T> and T share layout; the kernel reads and
        // writes every visited element
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        kernels_ops::unary_float_inplace::<T>(self, a_typed, la, expr)
    }
}

// /* #endregion */
