//! Elementwise device-op trait impls for `DeviceCuda`: arithmetic,
//! comparison, and unary float families. Element bounds live on these
//! device impls (`CudaDType` sub-traits), not on the tensor tier.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaArith, CudaCmp, CudaComplexFrom, CudaDType, CudaFloat, CudaReal};
use crate::kernels::elementwise;
use crate::prelude_dev::*;

/// CUDA C call expression `name<suffix>(arg)`.
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
        elementwise::binary_launch::<TC, TA, TB>(self, c, lc, a, la, b, lb, &format!("(a[ia] {} b[ib])", opchar))
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        elementwise::scalar_launch::<TC, TA, TB>(self, c, lc, a, la, &b, &format!("(t[it] {} v)", opchar))
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::scalar_launch::<TC, TB, TA>(self, c, lc, b, lb, &a, &format!("(v {} t[it])", opchar))
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
        elementwise::binary_launch::<bool, TA, TB>(self, c, lc, a, la, b, lb, &format!("(a[ia] {} b[ib])", opstr))
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        elementwise::scalar_launch::<bool, TA, TB>(self, c, lc, a, la, &b, &format!("(t[it] {} v)", opstr))
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::scalar_launch::<bool, TB, TA>(self, c, lc, b, lb, &a, &format!("(v {} t[it])", opstr))
    }
}

// /* #endregion */
// /* #region transcendentals: real (libdevice) + complex (own header) */
/// Real transcendentals use CUDA libdevice (`sinf`/`sin`, ...); complex
/// ones use the rstsr_complex.cuh functions (Smith/Hull-Tenan formulas
/// as in CuPy complex.cuh and pytorch ComplexMath.cuh; see the header).
/// Bound is `CudaArith` (f32/f64/complex); other dtypes are rejected at
/// source-gen (same omit behavior as before).
#[duplicate_item(
    OpAPI        fn_base  is_recip  cfn        compose;
    [OpSinAPI     ] ["sin"]   [false] ["csin"]   [""];
    [OpCosAPI] ["cos"]         [false] ["ccos"]   [""];
    [OpTanAPI] ["tan"]         [false] ["ctan"]   [""];
    [OpAsinAPI] ["asin"]        [false] ["casin"]  [""];
    [OpAcosAPI] ["acos"]        [false] ["cacos"]  [""];
    [OpAtanAPI] ["atan"]        [false] ["catan"]  [""];
    [OpSinhAPI] ["sinh"]        [false] ["csinh"]  [""];
    [OpCoshAPI] ["cosh"]        [false] ["ccosh"]  [""];
    [OpTanhAPI] ["tanh"]        [false] ["ctanh"]  [""];
    [OpAsinhAPI] ["asinh"]       [false] ["casinh"] [""];
    [OpAcoshAPI] ["acosh"]       [false] ["cacosh"] [""];
    [OpAtanhAPI] ["atanh"]       [false] ["catanh"] [""];
    [OpExpAPI] ["exp"]         [false] ["cexp"]   [""];
    [OpExpm1API   ] ["expm1"]  [false] ["cexp"]   ["sub_one"];
    [OpLogAPI] ["log"]         [false] ["clog"]   [""];
    [OpLog1pAPI   ] ["log1p"]  [false] ["clog"]   ["add_one"];
    [OpLog2API    ] ["log2"]   [false] ["clog"]   ["div_ln2"];
    [OpLog10API   ] ["log10"]  [false] ["clog"]   ["div_ln10"];
    [OpSqrtAPI] ["sqrt"]        [false] ["csqrt"]  [""];
    [OpInvAPI     ] ["inv"]    [true]  ["" ]      [""];
    [OpReciprocalAPI] ["recip"] [true] ["" ]      [""];
)]
impl<T, D> OpAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaArith,
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
        let expr = transc_expr(T::CNAME, fn_base, cfn, compose, is_recip, "b[ib]")?;
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = transc_expr(T::CNAME, fn_base, cfn, compose, is_recip, "a[io]")?;
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// The real component C type of a (possibly complex) C type name.
fn t_real(t: &str) -> &'static str {
    if t == "rstsr_c32" {
        "float"
    } else {
        "double"
    }
}

/// Element expression of one transcendental at C type `t`, operand `x`.
/// Real dtypes call libdevice (f-suffixed for float); complex calls the
/// header function, composing expm1/log1p/log2/log10 from cexp/clog
/// (CPU composes them the same way through num-complex).
fn transc_expr(t: &str, fn_base: &str, cfn: &str, compose: &str, is_recip: bool, x: &str) -> Result<String> {
    let s = crate::kernels::exprs::f(t);
    let complex = crate::dtype::is_complex_cname(t);
    // complex call names are template functions: one spelling for both
    // widths (overload resolution picks float/double real-math inside)
    if is_recip {
        return Ok(if complex {
            format!("rstsr_cdiv(rstsr_make_complex(({})1.0, ({})0.0), {x})", t_real(t), t_real(t))
        } else {
            format!("(1.0{s} / {x})")
        });
    }
    if compose.is_empty() {
        return Ok(if complex { format!("rstsr_{cfn}({x})") } else { format!("{fn_base}{s}({x})") });
    }
    if !complex {
        // real composed forms via libdevice
        return Ok(match compose {
            "sub_one" => format!("(exp{s}({x}) - 1.0{s})"),
            "add_one" => format!("log{s}(1.0{s} + {x})"),
            "div_ln2" => format!("(log{s}({x}) / 0.693147180559945309417232121458{s})"),
            "div_ln10" => format!("(log{s}({x}) / 2.302585092994045684017991454684{s})"),
            other => rstsr_raise!(UnImplemented, "compose form {other} unknown")?,
        });
    }
    let r = t_real(t);
    let one = format!("rstsr_make_complex(({})1.0, ({})0.0)", r, r);
    Ok(match compose {
        "sub_one" => format!("rstsr_csub(rstsr_cexp({x}), {one})"),
        "add_one" => format!("rstsr_clog(rstsr_cadd({one}, {x}))"),
        "div_ln2" => {
            let ln2 = format!("rstsr_make_complex(({})0.693147180559945309417232121458, ({})0.0)", r, r);
            format!("rstsr_cdiv(rstsr_clog({x}), {ln2})")
        },
        "div_ln10" => {
            let ln10 = format!("rstsr_make_complex(({})2.302585092994045684017991454684, ({})0.0)", r, r);
            format!("rstsr_cdiv(rstsr_clog({x}), {ln10})")
        },
        other => rstsr_raise!(UnImplemented, "compose form {other} unknown")?,
    })
}

// /* #endregion */
// /* #region promoting binaries (B4) */
/// The promoting-binary traits are `OpAPI<TA, TB, D>` with associated
/// `TOut`: the tensor tier resolves promotion (`DTypePromoteAPI`) and
/// instantiates the device op at the promoted dtype, so the device sees
/// TA = TB = TC = TOut for same-dtype-operand calls (mixed int/float
/// operands arrive pre-promoted). Float-only families bound `CudaFloat`;
/// real families bound `CudaReal`; pow bounds `CudaArith` (complex
/// included). (CPU reference: op_ternary_common.rs promotion tables.)
///
/// `f_suffix` aliases the exprs helper (real-math suffix by C type).
use crate::kernels::exprs::f as f_suffix;

macro_rules! impl_promoting_binary {
    ($OpAPI:ident, $bound:ident, |$a:ident, $b:ident, $t:ident| $expr:expr) => {
        impl<T, D> $OpAPI<T, T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
            T: CudaDType + $bound,
            D: DimAPI,
        {
            type TOut = T;

            fn op_mutc_refa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<T>>,
                lc: &Layout<D>,
                a: &CudaBuffer<T>,
                la: &Layout<D>,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = {
                    let ($a, $b, $t) = ("a[ia]", "b[ib]", T::CNAME);
                    $expr.to_string()
                };
                elementwise::binary_launch::<T, T, T>(self, c, lc, a, la, b, lb, &expr)
            }

            fn op_mutc_refa_numb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<T>>,
                lc: &Layout<D>,
                a: &CudaBuffer<T>,
                la: &Layout<D>,
                b: T,
            ) -> Result<()> {
                let expr = {
                    let ($a, $b, $t) = ("t[it]", "v", T::CNAME);
                    $expr.to_string()
                };
                elementwise::scalar_launch::<T, T, T>(self, c, lc, a, la, &b, &expr)
            }

            fn op_mutc_numa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<T>>,
                lc: &Layout<D>,
                a: T,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = {
                    let ($a, $b, $t) = ("v", "t[it]", T::CNAME);
                    $expr.to_string()
                };
                elementwise::scalar_launch::<T, T, T>(self, c, lc, b, lb, &a, &expr)
            }
        }
    };
}

impl_promoting_binary!(OpATan2API, CudaFloat, |a, b, t| format!("atan2{}({}, {})", f_suffix(t), a, b));
impl_promoting_binary!(OpCopySignAPI, CudaFloat, |a, b, t| format!("copysign{}({}, {})", f_suffix(t), a, b));
impl_promoting_binary!(OpHypotAPI, CudaFloat, |a, b, t| format!("hypot{}({}, {})", f_suffix(t), a, b));
impl_promoting_binary!(OpNextAfterAPI, CudaFloat, |a, b, t| format!("nextafter{}({}, {})", f_suffix(t), a, b));
// CPU: (a.exp() + b.exp()).ln() — CUDA libdevice has no logaddexp; emit
// the same formula (fmax inside for stability is a numerical upgrade we
// do NOT take: CPU parity rules).
impl_promoting_binary!(OpLogAddExpAPI, CudaFloat, |a, b, t| format!(
    "log{s}((exp{s}({a}) + exp{s}({b})))",
    s = f_suffix(t)
));

/// `floor_divide`: float floor(a/b); ints div_floor with zero-guard.
fn floor_divide_expr(tc: &str, x: &str, y: &str) -> String {
    if tc == "float" || tc == "double" {
        let s = f_suffix(tc);
        format!("floor{s}(({x}) / ({y}))")
    } else {
        format!(
            "({y} == 0 ? ({tc})0 : (({tc})(({x}) / ({y}))) - ({tc})((((({x}) % ({y})) != 0) && (((({x}) < 0) != (({y}) < 0)))) ? 1 : 0))"
        )
    }
}

impl_promoting_binary!(OpFloorDivideAPI, CudaReal, |a, b, t| floor_divide_expr(t, a, b));
impl_promoting_binary!(OpMaximumAPI, CudaReal, |a, b, t| crate::kernels::exprs::maximum(t, a, b));
impl_promoting_binary!(OpMinimumAPI, CudaReal, |a, b, t| crate::kernels::exprs::minimum(t, a, b));

// `pow`: float powf / int ipow / complex cpow (exprs::pow dispatches).
impl_promoting_binary!(OpPowAPI, CudaArith, |a, b, t| crate::kernels::exprs::pow(t, t, t, a, b));

// /* #endregion */
// /* #region non-transcendental unaries (B5) */
// Floor/ceil/trunc/round: dtype-preserving (ints are identity on CPU);
// CUDA `floor/ceil/trunc/rint` match exactly. Same-type in/out.
macro_rules! impl_rounding_unary {
    ($OpAPI:ident, $fnbase:expr) => {
        impl<T, D> $OpAPI<T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
            T: CudaDType + CudaReal,
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
                let expr = if crate::dtype::is_half_cname(T::CNAME) {
                    let l = crate::kernels::exprs::load(T::CNAME, "b[ib]");
                    let s = crate::kernels::exprs::store(T::CNAME, &format!("{}f({l})", $fnbase));
                    if T::CNAME == "double" {
                        format!("{}(b[ib])", $fnbase)
                    } else {
                        s
                    }
                } else if T::CNAME == "float" {
                    format!("{}f(b[ib])", $fnbase)
                } else if T::CNAME == "double" {
                    format!("{}(b[ib])", $fnbase)
                } else {
                    // integer: identity (CPU ext_floor/ceil/trunc/round)
                    "b[ib]".to_string()
                };
                elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
            }

            fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
                let expr = if T::CNAME == "float" {
                    format!("{}f(a[io])", $fnbase)
                } else if T::CNAME == "double" {
                    format!("{}(a[io])", $fnbase)
                } else {
                    "a[io]".to_string()
                };
                // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
                let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
                elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
            }
        }
    };
}

impl_rounding_unary!(OpCeilAPI, "ceil");
impl_rounding_unary!(OpFloorAPI, "floor");
impl_rounding_unary!(OpTruncAPI, "trunc");

/// `round`: ties-to-even = CUDA `rint` (CPU roundeven parity).
impl<T, D> OpRoundAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaReal,
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
        let expr = crate::kernels::exprs::round_even(T::CNAME, "b[ib]");
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = crate::kernels::exprs::round_even(T::CNAME, "a[io]");
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `abs`: same-type for reals (unsigned identity), magnitude for
/// complex (TOut = real part type). Complex is split out below.
impl<T, D> OpAbsAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceRawAPI<MaybeUninit<T::AbsOut>, Raw = CudaBuffer<MaybeUninit<T::AbsOut>>>,
    T: CudaDType + crate::dtype::CudaComplexFrom + ExtNum<AbsOut: CudaDType>,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::abs(T::CNAME, T::AbsOut::CNAME, "b[ib]");
        elementwise::unary_launch::<T::AbsOut, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if !<T as ExtNum>::ABS_SAME_TYPE {
            // complex/real-mismatch abs has no in-place form (CPU parity)
            rstsr_raise!(UnImplemented, "abs in-place requires a same-type dtype")?;
        }
        let expr = crate::kernels::exprs::abs(T::CNAME, T::AbsOut::CNAME, "a[io]");
        // SAFETY: T == T::AbsOut here (same-type abs)
        let a_typed: &mut CudaBuffer<T> =
            unsafe { &mut *(a as *mut CudaBuffer<MaybeUninit<T::AbsOut>> as *mut CudaBuffer<T>) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `real`/`imag`: complex -> real part type; real dtypes -> identity
/// (T::AbsOut == T for reals).
impl<T, D> OpRealAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T::AbsOut>, Raw = CudaBuffer<MaybeUninit<T::AbsOut>>>,
    T: CudaDType + ExtNum<AbsOut: CudaDType>,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr =
            if crate::dtype::is_complex_cname(T::CNAME) { "(b[ib]).re".to_string() } else { "b[ib]".to_string() };
        elementwise::unary_launch::<T::AbsOut, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>, _la: &Layout<D>) -> Result<()> {
        // real/imag of a real dtype is the identity (no-op in place)
        Ok(())
    }
}

impl<T, D> OpImagAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T::AbsOut>, Raw = CudaBuffer<MaybeUninit<T::AbsOut>>>,
    T: CudaDType + ExtNum<AbsOut: CudaDType>,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = if crate::dtype::is_complex_cname(T::CNAME) {
            "(b[ib]).im".to_string()
        } else {
            format!("({})0", T::AbsOut::CNAME)
        };
        elementwise::unary_launch::<T::AbsOut, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if !<T as ExtNum>::ABS_SAME_TYPE {
            rstsr_raise!(UnImplemented, "imag in-place requires a same-type dtype")?;
        }
        // imag of a real dtype is 0: fill in place
        let expr = format!("({})0", T::AbsOut::CNAME);
        let a_typed: &mut CudaBuffer<T> =
            unsafe { &mut *(a as *mut CudaBuffer<MaybeUninit<T::AbsOut>> as *mut CudaBuffer<T>) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `conj`: complex -> conjugate; real -> identity (CPU conj on real is
/// identity through ExtNum::ext_conj default).
impl<T, D> OpConjAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaComplexFrom,
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
        let expr = if crate::dtype::is_complex_cname(T::CNAME) {
            "rstsr_cconj(b[ib])".to_string()
        } else {
            "b[ib]".to_string()
        };
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = if crate::dtype::is_complex_cname(T::CNAME) {
            "rstsr_cconj(a[io])".to_string()
        } else {
            "a[io]".to_string()
        };
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `sign`: -1/0/1 (NaN->NaN floats; z/|z| complex).
impl<T, D> OpSignAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + crate::dtype::CudaComplexFrom,
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
        let expr = crate::kernels::exprs::sign(T::CNAME, "b[ib]");
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = crate::kernels::exprs::sign(T::CNAME, "a[io]");
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `square`: x*x, any CudaArith dtype (CPU bound Mul<Output=T>).
impl<T, D> OpSquareAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaArith,
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
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, "(b[ib] * b[ib])")
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, "(a[io] * a[io])")
    }
}

/// `ext_neg`: two's-complement wrap for unsigned, plain negate else.
impl<T, D> OpExtNegAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaArith,
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
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, "(-b[ib])")
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, "(-a[io])")
    }
}

/// `not` (logical_not / bitwise invert): bool -> `!`, ints -> `~`
/// (CPU bound: `!` applies to bool and integers; same-type in/out).
impl<T, D> OpNotAPI<T, T, D> for DeviceCuda
where
    Self: DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>> + DeviceRawAPI<T, Raw = CudaBuffer<T>>,
    T: CudaDType,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = if T::CNAME == "bool" { "(!b[ib])" } else { "(~b[ib])" };
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<T>, la: &Layout<D>) -> Result<()> {
        let expr = if T::CNAME == "bool" { "(!a[io])" } else { "(~a[io])" };
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_mu: &mut CudaBuffer<MaybeUninit<T>> =
            unsafe { &mut *(a as *mut CudaBuffer<T> as *mut CudaBuffer<MaybeUninit<T>>) };
        elementwise::unary_launch::<T, T>(self, a_mu, la, a, la, expr)
    }
}

/// boolean-output queries: is_nan/is_inf/is_finite (floats+complex on
/// CPU); generated per-component for complex (true if any component NaN).
macro_rules! impl_query_unary {
    ($OpAPI:ident, $real:expr, $c32:expr, $c64:expr) => {
        impl<T, D> $OpAPI<T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
                + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
            T: CudaDType,
            D: DimAPI,
        {
            type TOut = bool;

            fn op_muta_refb(
                &self,
                a: &mut CudaBuffer<MaybeUninit<bool>>,
                la: &Layout<D>,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = match T::CNAME {
                    "float" => format!("{}f(b[ib])", $real),
                    "double" => format!("{}(b[ib])", $real),
                    "__half" => format!("{}f(__half2float(b[ib]))", $real),
                    "__nv_bfloat16" => format!("{}f(__nv_bfloat162float(b[ib]))", $real),
                    "rstsr_c32" => $c32.to_string(),
                    "rstsr_c64" => $c64.to_string(),
                    other => rstsr_raise!(UnImplemented, "query unary dtype {other} not covered")?,
                };
                elementwise::unary_launch::<bool, T>(self, a, la, b, lb, &expr)
            }

            fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
                // bool output has no in-place form (CPU: unreachable!())
                rstsr_raise!(UnImplemented, "query unaries have no in-place form")?
            }
        }
    };
}

impl_query_unary!(
    OpIsNanAPI,
    "isnan",
    "(isnanf(b[ib].re) || isnanf(b[ib].im))",
    "(isnan(b[ib].re) || isnan(b[ib].im))"
);
impl_query_unary!(OpIsInfAPI, "isinf", "(isinf(b[ib].re) || isinf(b[ib].im))", "(isinf(b[ib].re) || isinf(b[ib].im))");
impl_query_unary!(
    OpIsFiniteAPI,
    "isfinite",
    "(isfinite(b[ib].re) && isfinite(b[ib].im))",
    "(isfinite(b[ib].re) && isfinite(b[ib].im))"
);

/// `signbit`: bool out; real dtypes (CPU bound ExtReal).
impl<T, D> OpSignBitAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
    T: CudaDType + crate::dtype::CudaReal,
    D: DimAPI,
{
    type TOut = bool;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<bool>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::signbit(T::CNAME, "b[ib]");
        elementwise::unary_launch::<bool, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
        // bool output has no in-place form (CPU: unreachable!())
        rstsr_raise!(UnImplemented, "signbit has no in-place form")?
    }
}

/// `positive`: identity copy (CPU: clone per element).
impl<T, D> OpPositiveAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, "b[ib]")
    }
}

// /* #endregion */
