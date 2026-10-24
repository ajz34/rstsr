//! CUDA C element-expression builders for the op families that need
//! dtype-class dispatch beyond a plain infix operator.
//!
//! Semantics mirror the CPU devices' bound tables (`ExtReal` /
//! `ExtNum` / `ExtComplexFloat` in rstsr-dtype-traits): float
//! min/max propagate NaN, float remainder is the IEEE fmod-class with
//! NumPy sign fixing, complex rem is the Gaussian-integer remainder.
//! CUDA-native functions are used where CPU and CUDA semantics
//! coincide (`fmax/fmin` + explicit NaN guards, `rint`, `trunc`, ...).

use crate::dtype::{is_complex_cname, is_half_cname};

/// The compute-type C name an operand computes in (half promotes to
/// float, complex stays complex, everything else is itself).
pub(crate) fn compute_type(cname: &str) -> String {
    match cname {
        "__half" | "__nv_bfloat16" => "float".to_string(),
        _ => cname.to_string(),
    }
}

/// Load expression of operand `x` of C type `t` into its compute type.
pub(crate) fn load(t: &str, x: &str) -> String {
    match t {
        "__half" => format!("__half2float({x})"),
        "__nv_bfloat16" => format!("__nv_bfloat162float({x})"),
        _ => x.to_string(),
    }
}

/// Store expression of compute-type value `x` into C type `t`.
pub(crate) fn store(t: &str, x: &str) -> String {
    match t {
        "__half" => format!("__float2half({x})"),
        "__nv_bfloat16" => format!("__float2bfloat16({x})"),
        _ => x.to_string(),
    }
}

/// The real-math function suffix of a compute type (`f` for float).
pub(crate) fn f(compute: &str) -> &'static str {
    if compute == "float" {
        "f"
    } else {
        ""
    }
}

/* #region remainder */

/// `rem` element expression on operands `x` (dividend) and `y` (divisor)
/// of C types `tx`/`ty`, output type `tc`.
///
/// - ints: Rust `%` (truncating) with the CPU guard `0 % x == 0` for a zero divisor (NumPy parity;
///   a plain `%` traps/UBs);
/// - floats: CPU `ExtReal::ext_rem` = fmod then sign-fix toward the divisor (`rstsr_frem` device
///   helper below);
/// - half: round-trips through f32 (CPU does the same);
/// - complex: Gaussian-integer remainder (num-complex `Rem`).
pub(crate) fn rem(tx: &str, ty: &str, tc: &str, x: &str, y: &str) -> String {
    if is_complex_cname(tx) {
        return format!("rstsr_crem({x}, {y})");
    }
    if is_half_cname(tx) {
        let (lx, ly) = (load(tx, x), load(ty, y));
        return store(tx, &format!("rstsr_fremf({lx}, {ly})"));
    }
    if tx.contains("char") || tx.contains("short") || tx.contains("int") || tx.contains("long") {
        return format!("({y} == 0 ? ({tc})0 : ({tc})(({x}) % ({y})))");
    }
    let s = f(&compute_type(tc));
    format!("rstsr_frem{s}({x}, {y})")
}

/// `pow` element expression; ints use the shift-square `rstsr_ipow`
/// helper (CPU `ext_pow` for ints), floats/complex `pow`-class.
pub(crate) fn pow(tx: &str, ty: &str, _tc: &str, x: &str, y: &str) -> String {
    if is_complex_cname(tx) {
        return format!("rstsr_cpow({x}, {y})");
    }
    if is_half_cname(tx) || is_half_cname(ty) {
        let (lx, ly) = (load(tx, x), load(ty, y));
        return store(tx, &format!("powf({lx}, {ly})"));
    }
    if tx.contains("char") || tx.contains("short") || tx.contains("int") || tx.contains("long") {
        return format!("rstsr_ipow({x}, {y})");
    }
    format!("pow({x}, {y})")
}

/* #endregion */

/* #region min/max */

/// `maximum` element expression (CPU `ExtReal::ext_max`: NaN propagates
/// for floats; ints compare directly; half compares through float).
pub(crate) fn maximum(tc: &str, x: &str, y: &str) -> String {
    if is_complex_cname(tc) {
        // declined at the tensor layer (as on CPU); unreachable guard
        return format!("({x})");
    }
    if is_half_cname(tc) {
        let (lx, ly) = (load(tc, x), load(tc, y));
        let isnan = if tc == "__half" { "__hisnan" } else { "__nv_isnanbf" };
        return store(
            tc,
            &format!("({isnan}({lx}) ? ({lx}) : ({isnan}({ly}) ? ({ly}) : ({lx} > {ly} ? ({lx}) : ({ly}))))"),
        );
    }
    if tc == "float" || tc == "double" {
        return nan_max_min(tc, x, y, true);
    }
    format!("(({x}) > ({y}) ? ({x}) : ({y}))")
}

/// `minimum` element expression (see [`maximum`]).
pub(crate) fn minimum(tc: &str, x: &str, y: &str) -> String {
    if is_complex_cname(tc) {
        return format!("({x})");
    }
    if is_half_cname(tc) {
        let (lx, ly) = (load(tc, x), load(tc, y));
        let isnan = if tc == "__half" { "__hisnan" } else { "__nv_isnanbf" };
        return store(
            tc,
            &format!("({isnan}({lx}) ? ({lx}) : ({isnan}({ly}) ? ({ly}) : ({lx} < {ly} ? ({lx}) : ({ly}))))"),
        );
    }
    if tc == "float" || tc == "double" {
        return nan_max_min(tc, x, y, false);
    }
    format!("(({x}) < ({y}) ? ({x}) : ({y}))")
}

/// NaN-propagating fmax/fmin pair (the CPU float semantics); `max`
/// selects the direction.
fn nan_max_min(tc: &str, x: &str, y: &str, max: bool) -> String {
    // Either-operand NaN propagation via the single-instruction PTX
    // helpers (rstsr_maxnan*); the three-arm select is the portable floor
    // for other float classes.
    let s = f(tc);
    if tc == "double" {
        let op = if max { "rstsr_maxnand" } else { "rstsr_minnand" };
        return format!("{op}({x}, {y})");
    }
    if tc == "float" {
        let op = if max { "rstsr_maxnanf" } else { "rstsr_minnanf" };
        return format!("{op}({x}, {y})");
    }
    let op = if max { "fmax" } else { "fmin" };
    format!("(isnan{s}({x}) ? ({x}) : (isnan{s}({y}) ? ({y}) : {op}{s}({x}, {y})))")
}

/* #endregion */

/* #region unary float-class helpers */

/// `round` = ties-to-even: CUDA `rint` (both f32/f64, half via float).
pub(crate) fn round_even(tc: &str, x: &str) -> String {
    if is_complex_cname(tc) {
        let r = t_real(tc);
        return format!(
            "rstsr_make_complex(rint{rs}(({x}).re), rint{rs}(({x}).im))",
            rs = if r == "float" { "f" } else { "" },
        );
    }
    if is_half_cname(tc) {
        return store(tc, &format!("rintf({})", load(tc, x)));
    }
    format!("rint{}({})", f(tc), x)
}

/// `sign` (CPU `ExtNum::ext_sign`): NaN->NaN, ±1, 0->0; complex is
/// z/|z| with 0->0 (the rstsr_csign helpers in the complex header).
pub(crate) fn sign(tc: &str, x: &str) -> String {
    if is_complex_cname(tc) {
        return format!("rstsr_csign({x})");
    }
    if is_half_cname(tc) {
        let lx = load(tc, x);
        return store(tc, &format!("(isnan({lx}) ? ({lx}) : (({lx}) > 0.0f ? 1.0f : (({lx}) < 0.0f ? -1.0f : 0.0f)))"));
    }
    if tc == "float" || tc == "double" {
        let s = f(tc);
        return format!("(isnan{s}({x}) ? ({x}) : (({x}) > 0 ? ({tc})1 : (({x}) < 0 ? ({tc})-1 : ({tc})0)))");
    }
    // integers: CPU ext_sign for ints is -1/0/1 by comparison
    format!("((({x}) > 0) - (({x}) < 0))")
}

/// `signbit` element expression into bool.
pub(crate) fn signbit(tc: &str, x: &str) -> String {
    if is_complex_cname(tc) {
        // declined on CPU (signbit of complex is not ExtReal); unreachable
        return "false".to_string();
    }
    if is_half_cname(tc) {
        let isnan = if tc == "__half" { "__hisnan" } else { "__nv_isnanbf" };
        let _ = isnan;
        // sign bit test through float conversion preserves -0.0
        return format!("(signbitf({}))", load(tc, x));
    }
    format!("(signbit({}))", x)
}

/// `abs` into the output C type `tout` (complex -> magnitude (real),
/// half -> half, ints/floats -> same-type abs).
/// The real component C type of a (possibly complex) C type name.
pub(crate) fn t_real(t: &str) -> &'static str {
    if t == "rstsr_c32" {
        "float"
    } else {
        "double"
    }
}

pub(crate) fn abs(tx: &str, tout: &str, x: &str) -> String {
    if is_complex_cname(tx) {
        let v = format!("rstsr_cabs({x})");
        // c32 -> f32 / c64 -> f64: no further cast needed when tout matches
        if (tx == "rstsr_c32" && tout == "float") || (tx == "rstsr_c64" && tout == "double") {
            return v;
        }
        return format!("(({tout})({v}))");
    }
    if is_half_cname(tx) {
        let lx = load(tx, x);
        return store(tx, &format!("fabsf({lx})"));
    }
    if tx.contains("unsigned") {
        // unsigned abs is identity on CPU
        return format!("({x})");
    }
    format!("abs{}({})", f(tx), x)
}

/* #endregion */
