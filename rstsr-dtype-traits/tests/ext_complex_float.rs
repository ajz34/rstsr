//! Integration tests for the `ext_complex_float` module.

use num::traits::{Float, FloatConst};
use num::Complex;
use rstsr_dtype_traits::*;

fn assert_close(actual: Complex<f64>, expected: Complex<f64>) {
    // relative tolerance, scaled to the expected magnitude so that tiny values
    // (e.g. 1e-20) are compared tightly rather than against a 1e-12 absolute bound.
    let tol = 1e-14 * expected.norm().max(f64::MIN_POSITIVE);
    let err = (actual - expected).norm();
    assert!(err <= tol, "actual = {actual:?}, expected = {expected:?}, err = {err:e}");
}

#[test]
fn test_log1p_real_special_cases() {
    assert!(f64::NAN.ext_log_1p().is_nan());
    assert!((-2.0_f64).ext_log_1p().is_nan()); // x < -1
    assert_eq!((-1.0_f64).ext_log_1p(), f64::NEG_INFINITY);
    assert_eq!(f64::INFINITY.ext_log_1p(), f64::INFINITY);
    assert!(f64::NEG_INFINITY.ext_log_1p().is_nan()); // x < -1
}

#[test]
fn test_log1p_real_precision() {
    // naive ln(1 + x) returns 0 here; log1p must return ~x
    let x = 1e-20_f64;
    assert!((x.ext_log_1p() - x).abs() < 1e-36);
    assert_eq!((1.0_f64 + x).ln(), 0.0); // documents the naive failure
                                         // -0 must be preserved
    assert!((-0.0_f64).ext_log_1p().is_sign_negative() || (-0.0_f64).ext_log_1p() == 0.0);
}

#[test]
fn test_log1p_complex_special_cases() {
    let c = Complex::new;
    // z = -1 + 0j -> -inf + 0j
    let r = c(-1.0_f64, 0.0).ext_log_1p();
    assert!(r.re.is_infinite() && r.re < 0.0 && r.im == 0.0, "{r:?}");
    // a finite, b = +inf -> +inf + pi/2 j
    let r = c(1.0, f64::INFINITY).ext_log_1p();
    assert!(r.re.is_infinite() && r.re > 0.0 && (r.im - std::f64::consts::FRAC_PI_2).abs() < 1e-12, "{r:?}");
    // a = +inf, b = +inf -> +inf + pi/4 j
    let r = c(f64::INFINITY, f64::INFINITY).ext_log_1p();
    assert!(r.re.is_infinite() && r.re > 0.0 && (r.im - std::f64::consts::FRAC_PI_4).abs() < 1e-12, "{r:?}");
    // NaN + finite -> NaN + NaN j
    let r = c(f64::NAN, 1.0).ext_log_1p();
    assert!(r.re.is_nan() && r.im.is_nan(), "{r:?}");
}

#[test]
fn test_log1p_complex_precision() {
    let z = Complex::new(1e-20_f64, 1e-20_f64);
    // naive ln(1 + z) loses the real part; compensated form keeps it
    assert_close(z.ext_log_1p(), z);
    let naive = (Complex::new(1.0, 0.0) + z).ln();
    assert!(naive.re.abs() < 1e-30, "naive real part = {:e}", naive.re);
}

#[test]
fn test_expm1_real_special_cases() {
    assert!(f64::NAN.ext_exp_m1().is_nan());
    assert_eq!(f64::INFINITY.ext_exp_m1(), f64::INFINITY);
    assert_eq!(f64::NEG_INFINITY.ext_exp_m1(), -1.0);
    assert_eq!(1e-20_f64.ext_exp_m1(), 1e-20);
}

#[test]
fn test_expm1_complex_special_cases() {
    let c = Complex::new;
    // a = -inf, b finite -> -1 + 0j
    let r = c(f64::NEG_INFINITY, 2.0).ext_exp_m1();
    assert!((r.re + 1.0).abs() < 1e-12 && r.im.abs() < 1e-12, "{r:?}");
    // a = +inf, b = +0 -> +inf + 0j
    let r = c(f64::INFINITY, 0.0).ext_exp_m1();
    assert!(r.re.is_infinite() && r.re > 0.0 && r.im == 0.0, "{r:?}");
    // a finite, b = +inf -> NaN + NaN j
    let r = c(1.0, f64::INFINITY).ext_exp_m1();
    assert!(r.re.is_nan() && r.im.is_nan(), "{r:?}");
    // a = ±0, b = +0 -> +0 + 0j (both components positive)
    for a in [0.0_f64, -0.0] {
        let r = c(a, 0.0).ext_exp_m1();
        assert!(!r.re.is_sign_negative() && !r.im.is_sign_negative(), "a={a}: {r:?}");
    }
}

#[test]
fn test_expm1_complex_precision() {
    let z = Complex::new(1e-20_f64, 1e-20_f64);
    assert_close(z.ext_exp_m1(), z);
    // large negative real part: must not overflow the identity
    let r = Complex::new(-2000.0_f64, 0.5).ext_exp_m1();
    assert!((r.re + 1.0).abs() < 1e-12 && r.im.abs() < 1e-12, "{r:?}");
}

/// Expected value of one component: NaN (matches any NaN), or a number.
/// A zero expected value matches either sign of zero, which the standard
/// leaves unspecified.
#[derive(Clone, Copy)]
enum V {
    Nan,
    Num(f64),
}
use V::{Nan, Num};

#[allow(clippy::redundant_guards)] // `Num(v) if v == 0.0` is not a plain pattern
fn chk(label: &str, got: Complex<f64>, re: V, im: V) {
    let cmp = |a: f64, e: V, which: &str| match e {
        Nan => assert!(a.is_nan(), "{label} {which}: got {a}, want NaN"),
        Num(v) if v.is_infinite() => assert_eq!(a, v, "{label} {which}: got {a}, want {v}"),
        Num(v) if v == 0.0 => assert_eq!(a, 0.0, "{label} {which}: got {a}, want ±0"),
        Num(v) => assert!((a - v).abs() <= 1e-15 * v.abs().max(1.0), "{label} {which}: got {a}, want {v}"),
    };
    cmp(got.re, re, "(re)");
    cmp(got.im, im, "(im)");
}

/// Special values of the C99 complex elementary functions, against the
/// values the array API conformance suite prescribes (which numpy matches).
#[test]
fn test_c99_special_values() {
    use std::f64::consts::{FRAC_PI_2 as H, FRAC_PI_4 as Q, PI};
    let inf = f64::INFINITY;
    let c = Complex::new;

    // square root
    chk("sqrt(inf,nan)", c(inf, f64::NAN).ext_sqrt(), Num(inf), Nan);
    chk("sqrt(inf,3)", c(inf, 3.0).ext_sqrt(), Num(inf), Num(0.0));
    chk("sqrt(-inf,nan)", c(-inf, f64::NAN).ext_sqrt(), Nan, Num(inf));
    chk("sqrt(-4,0)", c(-4.0, 0.0).ext_sqrt(), Num(0.0), Num(2.0));

    // hyperbolic cosine
    chk("cosh(0,inf)", c(0.0, inf).ext_cosh(), Nan, Num(0.0));
    chk("cosh(0,nan)", c(0.0, f64::NAN).ext_cosh(), Nan, Num(0.0));
    chk("cosh(inf,0)", c(inf, 0.0).ext_cosh(), Num(inf), Num(0.0));
    chk("cosh(inf,inf)", c(inf, inf).ext_cosh(), Num(inf), Nan);
    chk("cosh(inf,nan)", c(inf, f64::NAN).ext_cosh(), Num(inf), Nan);
    chk("cosh(nan,0)", c(f64::NAN, 0.0).ext_cosh(), Nan, Num(0.0));

    // hyperbolic sine
    chk("sinh(0,inf)", c(0.0, inf).ext_sinh(), Num(0.0), Nan);
    chk("sinh(0,nan)", c(0.0, f64::NAN).ext_sinh(), Num(0.0), Nan);
    chk("sinh(inf,0)", c(inf, 0.0).ext_sinh(), Num(inf), Num(0.0));
    chk("sinh(inf,inf)", c(inf, inf).ext_sinh(), Num(inf), Nan);
    chk("sinh(inf,nan)", c(inf, f64::NAN).ext_sinh(), Num(inf), Nan);
    chk("sinh(nan,0)", c(f64::NAN, 0.0).ext_sinh(), Nan, Num(0.0));

    // hyperbolic tangent
    chk("tanh(0,inf)", c(0.0, inf).ext_tanh(), Num(0.0), Nan);
    chk("tanh(0,nan)", c(0.0, f64::NAN).ext_tanh(), Num(0.0), Nan);
    chk("tanh(inf,inf)", c(inf, inf).ext_tanh(), Num(1.0), Num(0.0));
    chk("tanh(inf,nan)", c(inf, f64::NAN).ext_tanh(), Num(1.0), Num(0.0));
    chk("tanh(inf,1)", c(inf, 1.0).ext_tanh(), Num(1.0), Num(0.0));
    chk("tanh(nan,0)", c(f64::NAN, 0.0).ext_tanh(), Nan, Num(0.0));

    // arc cosine
    chk("acos(0,nan)", c(0.0, f64::NAN).ext_acos(), Num(H), Nan);
    chk("acos(-0,nan)", c(-0.0, f64::NAN).ext_acos(), Num(H), Nan);
    chk("acos(inf,nan)", c(inf, f64::NAN).ext_acos(), Nan, Num(-inf));
    chk("acos(-inf,nan)", c(-inf, f64::NAN).ext_acos(), Nan, Num(-inf));
    chk("acos(1,inf)", c(1.0, inf).ext_acos(), Num(H), Num(-inf));
    chk("acos(inf,inf)", c(inf, inf).ext_acos(), Num(Q), Num(-inf));
    chk("acos(inf,1)", c(inf, 1.0).ext_acos(), Num(0.0), Num(-inf));
    chk("acos(-inf,inf)", c(-inf, inf).ext_acos(), Num(3.0 * Q), Num(-inf));
    chk("acos(-inf,1)", c(-inf, 1.0).ext_acos(), Num(PI), Num(-inf));
    chk("acos(nan,inf)", c(f64::NAN, inf).ext_acos(), Nan, Num(-inf));

    // inverse hyperbolic cosine
    chk("acosh(inf,nan)", c(inf, f64::NAN).ext_acosh(), Num(inf), Nan);
    chk("acosh(-inf,nan)", c(-inf, f64::NAN).ext_acosh(), Num(inf), Nan);
    chk("acosh(1,inf)", c(1.0, inf).ext_acosh(), Num(inf), Num(H));
    chk("acosh(0,nan)", c(0.0, f64::NAN).ext_acosh(), Nan, Num(H));
    chk("acosh(inf,inf)", c(inf, inf).ext_acosh(), Num(inf), Num(Q));
    chk("acosh(inf,1)", c(inf, 1.0).ext_acosh(), Num(inf), Num(0.0));
    chk("acosh(-inf,inf)", c(-inf, inf).ext_acosh(), Num(inf), Num(3.0 * Q));
    chk("acosh(-inf,1)", c(-inf, 1.0).ext_acosh(), Num(inf), Num(PI));
    chk("acosh(nan,inf)", c(f64::NAN, inf).ext_acosh(), Num(inf), Nan);

    // inverse hyperbolic sine
    chk("asinh(1,inf)", c(1.0, inf).ext_asinh(), Num(inf), Num(H));
    chk("asinh(inf,inf)", c(inf, inf).ext_asinh(), Num(inf), Num(Q));
    chk("asinh(inf,1)", c(inf, 1.0).ext_asinh(), Num(inf), Num(0.0));
    chk("asinh(nan,0)", c(f64::NAN, 0.0).ext_asinh(), Nan, Num(0.0));
    chk("asinh(nan,inf)", c(f64::NAN, inf).ext_asinh(), Num(inf), Nan);
    chk("asinh(0,inf)", c(0.0, inf).ext_asinh(), Num(inf), Num(H));

    // inverse hyperbolic tangent
    chk("atanh(1,inf)", c(1.0, inf).ext_atanh(), Num(0.0), Num(H));
    chk("atanh(0,nan)", c(0.0, f64::NAN).ext_atanh(), Num(0.0), Nan);
    chk("atanh(inf,inf)", c(inf, inf).ext_atanh(), Num(0.0), Num(H));
    chk("atanh(inf,nan)", c(inf, f64::NAN).ext_atanh(), Num(0.0), Nan);
    chk("atanh(inf,1)", c(inf, 1.0).ext_atanh(), Num(0.0), Num(H));
    chk("atanh(nan,inf)", c(f64::NAN, inf).ext_atanh(), Num(0.0), Num(H));
}

fn chk_close(label: &str, got: Complex<f64>, want: Complex<f64>, tol: f64) {
    let err = (got - want).norm();
    let scale = want.norm().max(1.0);
    assert!(err <= tol * scale, "{label}: got {got:?}, want {want:?} (err {err:e})");
}

fn to64(z: Complex<f32>) -> Complex<f64> {
    Complex::new(z.re as f64, z.im as f64)
}

fn apply<T: Float + FloatConst>(name: &str, z: Complex<T>) -> Complex<T> {
    match name {
        "sqrt" => z.ext_sqrt(),
        "cosh" => z.ext_cosh(),
        "sinh" => z.ext_sinh(),
        "tanh" => z.ext_tanh(),
        "tan" => z.ext_tan(),
        "acos" => z.ext_acos(),
        "asin" => z.ext_asin(),
        "acosh" => z.ext_acosh(),
        "asinh" => z.ext_asinh(),
        "atanh" => z.ext_atanh(),
        _ => unreachable!(),
    }
}

/// `(function, input.re, input.im, expected.re, expected.im)`, from numpy.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const CASES_F64: &[(&str, f64, f64, f64, f64)] = &[
    ("sqrt", 0.5, 0.5, 0.7768869870150187, 0.3217971264527913),
    ("sqrt", 2.0, 3.0, 1.6741492280355401, 0.8959774761298381),
    ("sqrt", -1.5, 0.25, 0.10171192794985977, 1.2289610719169575),
    ("sqrt", 10.0, 0.001, 3.1622776641212265, 0.0001581138828107766),
    ("sqrt", 1e-08, 1e-08, 0.00010986841134678099, 4.5508986056222736e-05),
    ("sqrt", -0.9, 0.9, 0.43173614982752234, 1.0423032682803468),
    ("sqrt", 700.0, 0.5, 26.457514797987034, 0.009449111222608888),
    ("sqrt", 0.0, 0.0, 0.0, 0.0),
    ("sqrt", 1.0, 1.0, 1.09868411346781, 0.45508986056222733),
    ("sqrt", -2.0, 0.0, 0.0, 1.4142135623730951),
    ("sqrt", 0.999999, 0.001, 0.9999996250000235, 0.0005000001875000586),
    ("sqrt", 30.0, -0.4, 5.477347284414176, -0.0365140257892906),
    ("cosh", 0.5, 0.5, 0.9895848833999199, 0.24982639750046154),
    ("cosh", 2.0, 3.0, -3.7245455049153224, 0.5118225699873846),
    ("cosh", -1.5, 0.25, 2.2792788971607405, -0.526792167549771),
    ("cosh", 10.0, 0.001, 11013.227413487324, 11.013231039164673),
    ("cosh", 1e-08, 1e-08, 1.0, 1.0000000000000002e-16),
    ("cosh", -0.9, 0.9, 0.8908207825879338, -0.804098174429908),
    ("cosh", 700.0, 0.5, 4.45036182472841e+303, 2.431243745554885e+303),
    ("cosh", 0.0, 0.0, 1.0, 0.0),
    ("cosh", 1.0, 1.0, 0.8337300251311491, 0.9888977057628651),
    ("cosh", -2.0, 0.0, 3.7621956910836314, -0.0),
    ("cosh", 0.999999, 0.001, 1.5430786880751561, 0.0011751994546971556),
    ("cosh", 30.0, -0.4, 4921447450222.744, -2080754608330.393),
    ("sinh", 0.5, 0.5, 0.4573041531842493, 0.5406126857131534),
    ("sinh", 2.0, 3.0, -3.59056458998578, 0.5309210862485197),
    ("sinh", -1.5, 0.25, -2.0630853133346414, 0.5819954525995883),
    ("sinh", 10.0, 0.001, 11013.227368087415, 11.013231084564596),
    ("sinh", 1e-08, 1e-08, 1.0000000000000002e-08, 1e-08),
    ("sinh", -0.9, 0.9, -0.6380930292967651, 1.122575129542809),
    ("sinh", 700.0, 0.5, 4.45036182472841e+303, 2.431243745554885e+303),
    ("sinh", 0.0, 0.0, 0.0, 0.0),
    ("sinh", 1.0, 1.0, 0.6349639147847361, 1.2984575814159773),
    ("sinh", -2.0, 0.0, -3.626860407847019, 0.0),
    ("sinh", 0.999999, 0.001, 1.175199062963978, 0.0015430792024349247),
    ("sinh", 30.0, -0.4, 4921447450222.744, -2080754608330.393),
    ("tanh", 0.5, 0.5, 0.5640831412674985, 0.40389645531602575),
    ("tanh", 2.0, 3.0, 0.965385879022133, -0.009884375038322492),
    ("tanh", -1.5, 0.25, -0.9152719132613141, 0.04380217692516719),
    ("tanh", 10.0, 0.001, 0.9999999958777012, 8.244608959358927e-12),
    ("tanh", 1e-08, 1e-08, 1.0000000000000002e-08, 1e-08),
    ("tanh", -0.9, 0.9, -1.0214921459533572, 0.33810971373189097),
    ("tanh", 700.0, 0.5, 1.0, 0.0),
    ("tanh", 0.0, 0.0, 0.0, 0.0),
    ("tanh", 1.0, 1.0, 1.0839233273386946, 0.2717525853195118),
    ("tanh", -2.0, 0.0, -0.9640275800758168, 0.0),
    ("tanh", 0.999999, 0.001, 0.7615940558314462, 0.0004199748777099631),
    ("tanh", 30.0, -0.4, 1.0, -1.2563072661295148e-26),
    ("tan", 0.5, 0.5, 0.40389645531602575, 0.5640831412674985),
    ("tan", 2.0, 3.0, -0.0037640256415042484, 1.0032386273536096),
    ("tan", -1.5, 0.25, -1.0253320612293393, 3.7861089368147747),
    ("tan", 10.0, 0.001, 0.6483599065466764, 0.0014203706920430333),
    ("tan", 1e-08, 1e-08, 1e-08, 1.0000000000000002e-08),
    ("tan", -0.9, 0.9, -0.33810971373189097, 1.0214921459533572),
    ("tan", 700.0, 0.5, -0.467846449078739, 0.6022741715864179),
    ("tan", 0.0, 0.0, 0.0, 0.0),
    ("tan", 1.0, 1.0, 0.2717525853195118, 1.0839233273386946),
    ("tan", -2.0, 0.0, 2.185039863261519, 0.0),
    ("tan", 0.999999, 0.001, 1.5573989642567916, 0.003425498700579091),
    ("tan", 30.0, -0.4, -0.7916707307799349, -2.306637181427909),
    ("acos", 0.5, 0.5, 1.118517879643706, -0.5306375309525179),
    ("acos", 2.0, 3.0, 1.0001435424737972, -1.9833870299165355),
    ("acos", -1.5, 0.25, 2.925421937341255, -0.9937304660372428),
    ("acos", 10.0, 0.001, 0.0001005037811823974, -2.9932228512023293),
    ("acos", 1e-08, 1e-08, 1.5707963167948966, -1e-08),
    ("acos", -0.9, 0.9, 2.2123245753427025, -0.9659563609033214),
    ("acos", 700.0, 0.5, 0.0007142863216719347, -7.244227260501635),
    ("acos", 0.0, 0.0, 1.5707963267948966, -0.0),
    ("acos", 1.0, 1.0, 0.9045568943023813, -1.0612750619050357),
    ("acos", -2.0, 0.0, 3.141592653589793, -1.3169578969248166),
    ("acos", 0.999999, 0.001, 0.031635960069074126, -0.03160960775982661),
    ("acos", 30.0, -0.4, 0.013339954241577329, 4.094155697930839),
    ("asin", 0.5, 0.5, 0.45227844715119064, 0.5306375309525179),
    ("asin", 2.0, 3.0, 0.5706527843210994, 1.9833870299165355),
    ("asin", -1.5, 0.25, -1.3546256105463583, 0.9937304660372428),
    ("asin", 10.0, 0.001, 1.5706958230137142, 2.9932228512023293),
    ("asin", 1e-08, 1e-08, 1e-08, 1e-08),
    ("asin", -0.9, 0.9, -0.6415282485478061, 0.9659563609033214),
    ("asin", 700.0, 0.5, 1.5700820404732247, 7.244227260501635),
    ("asin", 0.0, 0.0, 0.0, 0.0),
    ("asin", 1.0, 1.0, 0.6662394324925153, 1.0612750619050357),
    ("asin", -2.0, 0.0, -1.5707963267948966, 1.3169578969248166),
    ("asin", 0.999999, 0.001, 1.5391603667258225, 0.03160960775982661),
    ("asin", 30.0, -0.4, 1.5574563725533193, -4.094155697930839),
    ("acosh", 0.5, 0.5, 0.5306375309525179, 1.118517879643706),
    ("acosh", 2.0, 3.0, 1.9833870299165355, 1.0001435424737972),
    ("acosh", -1.5, 0.25, 0.9937304660372428, 2.925421937341255),
    ("acosh", 10.0, 0.001, 2.9932228512023293, 0.0001005037811823974),
    ("acosh", 1e-08, 1e-08, 1e-08, 1.5707963167948966),
    ("acosh", -0.9, 0.9, 0.9659563609033214, 2.2123245753427025),
    ("acosh", 700.0, 0.5, 7.244227260501635, 0.0007142863216719347),
    ("acosh", 0.0, 0.0, 0.0, 1.5707963267948966),
    ("acosh", 1.0, 1.0, 1.0612750619050357, 0.9045568943023813),
    ("acosh", -2.0, 0.0, 1.3169578969248166, 3.141592653589793),
    ("acosh", 0.999999, 0.001, 0.03160960775982661, 0.031635960069074126),
    ("acosh", 30.0, -0.4, 4.094155697930839, -0.013339954241577329),
    ("asinh", 0.5, 0.5, 0.5306375309525179, 0.45227844715119064),
    ("asinh", 2.0, 3.0, 1.9686379257930964, 0.9646585044076028),
    ("asinh", -1.5, 0.25, -1.202745551643968, 0.13819517113326382),
    ("asinh", 10.0, 0.001, 2.9982229552238966, 9.950371869748096e-05),
    ("asinh", 1e-08, 1e-08, 1e-08, 1e-08),
    ("asinh", -0.9, 0.9, -0.9659563609033214, 0.6415282485478061),
    ("asinh", 700.0, 0.5, 7.244228280908236, 0.0007142848639474748),
    ("asinh", 0.0, 0.0, 0.0, 0.0),
    ("asinh", 1.0, 1.0, 1.0612750619050357, 0.6662394324925153),
    ("asinh", -2.0, 0.0, -1.4436354751788103, 0.0),
    ("asinh", 0.999999, 0.001, 0.8813730566893797, 0.0007071071052772739),
    ("asinh", 30.0, -0.4, 4.094710957420225, -0.013325144681435648),
    ("atanh", 0.5, 0.5, 0.40235947810852507, 0.5535743588970452),
    ("atanh", 2.0, 3.0, 0.14694666622552977, 1.3389725222944935),
    ("atanh", -1.5, 0.25, -0.7514206511017898, 1.3888068485400746),
    ("atanh", 10.0, 0.001, 0.10033534671077154, 1.570786225784899),
    ("atanh", 1e-08, 1e-08, 1e-08, 1.0000000000000002e-08),
    ("atanh", -0.9, 0.9, -0.42114765870336124, 0.951256664298873),
    ("atanh", 700.0, 0.5, 0.00142857167152434, 1.5707953063851714),
    ("atanh", 0.0, 0.0, 0.0, 0.0),
    ("atanh", 1.0, 1.0, 0.40235947810852507, 1.0172219678978514),
    ("atanh", -2.0, 0.0, -0.5493061443340549, 1.5707963267948966),
    ("atanh", 0.999999, 0.001, 3.8004507922711586, 0.7851481636682672),
    ("atanh", 30.0, -0.4, 0.033339749191885225, -1.5703514672654988),
];

/// Same, evaluated end to end in single precision (complex64 I/O).
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const CASES_F32: &[(&str, f64, f64, f64, f64)] = &[
    ("sqrt", 0.5, 0.5, 0.7768869996070862, 0.32179713249206543),
    ("sqrt", 2.0, 3.0, 1.6741492748260498, 0.8959774374961853),
    ("sqrt", 10.0, 0.0010000000474974513, 3.1622776985168457, 0.00015811389312148094),
    ("sqrt", 7281.0, -1.0, 85.32877349853516, -0.005859687924385071),
    ("sqrt", -2731.0, 1.0, 0.00956773478537798, 52.25897216796875),
    ("sqrt", 1.0, 45.0, 4.79641056060791, 4.6910080909729),
    ("sqrt", 0.9990000128746033, 0.0010000000474974513, 0.9994999766349792, 0.0005002501420676708),
    ("cosh", 0.5, 0.5, 0.9895848631858826, 0.24982638657093048),
    ("cosh", 2.0, 3.0, -3.724545478820801, 0.511822521686554),
    ("cosh", 10.0, 0.0010000000474974513, 11013.228515625, 11.01323127746582),
    ("cosh", 1.0, 45.0, 0.8106141686439514, 0.9999828338623047),
    ("cosh", 0.9990000128746033, 0.0010000000474974513, 1.5419055223464966, 0.0011736586457118392),
    ("sinh", 0.5, 0.5, 0.45730412006378174, 0.5406126976013184),
    ("sinh", 2.0, 3.0, -3.590564489364624, 0.5309210419654846),
    ("sinh", 10.0, 0.0010000000474974513, 11013.2275390625, 11.013232231140137),
    ("sinh", 1.0, 45.0, 0.6173589825630188, 1.313012719154358),
    ("sinh", 0.9990000128746033, 0.0010000000474974513, 1.1736581325531006, 0.0015419061528518796),
    ("tanh", 0.5, 0.5, 0.5640830993652344, 0.4038964807987213),
    ("tanh", 2.0, 3.0, 0.9653858542442322, -0.009884374216198921),
    ("tanh", 10.0, 0.0010000000474974513, 1.0000001192092896, 8.244610723295853e-12),
    ("tanh", 7281.0, -1.0, 1.0, -0.0),
    ("tanh", -2731.0, 1.0, -1.0, 0.0),
    ("tanh", 1.0, 45.0, 1.0943653583526611, 0.26975369453430176),
    ("tanh", 0.9990000128746033, 0.0010000000474974513, 0.761174201965332, 0.00042061429121531546),
    ("tan", 0.5, 0.5, 0.4038964807987213, 0.5640830993652344),
    ("tan", 2.0, 3.0, -0.003764025866985321, 1.0032386779785156),
    ("tan", 10.0, 0.0010000000474974513, 0.6483599543571472, 0.0014203706523403525),
    ("tan", 7281.0, -1.0, -0.21864227950572968, -1.2052949666976929),
    ("tan", -2731.0, 1.0, -0.27493351697921753, 1.0581328868865967),
    ("tan", 1.0, 45.0, 1.4901590038430467e-39, 1.0),
    ("tan", 0.9990000128746033, 0.0010000000474974513, 1.5539823770523071, 0.0034148681443184614),
    ("acos", 0.5, 0.5, 1.1185178756713867, -0.5306375622749329),
    ("acos", 2.0, 3.0, 1.0001435279846191, -1.9833869934082031),
    ("acos", 10.0, 0.0010000000474974513, 0.00010050377750303596, -2.993222951889038),
    ("acos", 7281.0, -1.0, 0.0001373437698930502, 9.58617115020752),
    ("acos", -2731.0, 1.0, 3.1412265300750732, -8.605570793151855),
    ("acos", 1.0, 45.0, 1.5485832691192627, -4.500179767608643),
    ("acos", 0.9990000128746033, 0.0010000000474974513, 0.049136821180582047, -0.020358122885227203),
    ("asin", 0.5, 0.5, 0.4522784352302551, 0.5306375622749329),
    ("asin", 2.0, 3.0, 0.5706527829170227, 1.9833869934082031),
    ("asin", 10.0, 0.0010000000474974513, 1.5706958770751953, 2.993222951889038),
    ("asin", 7281.0, -1.0, 1.5706590414047241, -9.58617115020752),
    ("asin", -2731.0, 1.0, -1.5704301595687866, 8.605570793151855),
    ("asin", 1.0, 45.0, 0.022213084623217583, 4.500179767608643),
    ("asin", 0.9990000128746033, 0.0010000000474974513, 1.52165949344635, 0.020358122885227203),
    ("acosh", 0.5, 0.5, 0.5306375622749329, 1.1185178756713867),
    ("acosh", 2.0, 3.0, 1.9833869934082031, 1.0001435279846191),
    ("acosh", 10.0, 0.0010000000474974513, 2.993222951889038, 0.00010050377750303596),
    ("acosh", 7281.0, -1.0, 9.58617115020752, -0.0001373437698930502),
    ("acosh", -2731.0, 1.0, 8.605570793151855, 3.1412265300750732),
    ("acosh", 1.0, 45.0, 4.500179767608643, 1.5485832691192627),
    ("acosh", 0.9990000128746033, 0.0010000000474974513, 0.020358122885227203, 0.049136821180582047),
    ("asinh", 0.5, 0.5, 0.5306375622749329, 0.4522784352302551),
    ("asinh", 2.0, 3.0, 1.9686379432678223, 0.9646584987640381),
    ("asinh", 10.0, 0.0010000000474974513, 2.998222827911377, 9.950372623279691e-05),
    ("asinh", 7281.0, -1.0, 9.58617115020752, -0.0001373437698930502),
    ("asinh", -2731.0, 1.0, -8.605570793151855, 0.00036616623401641846),
    ("asinh", 1.0, 45.0, 4.499933242797852, 1.548572301864624),
    ("asinh", 0.9990000128746033, 0.0010000000474974513, 0.8806665539741516, 0.0007074603927321732),
    ("atanh", 0.5, 0.5, 0.4023594856262207, 0.5535743832588196),
    ("atanh", 2.0, 3.0, 0.14694666862487793, 1.338972568511963),
    ("atanh", 10.0, 0.0010000000474974513, 0.10033534467220306, 1.5707862377166748),
    ("atanh", 7281.0, -1.0, 0.0001373437698930502, -1.570796251296997),
    ("atanh", -2731.0, 1.0, -0.000366166204912588, 1.570796251296997),
    ("atanh", 1.0, 45.0, 0.0004933400778099895, 1.548588752746582),
    ("atanh", 0.9990000128746033, 0.0010000000474974513, 3.626917600631714, 0.39295244216918945),
];

#[test]
fn test_c99_finite_values_f64() {
    for &(name, a, b, er, ei) in CASES_F64 {
        chk_close(name, apply(name, Complex::new(a, b)), Complex::new(er, ei), 1e-13);
    }
}

#[test]
fn test_c99_finite_values_f32() {
    for &(name, a, b, er, ei) in CASES_F32 {
        let z = Complex::<f32>::new(a as f32, b as f32);
        chk_close(name, to64(apply(name, z)), Complex::new(er, ei), 1e-5);
    }
}
