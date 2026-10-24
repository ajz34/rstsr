// Complex-math prelude for rstsr-cuda generated kernels.
//
// Real-type math uses CUDA's libdevice (sin/cos/exp/log/... directly);
// complex operations are built from real libdevice calls. Formula
// provenance: the Smith / Hull–Ashby–Dieter–Hohlfeld formulas for
// sqrt/log/exp and the inverse trig/hyperbolic set, as specialized in
// CuPy's `complex.cuh` (following cuComplex.h / C++ std semantics);
// pytorch's `ComplexMath.cuh` uses the same formulas. Simple ops
// (neg/conj/abs-magnitude/round components) are component-wise.
//
// One C++ template `rstsr_complex<T>` serves both widths; the `rstsr_c32` /
// `rstsr_c64` struct aliases live in the shared Rust-side prelude
// (kernel.rs CUDA_PRELUDE, always prepended before this header). Layout is
// {re, im} — byte-identical to the previous plain structs, so device
// buffers, marshaling, and cache keys are unchanged. Real-math calls below
// resolve per-T through the `rstsr_r*` overloads (NVRTC has no <math.h> on
// the include path, and float/double overload pairs keep the suffix logic
// in C++ instead of Rust-side name branching).

#pragma once

// --- per-T real-math overload surface ---------------------------------------

__device__ float rstsr_rabs(float x) { return fabsf(x); }
__device__ double rstsr_rabs(double x) { return fabs(x); }
__device__ float rstsr_rsqrt(float x) { return sqrtf(x); }
__device__ double rstsr_rsqrt(double x) { return sqrt(x); }
__device__ float rstsr_rexp(float x) { return expf(x); }
__device__ double rstsr_rexp(double x) { return exp(x); }
__device__ float rstsr_rlog(float x) { return logf(x); }
__device__ double rstsr_rlog(double x) { return log(x); }
__device__ float rstsr_rlog1p(float x) { return log1pf(x); }
__device__ double rstsr_rlog1p(double x) { return log1p(x); }
__device__ float rstsr_ratan2(float y, float x) { return atan2f(y, x); }
__device__ double rstsr_ratan2(double y, double x) { return atan2(y, x); }
__device__ float rstsr_rsin(float x) { return sinf(x); }
__device__ double rstsr_rsin(double x) { return sin(x); }
__device__ float rstsr_rcos(float x) { return cosf(x); }
__device__ double rstsr_rcos(double x) { return cos(x); }
__device__ float rstsr_rtan(float x) { return tanf(x); }
__device__ double rstsr_rtan(double x) { return tan(x); }
__device__ float rstsr_rsinh(float x) { return sinhf(x); }
__device__ double rstsr_rsinh(double x) { return sinh(x); }
__device__ float rstsr_rcosh(float x) { return coshf(x); }
__device__ double rstsr_rcosh(double x) { return cosh(x); }
__device__ float rstsr_rasin(float x) { return asinf(x); }
__device__ double rstsr_rasin(double x) { return asin(x); }
__device__ float rstsr_racos(float x) { return acosf(x); }
__device__ double rstsr_racos(double x) { return acos(x); }
__device__ float rstsr_ratanh(float x) { return atanhf(x); }
__device__ double rstsr_ratanh(double x) { return atanh(x); }
__device__ float rstsr_ratan(float x) { return atanf(x); }
__device__ double rstsr_ratan(double x) { return atan(x); }
__device__ float rstsr_rcopysign(float x, float y) { return copysignf(x, y); }
__device__ double rstsr_rcopysign(double x, double y) { return copysign(x, y); }
__device__ float rstsr_rhypot(float x, float y) { return hypotf(x, y); }
__device__ double rstsr_rhypot(double x, double y) { return hypot(x, y); }

// --- component-wise basics -------------------------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_cadd(rstsr_complex<T> a, rstsr_complex<T> b) {
    return rstsr_complex<T>(a.re + b.re, a.im + b.im);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_csub(rstsr_complex<T> a, rstsr_complex<T> b) {
    return rstsr_complex<T>(a.re - b.re, a.im - b.im);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_cneg(rstsr_complex<T> a) {
    return rstsr_complex<T>(-a.re, -a.im);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_cconj(rstsr_complex<T> a) {
    return rstsr_complex<T>(a.re, -a.im);
}

template <typename T>
__device__ T rstsr_cabs(rstsr_complex<T> a) {
    return rstsr_rsqrt(a.re * a.re + a.im * a.im);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_cmul(rstsr_complex<T> a, rstsr_complex<T> b) {
    return rstsr_complex<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_cdiv(rstsr_complex<T> a, rstsr_complex<T> b) {
    // Smith's algorithm (CuPy complex.cuh `operator/`): avoids overflow
    // when |b.re| or |b.im| dominates
    T ratio;
    T denom;
    if (rstsr_rabs(b.re) >= rstsr_rabs(b.im)) {
        ratio = b.im / b.re;
        denom = b.re + b.im * ratio;
        return rstsr_complex<T>((a.re + a.im * ratio) / denom, (a.im - a.re * ratio) / denom);
    } else {
        ratio = b.re / b.im;
        denom = b.im + b.re * ratio;
        return rstsr_complex<T>((a.re * ratio + a.im) / denom, (a.im * ratio - a.re) / denom);
    }
}

// --- remainder (num-complex semantics) --------------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_crem(rstsr_complex<T> x, rstsr_complex<T> m) {
    // Gaussian-integer remainder, matching num-complex `Rem`:
    // q = trunc(x/m) per component, r = x - m*q
    rstsr_complex<T> d = rstsr_cdiv(x, m);
    rstsr_complex<T> q = rstsr_complex<T>((T)(int)(d.re), (T)(long long)(d.im));
    return rstsr_csub(x, rstsr_cmul(m, q));
}

// --- exponential / logarithm / square root ---------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_cexp(rstsr_complex<T> a) {
    T e = rstsr_rexp(a.re);
    return rstsr_complex<T>(e * rstsr_rcos(a.im), e * rstsr_rsin(a.im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_clog(rstsr_complex<T> a) {
    return rstsr_complex<T>(rstsr_rlog(rstsr_cabs(a)), rstsr_ratan2(a.im, a.re));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_csqrt(rstsr_complex<T> z) {
    // principal branch: sqrt(r) * (cos(t/2), sin(t/2)) via half-angle
    // identities (CuPy complex.cuh csqrt)
    if (z.re == (T)0.0 && z.im == (T)0.0) {
        return rstsr_complex<T>((T)0.0, (T)0.0);
    }
    T mag = rstsr_cabs(z);
    T re, im;
    if (z.re >= (T)0.0) {
        re = rstsr_rsqrt((T)0.5 * (mag + z.re));
        im = z.im / ((T)2.0 * re);
    } else {
        im = rstsr_rcopysign(rstsr_rsqrt((T)0.5 * (mag - z.re)), z.im);
        re = z.im / ((T)2.0 * im);
    }
    return rstsr_complex<T>(re, im);
}

// --- trigonometry -----------------------------------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_csin(rstsr_complex<T> z) {
    return rstsr_complex<T>(rstsr_rsin(z.re) * rstsr_rcosh(z.im), rstsr_rcos(z.re) * rstsr_rsinh(z.im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_ccos(rstsr_complex<T> z) {
    return rstsr_complex<T>(rstsr_rcos(z.re) * rstsr_rcosh(z.im), -rstsr_rsin(z.re) * rstsr_rsinh(z.im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_ctan(rstsr_complex<T> z) {
    // tan = sin/cos via the exponential definitions (CuPy complex.cuh)
    T a = (T)2.0 * z.re;
    T b = (T)2.0 * z.im;
    T denom = rstsr_rcos(a) + rstsr_rcosh(b);
    return rstsr_complex<T>(rstsr_rsin(a) / denom, rstsr_rsinh(b) / denom);
}

// --- inverse trigonometry / hyperbolic (C99 Annex G) -------------------------
//
// FreeBSD msun algorithms (Stephen Montgomery-Smith), the same source as
// NumPy's npy_math_complex.c, rstsr's c99_complex.rs, and cupy's
// catrig.h/catrigf.h. Component-pair form. Python translation source:
// rstsr-dtype-traits/src/c99_complex.rs.

// NaN constants local to this header (it may splice before MATH_HELPER's
// rstsr_nan helpers; NVRTC has no <math.h> on the include path).
template <typename T>
__device__ T rstsr_cnan_t();
template <>
__device__ float rstsr_cnan_t<float>() { return __int_as_float(0x7fffffff); }
template <>
__device__ double rstsr_cnan_t<double>() { return __longlong_as_double(0x7fffffffffffffffLL); }

template <typename T>
__device__ T rstsr_f_hypot_minus(T a, T b, T hab) {
    if (b < (T)0.0) return (hab - b) / (T)2.0;
    if (b == (T)0.0) return a / (T)2.0;
    return a * a / (hab + b) / (T)2.0;
}

// outputs of the shared casinh/cacos core
template <typename T>
struct rstsr_hard_work {
    T rx;
    int b_usable;
    T b;
    T sa2my2;
    T ny;
};

// width-class constants (msun thresholds differ between float and double)
template <typename T>
struct rstsr_cw;
template <>
struct rstsr_cw<float> {
    static constexpr float eps = 1.1920929e-7f;
    static constexpr float ln2 = 0.6931472f;
    static constexpr float pio2 = 1.5707964f;
    static constexpr float sqrt_min = 1.0842021e-19f; // sqrt(FLT_MIN)
    static constexpr float t_max = 3.4028235e38f;
    static constexpr float e_base = 2.7182818f;
};
template <>
struct rstsr_cw<double> {
    static constexpr double eps = 2.220446049250313e-16;
    static constexpr double ln2 = 0.6931471805599453;
    static constexpr double pio2 = 1.5707963267948966;
    static constexpr double sqrt_min = 1.4916681462400413e-154; // sqrt(DBL_MIN)
    static constexpr double t_max = 1.7976931348623157e308;
    static constexpr double e_base = 2.718281828459045;
};

template <typename T>
__device__ rstsr_hard_work<T> rstsr_do_hard_work(T x, T y) {
    const T two = (T)2.0, four = (T)4.0;
    const T eps = rstsr_cw<T>::eps;
    const T a_cross = (T)10.0, b_cross = (T)0.6417;
    const T sqrt_min = rstsr_cw<T>::sqrt_min;
    const T four_sqrt_min = four * sqrt_min;
    rstsr_hard_work<T> o;
    T r = rstsr_rhypot(x, y + (T)1.0);
    T s = rstsr_rhypot(x, y - (T)1.0);
    T a = (r + s) / two;
    if (a < (T)1.0) a = (T)1.0;
    o.rx = (T)0.0;
    if (a < a_cross) {
        if (y == (T)1.0 && x < eps * eps / (T)128.0) {
            o.rx = rstsr_rsqrt(x);
        } else if (x >= eps * rstsr_rabs(y - (T)1.0)) {
            T am1 = rstsr_f_hypot_minus(x, (T)1.0 + y, r) + rstsr_f_hypot_minus(x, (T)1.0 - y, s);
            o.rx = rstsr_rlog1p(am1 + rstsr_rsqrt(am1 * (a + (T)1.0)));
        } else if (y < (T)1.0) {
            o.rx = x / rstsr_rsqrt(((T)1.0 - y) * ((T)1.0 + y));
        } else {
            o.rx = rstsr_rlog1p((y - (T)1.0) + rstsr_rsqrt((y - (T)1.0) * (y + (T)1.0)));
        }
    } else {
        o.rx = rstsr_rlog(a + rstsr_rsqrt(a * a - (T)1.0));
    }
    o.ny = y;
    if (y < four_sqrt_min) {
        T scale = two / eps;
        o.b_usable = 0; o.b = (T)0.0; o.sa2my2 = a * scale; o.ny = y * scale;
        return o;
    }
    T b = y / a;
    o.b_usable = 1; o.b = b; o.sa2my2 = (T)0.0;
    if (b > b_cross) {
        o.b_usable = 0;
        if (y == (T)1.0 && x < eps / (T)128.0) {
            o.sa2my2 = rstsr_rsqrt(x) * rstsr_rsqrt((a + y) / two);
        } else if (x >= eps * rstsr_rabs(y - (T)1.0)) {
            T amy = rstsr_f_hypot_minus(x, y + (T)1.0, r) + rstsr_f_hypot_minus(x, y - (T)1.0, s);
            o.sa2my2 = rstsr_rsqrt(amy * (a + y));
        } else if (y > (T)1.0) {
            T scale = four / eps / eps;
            o.sa2my2 = x * scale * y / rstsr_rsqrt((y + (T)1.0) * (y - (T)1.0));
            o.ny = y * scale;
        } else {
            o.sa2my2 = rstsr_rsqrt(((T)1.0 - y) * ((T)1.0 + y));
        }
    }
    return o;
}

template <typename T>
__device__ void rstsr_clog_large(T x, T y, T* re, T* im) {
    T ax = rstsr_rabs(x), ay = rstsr_rabs(y);
    if (ax < ay) { T t = ax; ax = ay; ay = t; }
    const T sqrt_min = rstsr_cw<T>::sqrt_min;
    const T t_max = rstsr_cw<T>::t_max;
    const T e_base = rstsr_cw<T>::e_base;
    T rr;
    if (ax > t_max / (T)2.0) {
        rr = rstsr_rlog(rstsr_rhypot(x / e_base, y / e_base)) + (T)1.0;
    } else if (ax > (T)1.0 / sqrt_min || ay < sqrt_min) {
        rr = rstsr_rlog(rstsr_rhypot(x, y));
    } else {
        rr = rstsr_rlog(ax * ax + ay * ay) / (T)2.0;
    }
    *re = rr; *im = rstsr_ratan2(y, x);
}

template <typename T>
__device__ T rstsr_recp_real(T x, T y) {
    if (isinf(x)) return (T)1.0 / x;
    if (isinf(y)) return x / y / y;
    return x / (x * x + y * y);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_casinh(rstsr_complex<T> z) {
    T x = z.re, y = z.im;
    const T eps = rstsr_cw<T>::eps;
    const T recips = (T)1.0 / eps;
    const T s6e4 = rstsr_rsqrt((T)6.0 * eps) / (T)4.0;
    const T LN2 = rstsr_cw<T>::ln2;
    T ax = rstsr_rabs(x), ay = rstsr_rabs(y);
    if (isnan(x) || isnan(y)) {
        if (isinf(x)) return rstsr_complex<T>(x, y + y);
        if (isinf(y)) return rstsr_complex<T>(y, x + x);
        if (y == (T)0.0) return rstsr_complex<T>(x + x, y);
        return rstsr_complex<T>(rstsr_cnan_t<T>(), rstsr_cnan_t<T>());
    }
    if (ax > recips || ay > recips) {
        T wx, wy;
        if (signbit(x)) { rstsr_clog_large(-x, -y, &wx, &wy); } else { rstsr_clog_large(x, y, &wx, &wy); }
        return rstsr_complex<T>(rstsr_rcopysign(wx + LN2, x), rstsr_rcopysign(wy, y));
    }
    if (x == (T)0.0 && y == (T)0.0) return rstsr_complex<T>(x, y);
    if (ax < s6e4 && ay < s6e4) return rstsr_complex<T>(x, y);
    rstsr_hard_work<T> hw = rstsr_do_hard_work(ax, ay);
    T ry = hw.b_usable ? rstsr_rasin(hw.b) : rstsr_ratan2(hw.ny, hw.sa2my2);
    return rstsr_complex<T>(rstsr_rcopysign(hw.rx, x), rstsr_rcopysign(ry, y));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_casin(rstsr_complex<T> z) {
    rstsr_complex<T> w = rstsr_casinh(rstsr_complex<T>(z.im, z.re));
    return rstsr_complex<T>(w.im, w.re);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_cacos(rstsr_complex<T> z) {
    T x = z.re, y = z.im;
    const T eps = rstsr_cw<T>::eps;
    const T recips = (T)1.0 / eps;
    const T s6e4 = rstsr_rsqrt((T)6.0 * eps) / (T)4.0;
    const T LN2 = rstsr_cw<T>::ln2;
    const T pio2 = rstsr_cw<T>::pio2;
    int sx = signbit(x), sy = signbit(y);
    T ax = rstsr_rabs(x), ay = rstsr_rabs(y);
    if (isnan(x) || isnan(y)) {
        if (isinf(x)) return rstsr_complex<T>(y + y, -(T)1.0 / (T)0.0);
        if (isinf(y)) return rstsr_complex<T>(x + x, -y);
        if (x == (T)0.0) return rstsr_complex<T>(pio2, y + y);
        return rstsr_complex<T>(rstsr_cnan_t<T>(), rstsr_cnan_t<T>());
    }
    if (ax > recips || ay > recips) {
        T wx, wy;
        rstsr_clog_large(x, y, &wx, &wy);
        T rx = rstsr_rabs(wy);
        T ry = sy ? (wx + LN2) : -(wx + LN2);
        return rstsr_complex<T>(rx, ry);
    }
    if (x == (T)1.0 && y == (T)0.0) return rstsr_complex<T>((T)0.0, -y);
    if (ax < s6e4 && ay < s6e4) return rstsr_complex<T>(pio2 - x, -y);
    rstsr_hard_work<T> hw = rstsr_do_hard_work(ay, ax);
    T rx = hw.b_usable ? (sx ? rstsr_racos(-hw.b) : rstsr_racos(hw.b))
                       : (sx ? rstsr_ratan2(hw.sa2my2, -hw.ny) : rstsr_ratan2(hw.sa2my2, hw.ny));
    T ry = sy ? hw.rx : -hw.rx;
    return rstsr_complex<T>(rx, ry);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_cacosh(rstsr_complex<T> z) {
    // cacosh from cacos (C99; same post-processing as msun ccosh)
    rstsr_complex<T> w = rstsr_cacos(z);
    if (isnan(w.re) && isnan(w.im)) return rstsr_complex<T>(w.im, w.re);
    if (isnan(w.re)) return rstsr_complex<T>(rstsr_rabs(w.im), w.re);
    if (isnan(w.im)) return rstsr_complex<T>(w.im, w.re);
    return rstsr_complex<T>(rstsr_rabs(w.im), rstsr_rcopysign(w.re, z.im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_catanh(rstsr_complex<T> z) {
    T x = z.re, y = z.im;
    const T eps = rstsr_cw<T>::eps;
    const T recips = (T)1.0 / eps;
    const T s3e2 = rstsr_rsqrt((T)3.0 * eps) / (T)2.0;
    const T LN2 = rstsr_cw<T>::ln2;
    const T pio2 = rstsr_cw<T>::pio2;
    T ax = rstsr_rabs(x), ay = rstsr_rabs(y);
    if (y == (T)0.0 && ax <= (T)1.0) return rstsr_complex<T>(rstsr_ratanh(x), y);
    if (x == (T)0.0) return rstsr_complex<T>(x, rstsr_ratan(y));
    if (isnan(x) || isnan(y)) {
        if (isinf(x)) return rstsr_complex<T>(rstsr_rcopysign((T)0.0, x), y + y);
        if (isinf(y)) return rstsr_complex<T>(rstsr_rcopysign((T)0.0, x), rstsr_rcopysign(pio2, y));
        return rstsr_complex<T>(rstsr_cnan_t<T>(), rstsr_cnan_t<T>());
    }
    if (ax > recips || ay > recips) return rstsr_complex<T>(rstsr_recp_real(x, y), rstsr_rcopysign(pio2, y));
    if (ax < s3e2 && ay < s3e2) return rstsr_complex<T>(x, y);
    T rx = (ax == (T)1.0 && ay < eps) ? (LN2 - rstsr_rlog(ay)) / (T)2.0
                                      : rstsr_rlog1p((T)4.0 * ax / ((ax - (T)1.0) * (ax - (T)1.0) + ay * ay)) / (T)4.0;
    T ry;
    if (ax == (T)1.0) ry = rstsr_ratan2((T)2.0, -ay) / (T)2.0;
    else if (ay < eps) ry = rstsr_ratan2((T)2.0 * ay, ((T)1.0 - ax) * ((T)1.0 + ax)) / (T)2.0;
    else ry = rstsr_ratan2((T)2.0 * ay, ((T)1.0 - ax) * ((T)1.0 + ax) - ay * ay) / (T)2.0;
    return rstsr_complex<T>(rstsr_rcopysign(rx, x), rstsr_rcopysign(ry, y));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_catan(rstsr_complex<T> z) {
    // catan(z) = -i * catanh(i * z) (C99; CPU c99_complex uses the same
    // rotation through the shared catanh core)
    rstsr_complex<T> w = rstsr_catanh(rstsr_complex<T>(-z.im, z.re));
    return rstsr_complex<T>(w.im, -w.re);
}

// --- hyperbolic --------------------------------------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_csinh(rstsr_complex<T> z) {
    // sinh = (e^z - e^-z)/2 (CuPy csinh)
    rstsr_complex<T> e = rstsr_cexp(z);
    rstsr_complex<T> emd = rstsr_cdiv(rstsr_complex<T>((T)1.0, (T)0.0), e);
    return rstsr_complex<T>((T)0.5 * (e.re - emd.re), (T)0.5 * (e.im - emd.im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_ccosh(rstsr_complex<T> z) {
    rstsr_complex<T> e = rstsr_cexp(z);
    rstsr_complex<T> emd = rstsr_cdiv(rstsr_complex<T>((T)1.0, (T)0.0), e);
    return rstsr_complex<T>((T)0.5 * (e.re + emd.re), (T)0.5 * (e.im + emd.im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_ctanh(rstsr_complex<T> z) {
    // tanh = sinh/cosh (CuPy ctanh; finite-z branch)
    rstsr_complex<T> s = rstsr_csinh(z);
    rstsr_complex<T> c = rstsr_ccosh(z);
    return rstsr_cdiv(s, c);
}

// --- pow / sign (z/|z|) ------------------------------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_cpow(rstsr_complex<T> x, rstsr_complex<T> y) {
    // exp(y * log x) with the 0-pow-0 special case (CuPy cpow / std::pow)
    if (x.re == (T)0.0 && x.im == (T)0.0) {
        return rstsr_complex<T>((T)0.0, (T)0.0);
    }
    return rstsr_cexp(rstsr_cmul(y, rstsr_clog(x)));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_csign(rstsr_complex<T> z) {
    // z/|z| with 0 -> 0 (num-complex `sign` semantics); NaN -> NaN
    if (isnan(z.re) || isnan(z.im)) return z;
    if (z.re == (T)0.0 && z.im == (T)0.0) return z;
    return rstsr_cdiv(z, rstsr_complex<T>(rstsr_cabs(z), (T)0.0));
}
