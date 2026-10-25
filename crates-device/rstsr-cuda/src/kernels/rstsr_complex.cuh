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
__device__ float rstsr_rfmod(float x, float y) { return fmodf(x, y); }
__device__ double rstsr_rfmod(double x, double y) { return fmod(x, y); }

// NaN/inf constants local to this header (NVRTC has no <math.h> on the
// include path, and the header may splice before MATH_HELPER's helpers).
template <typename T>
__device__ T rstsr_cnan_t();
template <>
__device__ float rstsr_cnan_t<float>() { return __int_as_float(0x7fffffff); }
template <>
__device__ double rstsr_cnan_t<double>() { return __longlong_as_double(0x7fffffffffffffffLL); }
template <typename T>
__device__ T rstsr_cinf_t();
template <>
__device__ float rstsr_cinf_t<float>() { return __int_as_float(0x7f800000); }
template <>
__device__ double rstsr_cinf_t<double>() { return __longlong_as_double(0x7ff0000000000000LL); }

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
    static constexpr float ctanh_huge = 11.0f; // |x| scale for ctanh's exp form
};
template <>
struct rstsr_cw<double> {
    static constexpr double eps = 2.220446049250313e-16;
    static constexpr double ln2 = 0.6931471805599453;
    static constexpr double pio2 = 1.5707963267948966;
    static constexpr double sqrt_min = 1.4916681462400413e-154; // sqrt(DBL_MIN)
    static constexpr double t_max = 1.7976931348623157e308;
    static constexpr double e_base = 2.718281828459045;
    static constexpr double ctanh_huge = 22.0;
};

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
    // CPU reference: num-complex `norm()` = hypot (overflow-safe)
    return rstsr_rhypot(a.re, a.im);
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
    // q = trunc(x/m) per component (`re - re % 1`, not an int cast),
    // r = x - m*q
    rstsr_complex<T> d = rstsr_cdiv(x, m);
    rstsr_complex<T> q = rstsr_complex<T>(d.re - rstsr_rfmod(d.re, (T)1.0), d.im - rstsr_rfmod(d.im, (T)1.0));
    return rstsr_csub(x, rstsr_cmul(m, q));
}

// --- exponential / logarithm / square root ---------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_cexp(rstsr_complex<T> a) {
    // num-complex `exp` corner cases (the CPU reference)
    T re = a.re, im = a.im;
    if (isinf(re)) {
        if (re < (T)0.0) {
            if (!isfinite(im)) return rstsr_complex<T>((T)0.0, (T)0.0);
        } else if (im == (T)0.0 || !isfinite(im)) {
            if (isinf(im)) im = im - im; // NaN
            return rstsr_complex<T>(re, im);
        }
    } else if (isnan(re) && im == (T)0.0) {
        return a;
    }
    T e = rstsr_rexp(re);
    return rstsr_complex<T>(e * rstsr_rcos(im), e * rstsr_rsin(im));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_clog(rstsr_complex<T> a) {
    return rstsr_complex<T>(rstsr_rlog(rstsr_cabs(a)), rstsr_ratan2(a.im, a.re));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_csqrt(rstsr_complex<T> z) {
    // C99 / msun csqrt (the CPU reference in rstsr-dtype-traits c99_complex.rs)
    T a = z.re, b = z.im;
    const T zero = (T)0.0, two = (T)2.0;
    if (a == zero && b == zero) return rstsr_complex<T>(zero, b);
    if (isinf(b)) return rstsr_complex<T>(rstsr_cinf_t<T>(), b);
    if (isnan(a)) {
        T n = (b - b) / (b - b);
        return rstsr_complex<T>(n, n);
    }
    if (isinf(a)) {
        T n = b - b;
        if (signbit(a)) return rstsr_complex<T>(rstsr_rabs(n), rstsr_rcopysign(a, b));
        return rstsr_complex<T>(a, rstsr_rcopysign(n, b));
    }

    // scale away overflow in a*a / b*b (Algorithm 312, CACM 1967)
    const T thresh = rstsr_cw<T>::t_max / ((T)1.0 + (T)1.4142135623730950488016887242096981);
    bool scale = rstsr_rabs(a) >= thresh || rstsr_rabs(b) >= thresh;
    if (scale) {
        a *= (T)0.25;
        b *= (T)0.25;
    }
    T re, im;
    if (a >= zero) {
        T t = rstsr_rsqrt((a + rstsr_rhypot(a, b)) * (T)0.5);
        re = t;
        im = b / (two * t);
    } else {
        T t = rstsr_rsqrt((-a + rstsr_rhypot(a, b)) * (T)0.5);
        re = rstsr_rabs(b) / (two * t);
        im = rstsr_rcopysign(t, b);
    }
    if (scale) re *= two;
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

// --- inverse trigonometry / hyperbolic (C99 Annex G) -------------------------
//
// FreeBSD msun algorithms (Stephen Montgomery-Smith), the same source as
// NumPy's npy_math_complex.c, rstsr's c99_complex.rs, and cupy's
// catrig.h/catrigf.h. Component-pair form. Python translation source:
// rstsr-dtype-traits/src/c99_complex.rs.

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

// --- hyperbolic (C99 / msun, as in rstsr-dtype-traits c99_complex.rs) --------

template <typename T>
__device__ rstsr_complex<T> rstsr_ccosh(rstsr_complex<T> z) {
    T x = z.re, y = z.im;
    const T zero = (T)0.0, half = (T)0.5, big = (T)22.0;
    bool xf = isfinite(x), yf = isfinite(y);

    if (xf && yf) {
        if (y == zero) return rstsr_complex<T>(rstsr_rcosh(x), x * y);
        if (rstsr_rabs(x) < big) {
            return rstsr_complex<T>(rstsr_rcosh(x) * rstsr_rcos(y), rstsr_rsinh(x) * rstsr_rsin(y));
        }
        T h = rstsr_rexp(rstsr_rabs(x)) * half;
        return rstsr_complex<T>(h * rstsr_rcos(y), rstsr_rcopysign(h, x) * rstsr_rsin(y));
    }
    if (x == zero && !yf) {
        T n = y - y;
        return rstsr_complex<T>(n, rstsr_rcopysign(zero, x * n));
    }
    if (y == zero && !xf) return rstsr_complex<T>(x * x, rstsr_rcopysign(zero, x) * y);
    if (xf && !yf) {
        T n = y - y;
        return rstsr_complex<T>(n, x * n);
    }
    if (isinf(x)) {
        if (!yf) {
            T n = y - y;
            return rstsr_complex<T>(x * x, x * n);
        }
        return rstsr_complex<T>((x * x) * rstsr_rcos(y), x * rstsr_rsin(y));
    }
    T n = y - y;
    return rstsr_complex<T>((x * x) * n, (x + x) * n);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_csinh(rstsr_complex<T> z) {
    T x = z.re, y = z.im;
    const T zero = (T)0.0, half = (T)0.5, big = (T)22.0;
    bool xf = isfinite(x), yf = isfinite(y);

    if (xf && yf) {
        if (y == zero) return rstsr_complex<T>(rstsr_rsinh(x), y);
        if (rstsr_rabs(x) < big) {
            return rstsr_complex<T>(rstsr_rsinh(x) * rstsr_rcos(y), rstsr_rcosh(x) * rstsr_rsin(y));
        }
        T h = rstsr_rexp(rstsr_rabs(x)) * half;
        return rstsr_complex<T>(rstsr_rcopysign(h, x) * rstsr_rcos(y), h * rstsr_rsin(y));
    }
    if (x == zero && !yf) {
        T n = y - y;
        return rstsr_complex<T>(rstsr_rcopysign(zero, x * n), n);
    }
    if (y == zero && !xf) {
        if (isnan(x)) return rstsr_complex<T>(x, y);
        return rstsr_complex<T>(x, rstsr_rcopysign(zero, y));
    }
    if (xf && !yf) {
        T n = y - y;
        return rstsr_complex<T>(n, x * n);
    }
    if (!xf && !isnan(x)) {
        if (!yf) {
            T n = y - y;
            return rstsr_complex<T>(x * x, x * n);
        }
        return rstsr_complex<T>(x * rstsr_rcos(y), rstsr_cinf_t<T>() * rstsr_rsin(y));
    }
    T n = y - y;
    return rstsr_complex<T>((x * x) * n, (x + x) * n);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_ctanh(rstsr_complex<T> z) {
    // C99 / msun ctanh (Kahan's algorithm)
    T x = z.re, y = z.im;
    const T zero = (T)0.0, one = (T)1.0, four = (T)4.0;

    if (!isfinite(x)) {
        if (isnan(x)) return rstsr_complex<T>(x, (y == zero) ? y : x * y);
        return rstsr_complex<T>(rstsr_rcopysign(one, x), rstsr_rcopysign(zero, y));
    }
    if (!isfinite(y)) {
        if (x == zero) return rstsr_complex<T>(x, y - y);
        T n = y - y;
        return rstsr_complex<T>(n, n);
    }
    if (rstsr_rabs(x) >= rstsr_cw<T>::ctanh_huge) {
        T e = rstsr_rexp(-rstsr_rabs(x));
        return rstsr_complex<T>(rstsr_rcopysign(one, x), four * rstsr_rsin(y) * rstsr_rcos(y) * e * e);
    }
    T t = rstsr_rtan(y);
    T beta = one + t * t;
    T s = rstsr_rsinh(x);
    T rho = rstsr_rsqrt(one + s * s);
    T denom = one + beta * s * s;
    return rstsr_complex<T>((beta * rho * s) / denom, t / denom);
}

template <typename T>
__device__ rstsr_complex<T> rstsr_ctan(rstsr_complex<T> z) {
    // ctan(z) = -i ctanh(i z)
    rstsr_complex<T> w = rstsr_ctanh(rstsr_complex<T>(-z.im, z.re));
    return rstsr_complex<T>(w.im, -w.re);
}

// --- expm1 / log1p (cancellation-free forms mirrored from the CPU) -----------

template <typename T>
__device__ rstsr_complex<T> rstsr_cexpm1(rstsr_complex<T> z) {
    const T zero = (T)0.0, one = (T)1.0, half = (T)0.5, two = (T)2.0;
    if (z.re == zero && z.im == zero) return rstsr_complex<T>(zero, zero);
    if (isfinite(z.re) && isfinite(z.im) && rstsr_cabs(z) < half) {
        // exp(z) - 1 = 2 exp(z/2) sinh(z/2)
        rstsr_complex<T> h = rstsr_complex<T>(z.re * half, z.im * half);
        rstsr_complex<T> p = rstsr_cmul(rstsr_cexp(h), rstsr_csinh(h));
        return rstsr_complex<T>(two * p.re, two * p.im);
    }
    return rstsr_csub(rstsr_cexp(z), rstsr_complex<T>(one, zero));
}

template <typename T>
__device__ rstsr_complex<T> rstsr_clog1p(rstsr_complex<T> z) {
    const T zero = (T)0.0, one = (T)1.0;
    rstsr_complex<T> u = rstsr_cadd(rstsr_complex<T>(one, zero), z);
    if (!isfinite(u.re) || !isfinite(u.im) || (u.re == zero && u.im == zero)) {
        return rstsr_clog(u);
    }
    // u = fl(1 + z); rho = u - (1 + z) is exact (Sterbenz)
    rstsr_complex<T> rho = rstsr_csub(rstsr_csub(u, rstsr_complex<T>(one, zero)), z);
    return rstsr_csub(rstsr_clog(u), rstsr_cdiv(rho, u));
}

// --- pow / sign (z/|z|) ------------------------------------------------------

template <typename T>
__device__ rstsr_complex<T> rstsr_cpow(rstsr_complex<T> x, rstsr_complex<T> y) {
    // num-complex `powc`: y == 0 -> 1, else exp(y log x). Zero base then
    // follows the log limit (0^positive -> 0, 0^negative -> NaN).
    if (y.re == (T)0.0 && y.im == (T)0.0) {
        return rstsr_complex<T>((T)1.0, (T)0.0);
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

// --- C++ operators -----------------------------------------------------------
// The arithmetic families, square, ext_neg and compound assigns emit plain
// C++ operators; they route through the Smith/naive helpers above so the
// nvcc/C++ overload resolution sees them.

template <typename T>
__device__ rstsr_complex<T> operator+(rstsr_complex<T> a, rstsr_complex<T> b) { return rstsr_cadd(a, b); }
template <typename T>
__device__ rstsr_complex<T> operator-(rstsr_complex<T> a, rstsr_complex<T> b) { return rstsr_csub(a, b); }
template <typename T>
__device__ rstsr_complex<T> operator*(rstsr_complex<T> a, rstsr_complex<T> b) { return rstsr_cmul(a, b); }
template <typename T>
__device__ rstsr_complex<T> operator/(rstsr_complex<T> a, rstsr_complex<T> b) { return rstsr_cdiv(a, b); }
template <typename T>
__device__ rstsr_complex<T> operator-(rstsr_complex<T> a) { return rstsr_cneg(a); }
template <typename T>
__device__ bool operator==(rstsr_complex<T> a, rstsr_complex<T> b) { return a.re == b.re && a.im == b.im; }
template <typename T>
__device__ bool operator!=(rstsr_complex<T> a, rstsr_complex<T> b) { return !(a == b); }

// mixed complex/real scalar forms (compound assigns with a real RHS)
template <typename T>
__device__ rstsr_complex<T> operator+(rstsr_complex<T> a, T b) { return rstsr_cadd(a, rstsr_complex<T>(b, (T)0.0)); }
template <typename T>
__device__ rstsr_complex<T> operator+(T a, rstsr_complex<T> b) { return rstsr_cadd(rstsr_complex<T>(a, (T)0.0), b); }
template <typename T>
__device__ rstsr_complex<T> operator-(rstsr_complex<T> a, T b) { return rstsr_csub(a, rstsr_complex<T>(b, (T)0.0)); }
template <typename T>
__device__ rstsr_complex<T> operator-(T a, rstsr_complex<T> b) { return rstsr_csub(rstsr_complex<T>(a, (T)0.0), b); }
template <typename T>
__device__ rstsr_complex<T> operator*(rstsr_complex<T> a, T b) { return rstsr_cmul(a, rstsr_complex<T>(b, (T)0.0)); }
template <typename T>
__device__ rstsr_complex<T> operator*(T a, rstsr_complex<T> b) { return rstsr_cmul(rstsr_complex<T>(a, (T)0.0), b); }
template <typename T>
__device__ rstsr_complex<T> operator/(rstsr_complex<T> a, T b) { return rstsr_cdiv(a, rstsr_complex<T>(b, (T)0.0)); }
template <typename T>
__device__ rstsr_complex<T> operator/(T a, rstsr_complex<T> b) { return rstsr_cdiv(rstsr_complex<T>(a, (T)0.0), b); }

template <typename T>
__device__ rstsr_complex<T>& operator+=(rstsr_complex<T>& a, rstsr_complex<T> b) { a = rstsr_cadd(a, b); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator-=(rstsr_complex<T>& a, rstsr_complex<T> b) { a = rstsr_csub(a, b); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator*=(rstsr_complex<T>& a, rstsr_complex<T> b) { a = rstsr_cmul(a, b); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator/=(rstsr_complex<T>& a, rstsr_complex<T> b) { a = rstsr_cdiv(a, b); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator+=(rstsr_complex<T>& a, T b) { a = rstsr_cadd(a, rstsr_complex<T>(b, (T)0.0)); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator-=(rstsr_complex<T>& a, T b) { a = rstsr_csub(a, rstsr_complex<T>(b, (T)0.0)); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator*=(rstsr_complex<T>& a, T b) { a = rstsr_cmul(a, rstsr_complex<T>(b, (T)0.0)); return a; }
template <typename T>
__device__ rstsr_complex<T>& operator/=(rstsr_complex<T>& a, T b) { a = rstsr_cdiv(a, rstsr_complex<T>(b, (T)0.0)); return a; }
