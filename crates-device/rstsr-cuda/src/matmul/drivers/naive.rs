//! The naive driver: a universal elementwise GEMM kernel.
//!
//! One thread per output element, a loop over the contraction, full layout addressing (signed and
//! zero strides, per-axis batch strides). It serves every type triple of [`NaiveScalar`] and every
//! layout the rule table admits, and it is the last driver of the chain: whatever the accelerated
//! drivers decline lands here. It is also the on-device oracle for them.
//!
//! The arithmetic is deliberately simple. Each operand is cast to the accumulator type `@TACC@`
//! (`f32` for a half output, the output type otherwise), the `k` products are added in `@TACC@`,
//! and the sum is rounded to the output type once. A mixed triple is therefore a "cast each
//! operand to the output type" promotion, with no materialized copy. The result is exact for
//! integer triples and within one rounding of the CPU serial kernel for the real ones; it is
//! DRAM-bound, which is why it runs last.
//!
//! Nothing is declined here. An empty output returns before any driver is asked, and the two
//! shapes that have no GEMM geometry at all (a batch rank over [`RANK_MAX`]) are refused by name
//! before the chain.

use super::super::geom::CallerGeom;
use super::{Problem, Verdict};
use crate::dtype::{is_complex_cname, is_half_cname, CudaDType};
use crate::kernel::{order_input, order_output, FlatLaunch};
use crate::kernels::elementwise::planned_function;
use crate::kernels::exprs::{compute_type, load};
use crate::prelude_dev::*;

/// Largest batch rank (number of axes in front of the two matrix axes) the kernel is generated
/// for; the batch shape and stride arrays of all three operands are padded to this rank.
const RANK_MAX: usize = 16;

/// Types the naive kernel is instantiated for: everything with CUDA C arithmetic and a zero.
pub(crate) trait NaiveScalar: CudaDType + num::Zero {
    /// C type the products are accumulated in: `float` for a half output, else the type itself.
    fn acc() -> &'static str {
        if is_half_cname(Self::CNAME) {
            "float"
        } else {
            Self::CNAME
        }
    }
}

impl<T: CudaDType + num::Zero> NaiveScalar for T {}

/// C expression of the zero value of C type `t`.
fn zero_expr(t: &str) -> String {
    if is_complex_cname(t) {
        format!("{t}(0.0f, 0.0f)")
    } else if is_half_cname(t) {
        // the half conversion helper: `( __half )0` is not a valid half zero
        crate::kernels::exprs::store(t, "0.0f")
    } else {
        format!("({t})0")
    }
}

/// C expression of `x` (of C type `from`) cast to C type `to`.
///
/// `x` is a real scalar or a value already of type `to`. Real to complex is `to(x, 0)`; the
/// reverse never occurs (no promotion produces a real output from a complex operand).
fn cast(from: &str, to: &str, x: &str) -> String {
    if from == to {
        return x.to_string();
    }
    if is_half_cname(to) {
        crate::kernels::exprs::store(to, x)
    } else if is_complex_cname(to) {
        format!("{to}({x}, 0.0f)")
    } else {
        format!("({to})({x})")
    }
}

/// The C expression of one product term, in the accumulator type.
fn term(ta: &str, tb: &str, tacc: &str, a: &str, b: &str) -> String {
    let (lta, ltb) = (compute_type(ta), compute_type(tb));
    let av = cast(&lta, tacc, &load(ta, a));
    let bv = cast(&ltb, tacc, &load(tb, b));
    format!("{av} * {bv}")
}

/// The C expression of the scaled sum, and of the epilogue `v + beta * *p`.
fn epilogue(tc: &str) -> (String, String) {
    use crate::kernels::exprs::{load, store};
    if is_half_cname(tc) {
        // `alpha`, `beta` and `v` are half, `acc` is float: compute the epilogue in float, store once
        (
            store(tc, &format!("{} * acc", load(tc, "alpha"))),
            store(tc, &format!("{} + {} * {}", load(tc, "v"), load(tc, "beta"), load(tc, "*p"))),
        )
    } else {
        ("alpha * acc".to_string(), "v + beta * (*p)".to_string())
    }
}

const NAIVE_ENTRY: &str = r#"
typedef struct { int n[@R@]; } rstsr_bsh;
typedef struct { rstsr_isz s[@R@]; } rstsr_bst;

extern "C" __global__ void rstsr_matmul_naive(
    @TC@* c, const @TA@* a, const @TB@* b,
    rstsr_isz scm, rstsr_isz scn, rstsr_isz sam, rstsr_isz sak, rstsr_isz sbk, rstsr_isz sbn,
    rstsr_usz total, rstsr_isz M, rstsr_isz N, rstsr_isz K,
    rstsr_bsh bsh_a, rstsr_bst bst_a,
    rstsr_bsh bsh_b, rstsr_bst bst_b,
    rstsr_bsh bsh_c, rstsr_bst bst_c,
    int read_c, @TC@ alpha, @TC@ beta)
{
    rstsr_usz t = (rstsr_usz)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    rstsr_usz r = t / (rstsr_usz)N;
    rstsr_isz j = (rstsr_isz)(t - r * (rstsr_usz)N);
    rstsr_usz bi = r / (rstsr_usz)M;
    rstsr_isz i = (rstsr_isz)(r - bi * (rstsr_usz)M);

    // batch offset of this slice, decomposed against each operand's own batch shape
    rstsr_isz oa = 0, ob = 0, oc = 0;
    rstsr_usz ia = bi, ib = bi, ic = bi;
    #pragma unroll
    for (int d = @R@ - 1; d >= 0; --d) {
        oc += (rstsr_isz)(ic % (rstsr_usz)bsh_c.n[d]) * bst_c.s[d]; ic /= (rstsr_usz)bsh_c.n[d];
        oa += (rstsr_isz)(ia % (rstsr_usz)bsh_a.n[d]) * bst_a.s[d]; ia /= (rstsr_usz)bsh_a.n[d];
        ob += (rstsr_isz)(ib % (rstsr_usz)bsh_b.n[d]) * bst_b.s[d]; ib /= (rstsr_usz)bsh_b.n[d];
    }

    const @TA@* ap = a + oa;
    const @TB@* bp = b + ob;
    @TACC@ acc = @ZERO@;
    for (rstsr_isz kk = 0; kk < K; ++kk) {
        acc += @TERM@;
    }
    @TC@* p = c + oc + i * scm + j * scn;
    const @TC@ v = @VALUE@;
    *p = read_c ? (@ADD_C@) : v;
}
"#;

/// Batch shape and stride arrays of one operand, padded with leading unit extents.
fn batch_arrays(part: &crate::matmul::geom::BatchPart, rank: usize) -> (Vec<i32>, Vec<isize>) {
    let pad = rank - part.shape.len();
    let shape = part.shape.iter().map(|&n| n as i32);
    let stride = part.stride.iter().copied();
    let ones = std::iter::repeat_n(1i32, pad);
    let zeros = std::iter::repeat_n(0isize, pad);
    (ones.chain(shape).collect(), zeros.chain(stride).collect())
}

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn bytes_isize(v: &[isize]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// Run the problem on the naive kernel. Errors are launch failures.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>) -> Result<Verdict>
where
    TA: NaiveScalar,
    TB: NaiveScalar,
    TC: NaiveScalar,
    TBuf: 'static,
{
    // The caller's roles, not the row-major ones: the kernel is templated on the caller's types.
    let CallerGeom { mnk, a, b, c, batch } = p.geom.caller();
    let (a_off, sam, sak) = a;
    let (b_off, sbk, sbn) = b;
    let (c_off, scm, scn) = c;
    let rank = batch.iter().map(|part| part.shape.len()).max().unwrap_or(0);
    if rank > RANK_MAX {
        rstsr_raise!(
            UnImplemented,
            "cuda matmul: the batch part has rank {rank}, over the limit of {RANK_MAX} of the naive kernel"
        )?
    }
    let rank = rank.max(1);
    let (sh_a, st_a) = batch_arrays(batch[0], rank);
    let (sh_b, st_b) = batch_arrays(batch[1], rank);
    let (sh_c, st_c) = batch_arrays(batch[2], rank);

    let ta = TA::CNAME;
    let tb = TB::CNAME;
    let tc = TC::CNAME;
    let tacc = TC::acc();
    let (value, add_c) = epilogue(tc);
    let subs: Vec<(&str, String)> = vec![
        ("@TA@", ta.to_string()),
        ("@TB@", tb.to_string()),
        ("@TC@", tc.to_string()),
        ("@TACC@", tacc.to_string()),
        ("@R@", rank.to_string()),
        ("@ZERO@", zero_expr(tacc)),
        ("@TERM@", term(ta, tb, tacc, "ap[i * sam + kk * sak]", "bp[kk * sbk + j * sbn]")),
        ("@VALUE@", value),
        ("@ADD_C@", add_c),
    ];
    let source = subs.iter().fold(NAIVE_ENTRY.to_string(), |s, (k, v)| s.replace(k, v));
    let mut includes: Vec<&'static str> = Vec::new();
    for inc in TA::INCLUDES.iter().chain(TB::INCLUDES).chain(TC::INCLUDES) {
        includes.push(inc);
    }
    let disc = format!("{ta}|{tb}|{tc}|{tacc}|R{rank}");
    let f = planned_function(p.device, "rstsr_matmul_naive", &includes, &disc, || source)?;

    let exec = p.device.stream();
    let (_ga, _gb, _gc) = (order_input(p.a, exec)?, order_input(p.b, exec)?, order_input(p.c, exec)?);
    let total = p.geom.n_batch as u64 * p.geom.m as u64 * p.geom.n as u64;
    let abytes = super::scalar_bytes(&p.alpha);
    let bbytes = super::scalar_bytes(&p.beta);
    let nb = core::mem::size_of::<TC>();

    FlatLaunch::new(exec, &f, total as usize)
        .arg_ptr(p.c, c_off)
        .arg_ptr(p.a, a_off)
        .arg_ptr(p.b, b_off)
        .arg_bytes(&scm.to_ne_bytes())
        .arg_bytes(&scn.to_ne_bytes())
        .arg_bytes(&sam.to_ne_bytes())
        .arg_bytes(&sak.to_ne_bytes())
        .arg_bytes(&sbk.to_ne_bytes())
        .arg_bytes(&sbn.to_ne_bytes())
        .arg_u64(total)
        .arg_bytes(&(mnk.0 as isize).to_ne_bytes())
        .arg_bytes(&(mnk.2 as isize).to_ne_bytes())
        .arg_bytes(&(mnk.1 as isize).to_ne_bytes())
        .arg_bytes(&bytes_i32(&sh_a))
        .arg_bytes(&bytes_isize(&st_a))
        .arg_bytes(&bytes_i32(&sh_b))
        .arg_bytes(&bytes_isize(&st_b))
        .arg_bytes(&bytes_i32(&sh_c))
        .arg_bytes(&bytes_isize(&st_c))
        .arg_i32(p.read_c as i32)
        .arg_bytes(&abytes[..nb])
        .arg_bytes(&bbytes[..nb])
        .launch()?;
    // the product is enqueued: order the output's allocation stream behind it
    let _gc = order_output(p.c, exec)?;

    Ok(Verdict::Ran)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accumulator is the output type, except for half outputs which accumulate in `f32`.
    #[test]
    fn accumulator_type() {
        assert_eq!(f32::acc(), "float");
        assert_eq!(f64::acc(), "double");
        assert_eq!(i32::acc(), "int");
        assert_eq!(num::Complex::<f64>::acc(), "rstsr_c64");
        assert_eq!(half::f16::acc(), "float");
        assert_eq!(half::bf16::acc(), "float");
    }

    /// The cast rule: identity, real to real, real to complex, and the half conversions.
    #[test]
    fn casts() {
        assert_eq!(cast("float", "float", "x"), "x");
        assert_eq!(cast("float", "double", "x"), "(double)(x)");
        assert_eq!(cast("double", "rstsr_c64", "x"), "rstsr_c64(x, 0.0f)");
        // a half operand is loaded as float first, so the cast that follows is the float one
        assert_eq!(cast(&compute_type("__half"), "double", "__half2float(x)"), "(double)(__half2float(x))");
        assert_eq!(term("__half", "float", "double", "a0", "b0"), "(double)(__half2float(a0)) * (double)(b0)");
        assert_eq!(cast("float", "__half", "x"), "__float2half(x)");
        assert_eq!(zero_expr("int"), "(int)0");
        assert_eq!(zero_expr("rstsr_c32"), "rstsr_c32(0.0f, 0.0f)");
        assert_eq!(zero_expr("__nv_bfloat16"), "__float2bfloat16(0.0f)");
    }

    /// The batch arrays are padded with leading unit extents, so a lower-rank operand decomposes
    /// its slice index against its own trailing axes.
    #[test]
    fn batch_arrays_pad_leading() {
        let part = crate::matmul::geom::BatchPart { shape: vec![2, 3], stride: vec![30, 10] };
        assert_eq!(batch_arrays(&part, 2), (vec![2, 3], vec![30, 10]));
        assert_eq!(batch_arrays(&part, 4), (vec![1, 1, 2, 3], vec![0, 0, 30, 10]));
        let none = crate::matmul::geom::BatchPart::default();
        assert_eq!(batch_arrays(&none, 2), (vec![1, 1], vec![0, 0]));
    }

    /// Every type the driver claims is covered by the trait.
    #[test]
    fn naive_scalar_coverage() {
        fn is_naive<T: NaiveScalar>() {}
        is_naive::<f32>();
        is_naive::<f64>();
        is_naive::<i8>();
        is_naive::<u64>();
        is_naive::<num::Complex<f32>>();
        is_naive::<num::Complex<f64>>();
        is_naive::<half::f16>();
        is_naive::<half::bf16>();
    }
}
