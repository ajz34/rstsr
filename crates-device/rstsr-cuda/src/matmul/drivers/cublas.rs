//! The classic cuBLAS driver: the typed `gemm` / `gemmStridedBatched` entries.
//!
//! The fast path for every layout it can express: on an A100 `dgemm` reaches the FP64 tensor-core
//! pipe (19.5 TFLOPS) where a SIMT tile tops out at 9.7, and on every other shape it is at least
//! as fast for a hundredth of the engineering. What it cannot express is derived in
//! [`super::super::cublas_args`] and handed to the next driver.
//!
//! The entries cover `f32`, `f64` and the complex pairs; half and mixed-storage triples come
//! with the `cublasGemmEx` and cuBLASLt drivers. The compute mode is a math mode of the handle here
//! (the typed entries have no compute-type argument), set inside the handle lock and restored
//! before it is released; [`with_gemm_call`] is the preparation the `cublasGemmEx` driver shares.
//!
//! # Thread safety
//!
//! A `cublasHandle_t` is not thread-safe, and this crate shares one per ordinal (see
//! [`CudaStreamObj`][crate::stream::CudaStreamObj]). [`launch`] therefore takes the handle's lock
//! itself. That is a requirement of cuBLAS, not a precaution: two host threads entering a `gemm` on
//! one handle at the same time is undefined.

use cudarc::cublas::sys;

use super::{scale, Buffers, CublasPlan, Problem, Refusal, Verdict};
use crate::dtype::CudaDType;
use crate::kernel::{order_input, order_output};
use crate::matmul::cublas_args;
use crate::matmul::mode::{ComputeType32F, ComputeType64F};
use crate::prelude_dev::*;
use crate::stream::CublasHandle;
use num::Complex;
use std::sync::Arc;

/// The types cuBLAS's typed `gemm` entries cover: the real and complex floats.
pub(crate) trait CublasScalar: CudaDType {
    /// # Safety
    /// `a`, `b`, `c` must be device pointers of at least the extents implied by
    /// `m, n, k, lda, ldb, ldc`, on the stream the handle is bound to.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: u64,
        lda: i32,
        b: u64,
        ldb: i32,
        beta: &Self,
        c: u64,
        ldc: i32,
    ) -> sys::cublasStatus_t;

    /// Same contract, plus `batch` matrices spaced `stride_*` elements apart.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_strided_batched(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: u64,
        lda: i32,
        stride_a: i64,
        b: u64,
        ldb: i32,
        stride_b: i64,
        beta: &Self,
        c: u64,
        ldc: i32,
        stride_c: i64,
        batch: i32,
    ) -> sys::cublasStatus_t;
}

macro_rules! impl_cublas_scalar {
    ($t:ty, $gemm:ident, $gemm_sb:ident) => {
        impl_cublas_scalar!($t, $gemm, $gemm_sb, $t);
    };
    ($t:ty, $gemm:ident, $gemm_sb:ident, $cx:path) => {
        impl CublasScalar for $t {
            unsafe fn gemm(
                handle: sys::cublasHandle_t,
                transa: sys::cublasOperation_t,
                transb: sys::cublasOperation_t,
                m: i32,
                n: i32,
                k: i32,
                alpha: &Self,
                a: u64,
                lda: i32,
                b: u64,
                ldb: i32,
                beta: &Self,
                c: u64,
                ldc: i32,
            ) -> sys::cublasStatus_t {
                unsafe {
                    sys::$gemm(
                        handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha as *const $t as *const $cx,
                        a as *const $t as *const $cx,
                        lda,
                        b as *const $t as *const $cx,
                        ldb,
                        beta as *const $t as *const $cx,
                        c as *mut $t as *mut $cx,
                        ldc,
                    )
                }
            }

            unsafe fn gemm_strided_batched(
                handle: sys::cublasHandle_t,
                transa: sys::cublasOperation_t,
                transb: sys::cublasOperation_t,
                m: i32,
                n: i32,
                k: i32,
                alpha: &Self,
                a: u64,
                lda: i32,
                stride_a: i64,
                b: u64,
                ldb: i32,
                stride_b: i64,
                beta: &Self,
                c: u64,
                ldc: i32,
                stride_c: i64,
                batch: i32,
            ) -> sys::cublasStatus_t {
                unsafe {
                    sys::$gemm_sb(
                        handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha as *const $t as *const $cx,
                        a as *const $t as *const $cx,
                        lda,
                        stride_a,
                        b as *const $t as *const $cx,
                        ldb,
                        stride_b,
                        beta as *const $t as *const $cx,
                        c as *mut $t as *mut $cx,
                        ldc,
                        stride_c,
                        batch,
                    )
                }
            }
        }
    };
}

impl_cublas_scalar!(f32, cublasSgemm_v2, cublasSgemmStridedBatched);
impl_cublas_scalar!(f64, cublasDgemm_v2, cublasDgemmStridedBatched);
impl_cublas_scalar!(Complex<f32>, cublasCgemm_v2, cublasCgemmStridedBatched, sys::cuFloatComplex);
impl_cublas_scalar!(Complex<f64>, cublasZgemm_v2, cublasZgemmStridedBatched, sys::cuDoubleComplex);

fn op(t: bool) -> sys::cublasOperation_t {
    if t { sys::cublasOperation_t::CUBLAS_OP_T } else { sys::cublasOperation_t::CUBLAS_OP_N }
}

/// A compute mode as the classic entries take it: a math mode of the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MathMode {
    /// `CUBLAS_PEDANTIC_MATH`: the effect of the matching `_PEDANTIC` compute type.
    Pedantic,
    /// `CUBLAS_TENSOR_OP_MATH` (deprecated): the effect of `CUBLAS_COMPUTE_32F_FAST_16F`.
    TensorOp,
    /// `CUBLAS_TF32_TENSOR_OP_MATH`.
    Tf32TensorOp,
    /// `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`.
    Fp64EmulatedFixedPoint,
}

impl MathMode {
    #[cfg(cublas_math_mode)]
    fn cublas(self) -> sys::cublasMath_t {
        use sys::cublasMath_t as M;
        match self {
            Self::Pedantic => M::CUBLAS_PEDANTIC_MATH,
            Self::TensorOp => M::CUBLAS_TENSOR_OP_MATH,
            Self::Tf32TensorOp => M::CUBLAS_TF32_TENSOR_OP_MATH,
            #[cfg(cublas_fixedpoint_math)]
            Self::Fp64EmulatedFixedPoint => M::CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH,
            #[cfg(not(cublas_fixedpoint_math))]
            Self::Fp64EmulatedFixedPoint => unreachable!("refused by `bound` before any launch"),
        }
    }
}

/// What the driver does with the handle for one compute mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ask {
    /// Run with the handle as it is (`Default`).
    Leave,
    /// Set this math mode for the call.
    Math(MathMode),
    /// Decline, for this reason.
    No(&'static str),
}

fn ask_32f(mode: ComputeType32F) -> Ask {
    match mode {
        ComputeType32F::Default => Ask::Leave,
        ComputeType32F::Pedantic => Ask::Math(MathMode::Pedantic),
        ComputeType32F::FastTF32 => Ask::Math(MathMode::Tf32TensorOp),
        ComputeType32F::Fast16F => Ask::Math(MathMode::TensorOp),
        ComputeType32F::Fast16BF => Ask::No("no math mode of the classic entries converts the inputs to bf16"),
    }
}

fn ask_64f(mode: ComputeType64F, mantissa_bits: u32) -> Result<Ask> {
    Ok(match mode {
        ComputeType64F::Default => Ask::Leave,
        ComputeType64F::Pedantic => Ask::Math(MathMode::Pedantic),
        ComputeType64F::EmulatedFixedPoint => {
            if mantissa_bits != 0 {
                rstsr_raise!(
                    UnImplemented,
                    "cuda matmul: fixedpoint_mantissa_bits = {mantissa_bits} cannot be applied: \
                     cudarc 0.19 binds no entry for the fixed-point mantissa bit count"
                )?
            }
            Ask::Math(MathMode::Fp64EmulatedFixedPoint)
        },
    })
}

/// A math mode this build's `cudarc` bindings do not carry is a refusal, never a silent `Default`.
fn bound(ask: Ask) -> Ask {
    match ask {
        Ask::Math(MathMode::Fp64EmulatedFixedPoint) if !cfg!(cublas_fixedpoint_math) => {
            Ask::No("this build's cudarc bindings have no fixed-point emulation math mode (CUDA 13.0 or newer needed)")
        },
        Ask::Math(_) if !cfg!(cublas_math_mode) => {
            Ask::No("this build's cudarc bindings have no cublasSetMathMode (CUDA 12.9 or newer needed)")
        },
        ask => ask,
    }
}

/// Set the handle's math mode and return the mode it had before.
///
/// # Safety
/// The handle must be live and its lock held.
#[cfg(cublas_math_mode)]
unsafe fn replace_math_mode(
    handle: sys::cublasHandle_t,
    mode: MathMode,
) -> core::result::Result<sys::cublasMath_t, sys::cublasStatus_t> {
    let mut previous = sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
    let status = unsafe { sys::cublasGetMathMode(handle, &mut previous) };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(status);
    }
    let status = unsafe { sys::cublasSetMathMode(handle, mode.cublas()) };
    if status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(previous) } else { Err(status) }
}

/// Issue the `gemm`. Returns the cuBLAS status; the caller hands the problem to the next driver on
/// any non-success, because a returned status is a launch-time validation failure and nothing has
/// been enqueued. The only [`Err`] is a handle that could not be put back after the product was
/// enqueued.
///
/// The pointers are named after the **geom** operands; the slot swap happens here, once, and is the
/// same swap `cublas_args::args` assumes.
///
/// # The strides are element counts
///
/// `strideA/B/C` are passed straight through from `CublasArgs`, which holds them in elements
/// because that is what cuBLAS's strided-batched API takes — "the offset in number of elements
/// between `A[i]` and `A[i+1]`". Multiplying by `size_of::<T>()` here is the obvious-looking
/// mistake and it does not fail loudly: `batchCount == 1` ignores the strides entirely, and for a
/// batch the first slice is still correct while every later slice is read and written eight times
/// too far along for `f64`. On hardware that showed up as a batched matmul whose first `m*n`
/// outputs were exact and whose remaining slices were untouched, plus *other* tests failing at
/// random with values larger than the operands could produce — the stray writes landing in whatever
/// the caching allocator had placed next.
///
/// The handle's lock is taken **here** rather than by the caller: a `cublasHandle_t` is not
/// thread-safe, and this stream's handle is shared by every thread that uses the ordinal, so this
/// is the one call that has to be serialized and the one place a guard cannot be forgotten. See
/// [`CublasHandle`]. A math mode is set inside that lock and the previous one restored before it
/// is released, so no other call of this crate ever sees it.
///
/// # Safety
/// The three pointers must be valid for `geom`'s extents on the handle's stream, including the
/// batch range `batchCount * stride` reaches.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch<T: CublasScalar>(
    blas: &CublasHandle,
    a: &cublas_args::CublasArgs,
    math: Option<MathMode>,
    alpha: &T,
    beta: &T,
    ptrs: [u64; 3],
) -> Result<sys::cublasStatus_t> {
    let _api = blas.lock();
    let handle = *blas.blas().handle();
    // Nothing is enqueued yet, so a failure to set the mode is a plain refusal.
    #[cfg(not(cublas_math_mode))]
    let _ = math;
    #[cfg(cublas_math_mode)]
    let previous = match math {
        None => None,
        Some(mode) => match unsafe { replace_math_mode(handle, mode) } {
            Ok(previous) => Some(previous),
            Err(status) => return Ok(status),
        },
    };
    let status = unsafe { gemm_unchecked::<T>(handle, a, alpha, beta, ptrs) };
    // The product is enqueued by now: a handle left in the asked mode would leak the mode into
    // later calls, which is an error rather than a fall-through.
    #[cfg(cublas_math_mode)]
    if let Some(previous) = previous {
        let restored = unsafe { sys::cublasSetMathMode(handle, previous) };
        if restored != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            rstsr_raise!(
                RuntimeError,
                "cuda matmul: cuBLAS status {restored:?} while restoring the handle math mode after a gemm"
            )?;
        }
    }
    Ok(status)
}

/// The `gemm` call proper, on a handle whose lock is held.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_unchecked<T: CublasScalar>(
    handle: sys::cublasHandle_t,
    a: &cublas_args::CublasArgs,
    alpha: &T,
    beta: &T,
    ptrs: [u64; 3],
) -> sys::cublasStatus_t {
    let [ptr_a, ptr_b, ptr_c] = ptrs;
    let (stride_a, stride_b, stride_c) = cublas_args::batch_strides(a);
    unsafe {
        if a.batch > 1 {
            T::gemm_strided_batched(
                handle,
                op(a.transa),
                op(a.transb),
                a.mo,
                a.no,
                a.ko,
                alpha,
                // slot A holds `geom.b`, slot B holds `geom.a`; the strides follow the buffer.
                ptr_b,
                a.lda,
                stride_a,
                ptr_a,
                a.ldb,
                stride_b,
                beta,
                ptr_c,
                a.ldc,
                stride_c,
                a.batch,
            )
        } else {
            T::gemm(
                handle,
                op(a.transa),
                op(a.transb),
                a.mo,
                a.no,
                a.ko,
                alpha,
                ptr_b,
                a.lda,
                ptr_a,
                a.ldb,
                beta,
                ptr_c,
                a.ldc,
            )
        }
    }
}

/// One `gemm` call prepared for the cuBLAS drivers.
pub(crate) struct GemmCall<'a> {
    /// The stream the call is enqueued on.
    pub exec: &'a Arc<CudaStreamObj>,
    /// The arguments, in cuBLAS's terms.
    pub args: cublas_args::CublasArgs,
    /// `[a, b, c]` named after the **geom** operands: each the buffer's base folded with its
    /// layout's own offset in bytes.
    pub ptrs: [u64; 3],
    /// Cross-stream ordering events and read-event guards; held until the enqueue is done.
    #[allow(dead_code)]
    orders: [Option<crate::kernel::Event>; 3],
    #[allow(dead_code)]
    reads: [cudarc::driver::SyncOnDrop<'a>; 3],
}

/// A value, or the refusal of the problem.
pub(crate) type Served<T> = core::result::Result<T, Refusal>;

/// Derive the arguments and the pointers of a `gemm` on this problem.
///
/// Refuses the problems cuBLAS cannot express (the rule table in [`cublas_args`]). The read-event
/// guards of the three buffers live in the returned call, so they are held until the caller drops
/// it, after the enqueue.
pub(crate) fn prepare<'a, A, B, C, TA, TB, TC, TBuf>(
    u: &Buffers<'a, A, B, C>,
    p: &Problem<'a, TA, TB, TC, TBuf>,
) -> Result<Served<GemmCall<'a>>>
where
    A: 'static,
    B: 'static,
    C: 'static,
    TA: 'static,
    TB: 'static,
    TC: 'static,
    TBuf: 'static,
{
    let geom = &p.geom;
    let Some(args) = cublas_args::args(geom) else {
        return Ok(Err(Refusal::Cannot(
            "no column-major expression (stepped or reversed layout, no unit-stride operand axis, \
             non-affine batch, or a dimension over i32)",
        )));
    };
    let exec = p.device.stream();
    // The pointers are named after the **geom** operands: `geom.a` is the caller's `b` under
    // `ColMajor`, so each operand's buffer, element size and input order follow the swap.
    let (ord_a, ord_b, ord_c) = (order_input(u.a, exec)?, order_input(u.b, exec)?, order_output(u.c, exec)?);
    let (ua, ra) = u.a.read_ptr_on(exec);
    let (ub, rb) = u.b.read_ptr_on(exec);
    let (uc, rc) = u.c.read_ptr_on(exec);
    let sa = core::mem::size_of::<A>() as u64;
    let sb = core::mem::size_of::<B>() as u64;
    let (orders, pa, ea, pb, eb) = if geom.swapped {
        ([ord_b, ord_a, ord_c], ub as usize, sb, ua as usize, sa)
    } else {
        ([ord_a, ord_b, ord_c], ua as usize, sa, ub as usize, sb)
    };
    let ec = core::mem::size_of::<C>() as u64;
    let ptr = |base: u64, off: u64, esz: u64| base.wrapping_add(off.wrapping_mul(esz));
    Ok(Ok(GemmCall {
        exec,
        args,
        ptrs: [
            ptr(pa as u64, geom.a.0 as u64, ea),
            ptr(pb as u64, geom.b.0 as u64, eb),
            ptr(uc as u64, geom.c.0 as u64, ec),
        ],
        orders,
        reads: [ra, rb, rc],
    }))
}

/// Prepare one `gemm` call and run `f` with what a cuBLAS driver needs.
///
/// Declines with [`Refusal::NoCublas`] when `libcublas` is not loadable; otherwise `f` gets the
/// handle (unlocked: `f` takes the lock), the arguments and the pointers of [`prepare`].
pub(crate) fn with_gemm_call<A, B, C, TA, TB, TC, TBuf, F>(
    u: &Buffers<'_, A, B, C>,
    p: &Problem<'_, TA, TB, TC, TBuf>,
    f: F,
) -> Result<Verdict>
where
    A: 'static,
    B: 'static,
    C: 'static,
    TA: 'static,
    TB: 'static,
    TC: 'static,
    TBuf: 'static,
    F: FnOnce(&Arc<CublasHandle>, &cublas_args::CublasArgs, [u64; 3]) -> Result<Verdict>,
{
    let call = match prepare(u, p)? {
        Ok(call) => call,
        Err(r) => return Ok(Verdict::Declined(r)),
    };
    let Some(blas) = call.exec.cublas() else {
        return Ok(Verdict::Declined(Refusal::NoCublas));
    };
    f(blas, &call.args, call.ptrs)
}

/// The status as a verdict: a non-success status is a refusal (nothing was enqueued).
pub(crate) fn status_verdict(status: sys::cublasStatus_t) -> Verdict {
    if status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Verdict::Ran
    } else {
        Verdict::Declined(Refusal::CublasStatus(status))
    }
}

/// One attempt of the classic driver.
///
/// The compute mode is a math mode of the handle here: the typed entries have no compute-type
/// argument. `Default` leaves the handle alone.
pub(super) fn run<TA, TB, TC, TBuf>(
    p: &Problem<'_, TA, TB, TC, TBuf>,
    plan: CublasPlan,
    mantissa_bits: u32,
) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    // the typed entries cover f32 and f64 only
    match plan {
        CublasPlan::F32(mode) => {
            let ask = bound(ask_32f(mode));
            if let Ask::No(why) = ask {
                return Ok(Verdict::Declined(Refusal::Cannot(why)));
            }
            typed::<f32, TA, TB, TC, TBuf>(p, ask)
        },
        CublasPlan::F64(mode) => {
            let ask = bound(ask_64f(mode, mantissa_bits)?);
            if let Ask::No(why) = ask {
                return Ok(Verdict::Declined(Refusal::Cannot(why)));
            }
            typed::<f64, TA, TB, TC, TBuf>(p, ask)
        },
        CublasPlan::C32(mode) => {
            let ask = bound(ask_32f(mode));
            if let Ask::No(why) = ask {
                return Ok(Verdict::Declined(Refusal::Cannot(why)));
            }
            typed::<Complex<f32>, TA, TB, TC, TBuf>(p, ask)
        },
        CublasPlan::C64(mode) => {
            let ask = bound(ask_64f(mode, mantissa_bits)?);
            if let Ask::No(why) = ask {
                return Ok(Verdict::Declined(Refusal::Cannot(why)));
            }
            typed::<Complex<f64>, TA, TB, TC, TBuf>(p, ask)
        },
        CublasPlan::None => {
            Ok(Verdict::Declined(Refusal::Cannot("the typed gemm entries cover f32, f64 and complex only")))
        },
        CublasPlan::F16(_) | CublasPlan::F16F32(_) | CublasPlan::BF16(_) | CublasPlan::BF16F32(_) => {
            Ok(Verdict::Declined(Refusal::Cannot("the typed gemm entries cover f32, f64 and complex only")))
        },
    }
}

/// Derive the arguments, decline on anything cuBLAS cannot express, launch.
fn typed<T: CublasScalar, TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>, ask: Ask) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    let Some(u) = p.uniform::<T>() else {
        return Ok(Verdict::Declined(Refusal::Cannot("the operands and the output must share one dtype")));
    };
    // SAFETY: the three buffers were re-typed to `T`, so the offsets are in `T`s.
    let math = match ask {
        Ask::Leave => None,
        Ask::Math(mode) => Some(mode),
        Ask::No(_) => unreachable!("a refusal is answered before the buffers are re-typed"),
    };
    let (Some(alpha), Some(beta)) = (scale::<T, TC>(p.alpha), scale::<T, TC>(p.beta)) else {
        return Ok(Verdict::Declined(Refusal::Cannot("the output type is not of the plan's family")));
    };
    with_gemm_call(&u, p, |blas, args, ptrs| {
        // SAFETY: the pointers are the buffers' bases folded with the layouts' own offsets, and
        // `args` was derived from the same `geom` those offsets came from.
        let status = unsafe { launch::<T>(blas, args, math, &alpha, &beta, ptrs)? };
        Ok(status_verdict(status))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every 32F mode has a math mode of the classic entries but `Fast16BF`.
    #[test]
    fn math_modes_of_the_32f_family() {
        assert_eq!(ask_32f(ComputeType32F::default()), Ask::Leave);
        assert_eq!(ask_32f(ComputeType32F::Pedantic), Ask::Math(MathMode::Pedantic));
        assert_eq!(ask_32f(ComputeType32F::FastTF32), Ask::Math(MathMode::Tf32TensorOp));
        assert_eq!(ask_32f(ComputeType32F::Fast16F), Ask::Math(MathMode::TensorOp));
        assert!(matches!(ask_32f(ComputeType32F::Fast16BF), Ask::No(why) if why.contains("bf16")));
    }

    /// Every 64F mode has one; the fixed-point emulation asks for its bindings.
    #[test]
    fn math_modes_of_the_64f_family() {
        assert_eq!(ask_64f(ComputeType64F::default(), 0).unwrap(), Ask::Leave);
        assert_eq!(ask_64f(ComputeType64F::Pedantic, 0).unwrap(), Ask::Math(MathMode::Pedantic));
        let ask = ask_64f(ComputeType64F::EmulatedFixedPoint, 0).unwrap();
        assert_eq!(ask, Ask::Math(MathMode::Fp64EmulatedFixedPoint));
        assert_eq!(bound(ask).is_refusal(), !cfg!(cublas_fixedpoint_math));
        // a mantissa width cannot be applied by this cudarc: an error, never a silent default
        assert!(ask_64f(ComputeType64F::EmulatedFixedPoint, 30).is_err());
    }

    /// A math mode the bindings lack is a refusal naming the bindings, not a `Default`.
    #[test]
    fn unbound_math_modes_refuse() {
        for mode in [ComputeType32F::Pedantic, ComputeType32F::FastTF32, ComputeType32F::Fast16F] {
            assert_eq!(bound(ask_32f(mode)).is_refusal(), !cfg!(cublas_math_mode), "{mode:?}");
        }
    }

    impl Ask {
        fn is_refusal(&self) -> bool {
            matches!(self, Ask::No(_))
        }
    }
}
