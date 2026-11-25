//! The classic cuBLAS driver: the typed `gemm` / `gemmStridedBatched` entries.
//!
//! The fast path for every layout it can express: on an A100 `dgemm` reaches the FP64 tensor-core
//! pipe (19.5 TFLOPS) where a SIMT tile tops out at 9.7, and on every other shape it is at least
//! as fast for a hundredth of the engineering. What it cannot express is derived in
//! [`super::super::cublas_args`] and handed to the next driver.
//!
//! The entries cover `f32` and `f64` only; the other accelerated triples come with the
//! `cublasGemmEx` and cuBLASLt drivers.
//!
//! # Thread safety
//!
//! A `cublasHandle_t` is not thread-safe, and this crate shares one per ordinal (see
//! [`CudaStreamObj`][crate::stream::CudaStreamObj]). [`launch`] therefore takes the handle's lock
//! itself. That is a requirement of cuBLAS, not a precaution: two host threads entering a `gemm` on
//! one handle at the same time is undefined.

use std::any::TypeId;

use cudarc::cublas::sys;

use super::{Problem, Refusal, Uniform, Verdict};
use crate::dtype::CudaDType;
use crate::kernel::{order_input, order_output};
use crate::matmul::cublas_args;
use crate::prelude_dev::*;
use crate::stream::CublasHandle;

/// The float widths cuBLAS's typed `gemm` covers, which is exactly what `DeviceMatMulAPI` is
/// instantiated for on this device.
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
                        alpha,
                        a as *const $t,
                        lda,
                        b as *const $t,
                        ldb,
                        beta,
                        c as *mut $t,
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
                        alpha,
                        a as *const $t,
                        lda,
                        stride_a,
                        b as *const $t,
                        ldb,
                        stride_b,
                        beta,
                        c as *mut $t,
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

fn op(t: bool) -> sys::cublasOperation_t {
    if t {
        sys::cublasOperation_t::CUBLAS_OP_T
    } else {
        sys::cublasOperation_t::CUBLAS_OP_N
    }
}

/// Issue the `gemm`. Returns the cuBLAS status; the caller hands the problem to the next driver on
/// any non-success, because a returned status is a launch-time validation failure and nothing has
/// been enqueued.
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
/// [`CublasHandle`].
///
/// # Safety
/// The three pointers must be valid for `geom`'s extents on the handle's stream, including the
/// batch range `batchCount * stride` reaches.
pub(crate) unsafe fn launch<T: CublasScalar>(
    blas: &CublasHandle,
    a: &cublas_args::CublasArgs,
    alpha: &T,
    beta: &T,
    ptr_a: u64,
    ptr_b: u64,
    ptr_c: u64,
) -> sys::cublasStatus_t {
    let _api = blas.lock();
    let handle = *blas.blas().handle();
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

/// One attempt of the classic driver.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    // the typed entries cover f32 and f64 only
    if TypeId::of::<TC>() == TypeId::of::<f32>() {
        return typed(p.uniform::<f32>(), p);
    }
    if TypeId::of::<TC>() == TypeId::of::<f64>() {
        return typed(p.uniform::<f64>(), p);
    }
    Ok(Verdict::Declined(Refusal::Cannot("the typed gemm entries cover f32 and f64 only")))
}

/// Derive the arguments, decline on anything cuBLAS cannot express, launch.
fn typed<T: CublasScalar, TA, TB, TC, TBuf>(
    u: Option<Uniform<'_, T>>,
    p: &Problem<'_, TA, TB, TC, TBuf>,
) -> Result<Verdict> {
    let Some(u) = u else {
        return Ok(Verdict::Declined(Refusal::Cannot("the operands and the output must share one dtype")));
    };
    let geom = &p.geom;
    let Some(args) = cublas_args::args(geom) else {
        return Ok(Verdict::Declined(Refusal::Cannot(
            "no column-major expression (stepped or reversed layout, no unit-stride operand axis, \
             non-affine batch, or a dimension over i32)",
        )));
    };
    let exec = p.device.stream();
    let Some(blas) = exec.cublas() else {
        return Ok(Verdict::Declined(Refusal::NoCublas));
    };
    // Read-event guards held until after the cuBLAS enqueue (fn scope). `geom.a` is the buffer
    // whose layout went first to the rule table: the caller's `b` under `ColMajor`.
    let (buf_a, buf_b) = if geom.swapped { (u.b, u.a) } else { (u.a, u.b) };
    let _ga = order_input(buf_a, exec)?;
    let _gb = order_input(buf_b, exec)?;
    let _gc = order_output(u.c, exec)?;
    let (pa, _ra) = buf_a.read_ptr_on(exec);
    let (pb, _rb) = buf_b.read_ptr_on(exec);
    let (pc, _rc) = u.c.read_ptr_on(exec);
    let esz = core::mem::size_of::<T>() as u64;
    let (pa, pb, pc) = (
        (pa as u64).wrapping_add(geom.a.0 as u64 * esz),
        (pb as u64).wrapping_add(geom.b.0 as u64 * esz),
        (pc as u64).wrapping_add(geom.c.0 as u64 * esz),
    );

    // SAFETY: the pointers are the buffers' bases folded with the layouts' own offsets, and
    // `args` was derived from the same `geom` those offsets came from.
    let status = unsafe { launch(blas, &args, &u.alpha, &u.beta, pa, pb, pc) };
    if matches!(status, sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
        Ok(Verdict::Ran)
    } else {
        Ok(Verdict::Declined(Refusal::CublasStatus(status)))
    }
}
