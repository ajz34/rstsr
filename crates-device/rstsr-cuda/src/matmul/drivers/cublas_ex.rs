//! The `cublasGemmEx` driver: `cublasGemmEx` / `cublasGemmStridedBatchedEx`.
//!
//! The same column-major mapping as the classic entries (it shares [`cublas::with_gemm_call`]),
//! with the compute type as an argument instead of a handle math mode, so nothing on the handle
//! is touched. The storage type of all three matrices and the type of `alpha`/`beta` is the
//! output type (`CUDA_R_32F` for the 32F family, `CUDA_R_64F` for the 64F family).
//!
//! What it cannot express: the FP64 fixed-point emulation (a math mode of the classic handle) and
//! a triple whose storage it has no uniform type for (mixed operands).

use cudarc::cublas::sys;

use super::cublas::{status_verdict, with_gemm_call, CublasScalar};
use super::{CublasPlan, Problem, Refusal, Verdict};
use crate::matmul::cublas_args::CublasArgs;
use crate::prelude_dev::*;

/// A scalar the `*Ex` entries take in all three matrix slots, with the scales of the same type.
pub(crate) trait ExScalar: CublasScalar {
    /// The `cudaDataType_t` of `A`, `B` and `C`.
    const DATA: sys::cudaDataType_t;

    /// # Safety
    /// `ptrs` must be valid for `args`' extents on the handle's stream; the handle's lock must be
    /// held.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_ex(
        handle: sys::cublasHandle_t,
        a: &CublasArgs,
        compute: sys::cublasComputeType_t,
        alpha: &Self,
        beta: &Self,
        ptrs: [u64; 3],
    ) -> sys::cublasStatus_t;
}

/// The `gemm` call proper, on a handle whose lock is held.
///
/// # Safety
/// See [`ExScalar::gemm_ex`].
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_ex_unchecked<T: ExScalar>(
    handle: sys::cublasHandle_t,
    a: &CublasArgs,
    compute: sys::cublasComputeType_t,
    alpha: &T,
    beta: &T,
    ptrs: [u64; 3],
) -> sys::cublasStatus_t {
    let [ptr_a, ptr_b, ptr_c] = ptrs;
    let (stride_a, stride_b, stride_c) = crate::matmul::cublas_args::batch_strides(a);
    let op = |t: bool| if t { sys::cublasOperation_t::CUBLAS_OP_T } else { sys::cublasOperation_t::CUBLAS_OP_N };
    // slot A holds `geom.b`, slot B holds `geom.a`; the strides follow the buffer.
    let (a_ptr, b_ptr) = (ptr_b as *const core::ffi::c_void, ptr_a as *const core::ffi::c_void);
    let c_ptr = ptr_c as *mut core::ffi::c_void;
    let (alpha, beta) = (alpha as *const T as *const core::ffi::c_void, beta as *const T as *const core::ffi::c_void);
    let algo = sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;
    unsafe {
        if a.batch > 1 {
            sys::cublasGemmStridedBatchedEx(
                handle,
                op(a.transa),
                op(a.transb),
                a.mo,
                a.no,
                a.ko,
                alpha,
                a_ptr,
                T::DATA,
                a.lda,
                stride_a,
                b_ptr,
                T::DATA,
                a.ldb,
                stride_b,
                beta,
                c_ptr,
                T::DATA,
                a.ldc,
                stride_c,
                a.batch,
                compute,
                algo,
            )
        } else {
            sys::cublasGemmEx(
                handle,
                op(a.transa),
                op(a.transb),
                a.mo,
                a.no,
                a.ko,
                alpha,
                a_ptr,
                T::DATA,
                a.lda,
                b_ptr,
                T::DATA,
                a.ldb,
                beta,
                c_ptr,
                T::DATA,
                a.ldc,
                compute,
                algo,
            )
        }
    }
}

macro_rules! impl_ex_scalar {
    ($t:ty, $data:ident) => {
        impl ExScalar for $t {
            const DATA: sys::cudaDataType_t = sys::cudaDataType_t::$data;

            unsafe fn gemm_ex(
                handle: sys::cublasHandle_t,
                a: &CublasArgs,
                compute: sys::cublasComputeType_t,
                alpha: &Self,
                beta: &Self,
                ptrs: [u64; 3],
            ) -> sys::cublasStatus_t {
                unsafe { gemm_ex_unchecked::<$t>(handle, a, compute, alpha, beta, ptrs) }
            }
        }
    };
}

impl_ex_scalar!(f32, CUDA_R_32F);
impl_ex_scalar!(f64, CUDA_R_64F);

/// One attempt of the `cublasGemmEx` driver.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>, plan: CublasPlan) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    let compute = match plan {
        CublasPlan::F32(mode) => return typed::<f32, TA, TB, TC, TBuf>(p, mode.gemm_ex()),
        CublasPlan::F64(mode) => mode.gemm_ex(),
        CublasPlan::None => None,
    };
    match compute {
        Some(compute) => typed::<f64, TA, TB, TC, TBuf>(p, compute),
        None => Ok(Verdict::Declined(Refusal::Cannot(
            "cublasGemmEx serves f32 and f64 outputs in their own storage, and has no fixed-point emulation",
        ))),
    }
}

/// Derive the arguments, decline on anything cuBLAS cannot express, launch.
fn typed<T: ExScalar, TA, TB, TC, TBuf>(
    p: &Problem<'_, TA, TB, TC, TBuf>,
    compute: sys::cublasComputeType_t,
) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    let Some(u) = p.uniform::<T>() else {
        return Ok(Verdict::Declined(Refusal::Cannot("the operands and the output must share one dtype")));
    };
    with_gemm_call(&u, p, |blas, args, ptrs| {
        let _api = blas.lock();
        // SAFETY: the pointers are the buffers' bases folded with the layouts' own offsets, and
        // `args` was derived from the same `geom` those offsets came from.
        let status = unsafe { T::gemm_ex(*blas.blas().handle(), args, compute, &u.alpha, &u.beta, ptrs) };
        Ok(status_verdict(status))
    })
}
