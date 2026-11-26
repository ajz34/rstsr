//! The `cublasGemmEx` driver: `cublasGemmEx` / `cublasGemmStridedBatchedEx`.
//!
//! The same column-major mapping as the classic entries (it shares [`cublas::with_gemm_call`]),
//! with the compute type as an argument instead of a handle math mode, so nothing on the handle
//! is touched. The storage type of all three matrices and the type of `alpha`/`beta` is the
//! output type (`CUDA_R_32F` for the 32F family, `CUDA_R_64F` for the 64F family).
//!
//! What it cannot express: the FP64 fixed-point emulation (a math mode of the classic handle) and
//! a triple whose storage it has no uniform type for (mixed operands).

use crate::dtype::CudaDType;
use cudarc::cublas::sys;

use super::cublas::{status_verdict, with_gemm_call};
use super::{scale, AlignedScale, CublasGemmTypes, CublasPlan, Problem, Refusal, Verdict};
use crate::matmul::cublas_args::CublasArgs;
use crate::matmul::mode::ComputeType16F;
use crate::prelude_dev::*;

/// The Rust type of `alpha`/`beta` for a family of storage types.
pub(crate) trait ExScalar: CudaDType {
    /// The `cudaDataType_t` of the scales, hence of the matmul descriptor.
    const SCALE: sys::cudaDataType_t;

    /// # Safety
    /// `ptrs` must be valid for `args`' extents on the handle's stream; the handle's lock must be
    /// held.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_ex(
        handle: sys::cublasHandle_t,
        a: &CublasArgs,
        st: CublasGemmTypes,
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
    st: CublasGemmTypes,
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
                st.a,
                a.lda,
                stride_a,
                b_ptr,
                st.b,
                a.ldb,
                stride_b,
                beta,
                c_ptr,
                st.c,
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
                st.a,
                a.lda,
                b_ptr,
                st.b,
                a.ldb,
                beta,
                c_ptr,
                st.c,
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
            const SCALE: sys::cudaDataType_t = sys::cudaDataType_t::$data;

            unsafe fn gemm_ex(
                handle: sys::cublasHandle_t,
                a: &CublasArgs,
                st: CublasGemmTypes,
                compute: sys::cublasComputeType_t,
                alpha: &Self,
                beta: &Self,
                ptrs: [u64; 3],
            ) -> sys::cublasStatus_t {
                unsafe { gemm_ex_unchecked::<$t>(handle, a, st, compute, alpha, beta, ptrs) }
            }
        }
    };
}

impl_ex_scalar!(f32, CUDA_R_32F);
impl_ex_scalar!(f64, CUDA_R_64F);
impl_ex_scalar!(half::f16, CUDA_R_16F);
impl_ex_scalar!(num::Complex<f32>, CUDA_C_32F);
impl_ex_scalar!(num::Complex<f64>, CUDA_C_64F);

/// One attempt of the `cublasGemmEx` driver: the scale type and the storage types follow the
/// plan, and the buffers are re-typed accordingly.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>, plan: CublasPlan) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    if plan.gemm_types().is_none() {
        return Ok(Verdict::Declined(Refusal::Cannot("no cuBLAS problem for this type triple")));
    }
    if plan.compute().is_none() {
        return Ok(Verdict::Declined(Refusal::Cannot(
            "cublasGemmEx has no fixed-point emulation (a math mode of the classic handle)",
        )));
    }
    /// `typed` with the plan's scale type `S` and storage types `A`, `B`, `C`.
    macro_rules! serve {
        ($s:ty, $a:ty, $b:ty, $c:ty) => {
            typed::<$s, $a, $b, $c, TA, TB, TC, TBuf>(p, plan)
        };
    }
    match plan {
        CublasPlan::F32(_) => serve!(f32, f32, f32, f32),
        CublasPlan::F64(_) => serve!(f64, f64, f64, f64),
        CublasPlan::C32(_) => serve!(num::Complex<f32>, num::Complex<f32>, num::Complex<f32>, num::Complex<f32>),
        CublasPlan::C64(_) => serve!(num::Complex<f64>, num::Complex<f64>, num::Complex<f64>, num::Complex<f64>),
        // 16-bit accumulation takes half scales
        CublasPlan::F16(ComputeType16F::Acc16F) => serve!(half::f16, half::f16, half::f16, half::f16),
        CublasPlan::F16(_) => serve!(f32, half::f16, half::f16, half::f16),
        CublasPlan::F16F32(_) => serve!(f32, half::f16, half::f16, f32),
        CublasPlan::BF16(_) => serve!(f32, half::bf16, half::bf16, half::bf16),
        CublasPlan::BF16F32(_) => serve!(f32, half::bf16, half::bf16, f32),
        CublasPlan::None => unreachable!("declined above"),
    }
}

/// Derive the arguments, decline on anything cuBLAS cannot express, launch.
fn typed<S: ExScalar, A, B, C, TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>, plan: CublasPlan) -> Result<Verdict>
where
    A: 'static,
    B: 'static,
    C: 'static,
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    let Some(u) = p.retyped::<A, B, C>() else {
        return Ok(Verdict::Declined(Refusal::Cannot("the buffers are not of the plan's storage types")));
    };
    let (Some(alpha), Some(beta)) = (scale::<S, TC>(p.alpha), scale::<S, TC>(p.beta)) else {
        return Ok(Verdict::Declined(Refusal::Cannot("the output type is not of the plan's family")));
    };
    let (Some(st), Some(compute)) = (plan.gemm_types(), plan.compute()) else {
        return Ok(Verdict::Declined(Refusal::Cannot("no cuBLAS problem for this type triple")));
    };
    with_gemm_call(&u, p, |blas, args, ptrs| {
        let _api = blas.lock();
        // SAFETY: the pointers are the buffers' bases folded with the layouts' own offsets, and
        // `args` was derived from the same `geom` those offsets came from.
        let (alpha, beta) = (AlignedScale(alpha), AlignedScale(beta));
        let status = unsafe { S::gemm_ex(*blas.blas().handle(), args, st, compute, &alpha.0, &beta.0, ptrs) };
        Ok(status_verdict(status))
    })
}
