//! Trait-bundle sufficiency test: generic code bounded only by
//! `LapackDriverAPI<T>` must be able to call the core matmul entry points
//! (including the `MaybeUninit` raw projection that `matmul_from` needs).
//!
//! Regression for the `261015/lapackdriverapi-trait-bound` change.

// device alias for the shared test style (rstsr-blas-traits/src/blas_tests_impl)
pub use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;

use rstsr::prelude::rt::blas::{BlasFloat, LapackDriverAPI};
use rstsr::prelude::*;

fn generic_matmul<B, T>(a: &Tensor<T, B, Ix2>, b: &Tensor<T, B, Ix2>) -> Tensor<T, B, Ix2>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    rt::matmul(a, b)
}

fn generic_matmul_scratch<B, T>(mut scratch: TensorMut<T, B, Ix2>, a: TensorView<T, B, Ix2>, b: TensorView<T, B, Ix2>)
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    scratch.matmul_from(a, b, T::one(), T::zero());
}

fn generic_matmul_dyn<B, T>(scratch_vec: &mut [T], a: &Tensor<T, B, IxD>, b: &Tensor<T, B, Ix2>, device: &B)
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let shape = [a.shape()[0], b.shape()[1]];
    let mut scratch = rt::asarray((scratch_vec, shape.f(), device)).into_dim::<Ix2>();
    scratch.matmul_from(a.view().into_dim::<Ix2>().view(), b.view(), T::one(), T::zero());
}

#[test]
fn test_generic_matmul_under_lapack_driver() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![1.0, 2.0, 3.0, 4.0], [2, 2].f(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((vec![1.0, 0.0, 0.0, 1.0], [2, 2].f(), &device)).into_dim::<Ix2>();

    // allocating entry (a = [[1, 3], [2, 4]] since the fill order is column-major)
    let c = generic_matmul(&a, &b);
    let c = c.to_contig(RowMajor).into_owned();
    assert_eq!(c.view().into_shape((-1,)).to_vec(), vec![1.0, 3.0, 2.0, 4.0]);

    // scratch (matmul_from) entry
    let mut scratch = rt::zeros(([2, 2].f(), &device)).into_dim::<Ix2>().into_owned();
    generic_matmul_scratch(scratch.view_mut(), a.view(), b.view());
    let scratch = scratch.to_contig(RowMajor).into_owned();
    assert_eq!(scratch.view().into_shape((-1,)).to_vec(), vec![1.0, 3.0, 2.0, 4.0]);

    // dynamic-dimension scratch entry (batched style)
    let a_dyn = a.view().into_dim::<IxD>().into_owned();
    let mut scratch_vec = vec![0.0; 4];
    generic_matmul_dyn(&mut scratch_vec, &a_dyn, &b, &device);
    assert_eq!(scratch_vec, vec![1.0, 2.0, 3.0, 4.0]);
}
