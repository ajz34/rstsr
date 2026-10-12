//! Eigenvalues of a general (non-Hermitian) matrix on the faer device (the
//! array-API `linalg.eigvals`).
//!
//! The eigenvalue-only companion of [`eig`](crate::faer_impl::eig): it uses
//! faer's `eigenvalues()` (no eigenvectors) and returns the complex spectrum in
//! faer's (LAPACK-like) order — no sorting.

use crate::faer_impl::batch::batch_and_matrix_shape;
use crate::faer_impl::batch::{map_stack_slices, stack_shape, with_parallel};
use crate::traits_def::EigvalsAPI;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Complex;
use rstsr_core::prelude_dev::*;

/// Eigenvalues of a single 2-D matrix (no parallel-mode handling).
fn faer_eigvals_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<Tensor<Complex<T::Real>, DeviceFaer, Ix1>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let result = a.into_faer().eigenvalues().map_err(|e| rstsr_error!(FaerError, "Faer Eigen error: {e:?}"))?;
    Ok(asarray((result, &device)).into_dim::<Ix1>())
}

pub fn faer_impl_eigvals_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
) -> Result<Tensor<Complex<T::Real>, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "eigvals: the matrix must be square, got {m}x{n}")?;

    // a zero-size matrix axis or batch yields an empty spectrum (faer is not called)
    if m == 0 || n == 0 || batch_shape.contains(&0) {
        return zeros_f((stack_shape(&batch_shape, &[m], order), &device));
    }

    with_parallel(&device, || {
        let mut out = zeros_f((stack_shape(&batch_shape, &[m], order), &device))?;
        map_stack_slices::<T, Complex<T::Real>, Ix1, _>(a, out.view_mut(), order, |a_slice, mut out_slice| {
            out_slice.assign_f(faer_eigvals_ix2(a_slice)?)
        })?;
        Ok(out)
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigvalsAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<Complex<T::Real>, DeviceFaer, D::SmallerOne>;
    fn eigvals_f(self) -> Result<Self::Out> {
        let a = self;
        let result = faer_impl_eigvals_f(a.view().to_dyn())?;
        Ok(result.into_dim::<D::SmallerOne>())
    }
}
