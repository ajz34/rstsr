use crate::faer_impl::batch::{map_stack_slices, stack_shape, with_parallel};
use crate::linalg_util::batch_and_matrix_shape;
use crate::traits_def::EigvalshAPI;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// Eigenvalues of a single 2-D symmetric/Hermitian matrix (no parallel-mode
/// handling).
fn faer_eigvalsh_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T::Real, DeviceFaer, Ix1>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_a = a.into_faer();
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    let result = faer_a
        .self_adjoint_eigenvalues(faer_uplo)
        .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;
    Ok(asarray((result, &device)).into_dim::<Ix1>())
}

pub fn faer_impl_eigvalsh_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T::Real, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "eigvalsh: the matrix must be square, got {m}x{n}")?;

    with_parallel(&device, || {
        let mut out = zeros_f((stack_shape(&batch_shape, &[m], order), &device))?;
        map_stack_slices::<T, T::Real, Ix1, _>(a, out.view_mut(), order, |a_slice, mut out_slice| {
            out_slice.assign_f(faer_eigvalsh_ix2(a_slice, uplo)?)
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
impl<ImplType> EigvalshAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        let result = faer_impl_eigvalsh_f(a.view().to_dyn(), uplo)?;
        Ok(result.into_dim::<D::SmallerOne>())
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigvalshAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        EigvalshAPI::<DeviceFaer>::eigvalsh_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigvalshAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let a = self;
        EigvalshAPI::<DeviceFaer>::eigvalsh_f((a, None))
    }
}
