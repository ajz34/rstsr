use crate::faer_impl::batch::batch_and_matrix_shape;
use crate::faer_impl::batch::{map_stack_slices, stack_shape, with_parallel};
use crate::traits_def::CholeskyAPI;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Num;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_cholesky_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "cholesky: the matrix must be square, got {m}x{n}")?;

    with_parallel(&device, || {
        let mut out = zeros_f((stack_shape(&batch_shape, &[m, m], order), &device))?;
        map_stack_slices::<T, T, Ix2, _>(a, out.view_mut(), order, |a_slice, out_slice| {
            faer_impl_cholesky_ix2_f(a_slice, uplo, out_slice)
        })?;
        Ok(out)
    })
}

pub fn faer_impl_cholesky_ix2_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
    mut out: TensorMut<'_, T, DeviceFaer, Ix2>,
) -> Result<()>
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

    // llt computation
    let result = faer_a.llt(faer_uplo).map_err(|e| rstsr_error!(FaerError, "Faer cholesky error: {e}"))?;

    // faer always returns lower triangular matrix
    let result = match uplo {
        Lower => result.L().to_owned(),
        Upper => result.L().adjoint().to_owned(),
    };
    // convert to rstsr tensor with certain layout
    let result = result.into_rstsr().into_contig(device.default_order());

    out.assign_f(result)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        let result = faer_impl_cholesky_f(a.view().to_dyn(), uplo)?;
        let result = result.into_dim::<IxD>().into_dim::<D>();
        Ok(result)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        CholeskyAPI::<DeviceFaer>::cholesky_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for Tr
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        CholeskyAPI::<DeviceFaer>::cholesky_f((a, None))
    }
}
