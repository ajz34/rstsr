use crate::traits_def::CholeskyAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// Cholesky factor of a single 2-D matrix.
fn faer_cholesky_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let uplo = uplo.unwrap_or(match device.default_order() {
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
    Ok(result.into_rstsr().into_contig(device.default_order()))
}

/// n-dim `cholesky` over the batch dims.
pub fn faer_impl_cholesky_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();

    // set parallel mode once for the whole batch
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    if let Some(pool) = pool {
        faer::set_global_parallelism(Par::rayon(pool.current_num_threads()));
    }

    let result = crate::linalg_util::map_batch_matrices(a, order, &mut |m| faer_cholesky_ix2(m, uplo));

    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    let (batch_shape, matrix, mats) = result?;
    crate::linalg_util::assemble_batch_matrices_f(mats, &batch_shape, &matrix, order, &device)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        faer_impl_cholesky_f(a.to_dyn(), uplo)
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
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
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
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        CholeskyAPI::<DeviceFaer>::cholesky_f((a, None))
    }
}
