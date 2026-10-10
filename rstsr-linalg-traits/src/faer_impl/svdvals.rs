use crate::traits_def::SVDvalsAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// Singular values of a single 2-D matrix.
fn faer_svdvals_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<Tensor<T::Real, DeviceFaer, Ix1>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let faer_a = a.into_faer();

    let result =
        faer_a.singular_values().map_err(|e| rstsr_error!(FaerError, "Faer SVD singular values error: {e:?}"))?;
    Ok(asarray((result, &device)).into_dim::<Ix1>())
}

/// n-dim `svdvals` over the batch dims; the output has shape `batch ++ [k]`
/// (row-major) / `[k] ++ batch` (col-major), `k = min(m, n)`.
pub fn faer_impl_svdvals_f<T>(a: TensorView<'_, T, DeviceFaer, IxD>) -> Result<Tensor<T::Real, DeviceFaer, IxD>>
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

    let result = crate::linalg_util::map_batch_matrices(a, order, &mut faer_svdvals_ix2);

    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    let (batch_shape, matrix, vals) = result?;
    let k = [Ord::min(matrix[0], matrix[1])];
    crate::linalg_util::assemble_batch_matrices_f(vals, &batch_shape, &k, order, &device)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> SVDvalsAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, IxD>;
    fn svdvals_f(self) -> Result<Self::Out> {
        let a = self;
        faer_impl_svdvals_f(a.to_dyn())
    }
}
