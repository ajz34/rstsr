use crate::traits_def::InvAPI;
use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// Inverse of a single 2-D matrix.
fn faer_inv_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<Tensor<T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let faer_a = a.into_faer();

    // inverse computation
    let svd_result = faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let result = svd_result.inverse();

    // convert to rstsr tensor with certain layout
    Ok(result.as_ref().into_rstsr().into_contig(device.default_order()))
}

/// n-dim `inv` over the batch dims.
pub fn faer_impl_inv_f<T>(a: TensorView<'_, T, DeviceFaer, IxD>) -> Result<Tensor<T, DeviceFaer, IxD>>
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

    let result = crate::linalg_util::map_batch_matrices(a, order, &mut faer_inv_ix2);

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
impl<ImplType> InvAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn inv_f(self) -> Result<Self::Out> {
        let a = self;
        faer_impl_inv_f(a.to_dyn())
    }
}
