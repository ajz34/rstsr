use crate::traits_def::DetAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// Determinant of a single 2-D matrix.
fn faer_det_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<T>
where
    T: ComplexField,
{
    Ok(a.into_faer().determinant())
}

/// n-dim `det` over the batch dims; the output has the batch shape.
pub fn faer_impl_det_f<T>(a: TensorView<'_, T, DeviceFaer, IxD>) -> Result<Tensor<T, DeviceFaer, IxD>>
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

    let result = crate::linalg_util::map_batch_matrices(a, order, &mut faer_det_ix2);

    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    let (batch_shape, _matrix, dets) = result?;
    crate::linalg_util::batch_tensor_f(dets, batch_shape, &device)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> DetAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn det_f(self) -> Result<Self::Out> {
        let a = self;
        faer_impl_det_f(a.to_dyn())
    }
}
