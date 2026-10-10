use crate::traits_def::{SVDResult, SVDAPI};
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// SVD of a single 2-D matrix.
fn faer_svd_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    full_matrices: bool,
) -> Result<(Tensor<T, DeviceFaer, Ix2>, Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let faer_a = a.into_faer();

    // svd computation
    let svd_result = match full_matrices {
        true => faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?,
        false => faer_a.thin_svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?,
    };
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    // return to rstsr tensors
    let u = u.into_rstsr().into_owned();
    let s = s.column_vector().into_rstsr();
    let v = v.into_rstsr();

    Ok((
        u.into_contig(device.default_order()),
        s.mapv(|v| T::real_part_impl(&v)).into_contig(device.default_order()),
        v.into_reverse_axes().into_contig(device.default_order()),
    ))
}

/// n-dim `svd` over the batch dims.
pub fn faer_impl_svd_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    full_matrices: bool,
) -> Result<SVDResult<Tensor<T, DeviceFaer, IxD>, Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>>
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

    let result = crate::linalg_util::map_batch_matrices(a, order, &mut |m| faer_svd_ix2(m, full_matrices));

    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    let (batch_shape, matrix, items) = result?;
    let [m, n] = matrix;
    let k = Ord::min(m, n);
    let (mut us, mut ss, mut vts) = (Vec::new(), Vec::new(), Vec::new());
    for (u, s, vt) in items {
        us.push(u);
        ss.push(s);
        vts.push(vt);
    }
    let u_cols = if full_matrices { m } else { k };
    let vt_rows = if full_matrices { n } else { k };
    let u = crate::linalg_util::assemble_batch_matrices_f(us, &batch_shape, &[m, u_cols], order, &device)?;
    let s = crate::linalg_util::assemble_batch_matrices_f(ss, &batch_shape, &[k], order, &device)?;
    let vt = crate::linalg_util::assemble_batch_matrices_f(vts, &batch_shape, &[vt_rows, n], order, &device)?;
    Ok(SVDResult { u, s, vt })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> SVDAPI<DeviceFaer> for (Tr, bool)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = SVDResult<Tensor<T, DeviceFaer, IxD>, Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>;
    fn svd_f(self) -> Result<Self::Out> {
        let (a, full_matrices) = self;
        faer_impl_svd_f(a.to_dyn(), full_matrices)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> SVDAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = SVDResult<Tensor<T, DeviceFaer, IxD>, Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>;
    fn svd_f(self) -> Result<Self::Out> {
        SVDAPI::<DeviceFaer>::svd_f((self, true))
    }
}
