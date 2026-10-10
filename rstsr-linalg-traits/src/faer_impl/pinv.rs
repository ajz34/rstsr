use crate::traits_def::{PinvAPI, PinvResult};
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::{Float, FromPrimitive, Num, Zero};
use rstsr_core::prelude_dev::*;

/// Pseudoinverse (and its rank) of a single 2-D matrix.
fn faer_pinv_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<(Tensor<T, DeviceFaer, Ix2>, usize)>
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Send + Sync,
{
    // compute rcond value
    let atol = atol.unwrap_or(T::Real::zero());
    let rtol = rtol.unwrap_or({
        let [m, n]: [usize; 2] = *a.shape();
        let mnmax = T::Real::from_usize(Ord::max(m, n)).unwrap();
        mnmax * T::Real::epsilon()
    });

    // transform to faer matrix
    let faer_a = a.into_faer();

    // svd computation
    let svd_result = faer_a.thin_svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    // return to rstsr tensors
    let u = u.into_rstsr().into_owned();
    let s = s.column_vector().into_rstsr();
    let v = v.into_rstsr();

    // compute pinv
    let s = s.mapv(|x| T::real_part_impl(&x));
    let maxs = *s.raw().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let val = atol + rtol * maxs;
    let rank = s.raw().iter().take_while(|&&x| x > val).count();
    let mut u = u.into_slice((.., ..rank));
    u /= s.i((None, ..rank));
    let a_pinv = v.i((.., ..rank)) % u.mapv(|x| T::conj_impl(&x)).t();
    let pinv = a_pinv.into_dim::<Ix2>();

    Ok((pinv, rank))
}

/// n-dim `pinv` over the batch dims. The `rank` scalar reports the largest
/// per-matrix rank in the stack (the array-API drops it).
pub fn faer_impl_pinv_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<(Tensor<T, DeviceFaer, IxD>, usize)>
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Zero + Send + Sync,
{
    let device = a.device().clone();
    let order = device.default_order();

    // set parallel mode once for the whole batch
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    if let Some(pool) = pool {
        faer::set_global_parallelism(Par::rayon(pool.current_num_threads()));
    }

    let result = crate::linalg_util::map_batch_matrices(a, order, &mut |m| faer_pinv_ix2(m, atol, rtol));

    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    let (batch_shape, matrix, items) = result?;
    let [m, n] = matrix;
    let mut rank = 0;
    let mut mats = Vec::with_capacity(items.len());
    for (pinv, r) in items {
        rank = Ord::max(rank, r);
        mats.push(pinv);
    }
    // the pseudoinverse of an m x n matrix is n x m
    let pinv = crate::linalg_util::assemble_batch_matrices_f(mats, &batch_shape, &[n, m], order, &device)?;
    Ok((pinv, rank))
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> PinvAPI<DeviceFaer> for (Tr, T::Real, T::Real)
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Zero + Send + Sync,
    D: DimAPI,
{
    type Out = PinvResult<Tensor<T, DeviceFaer, IxD>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        let (pinv, rank) = faer_impl_pinv_f(a.to_dyn(), Some(atol), Some(rtol))?;
        Ok(PinvResult { pinv, rank })
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> PinvAPI<DeviceFaer> for Tr
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Zero + Send + Sync,
    D: DimAPI,
{
    type Out = PinvResult<Tensor<T, DeviceFaer, IxD>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        let (pinv, rank) = faer_impl_pinv_f(a.to_dyn(), None, None)?;
        Ok(PinvResult { pinv, rank })
    }
}
