use crate::faer_impl::batch::batch_and_matrix_shape;
use crate::faer_impl::batch::{map_stack_slices, stack_shape, with_parallel};
use crate::traits_def::{PinvAPI, PinvResult};
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::{Float, FromPrimitive, Num, Zero};
use rstsr_core::prelude_dev::*;

/// Pseudoinverse (and its rank) of a single 2-D matrix (no parallel-mode
/// handling).
fn faer_pinv_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<PinvResult<Tensor<T, DeviceFaer, Ix2>>>
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
    let device = a.device().clone();
    let faer_a = a.into_faer();

    // svd computation
    let svd_result = faer_a.thin_svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    // return to rstsr tensors; `into_rstsr` homes them on `DeviceFaer::default()`,
    // so each device must be changed back to the input's
    let u = u.into_rstsr().into_owned().change_device_f(&device)?;
    let s = s.column_vector().into_rstsr().change_device_f(&device)?;
    let v = v.into_rstsr().change_device_f(&device)?;

    // compute pinv
    let s = s.mapv(|x| T::real_part_impl(&x));
    let maxs = *s.raw().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let val = atol + rtol * maxs;
    let rank = s.raw().iter().take_while(|&&x| x > val).count();
    let mut u = u.into_slice((.., ..rank));
    u /= s.i((None, ..rank));
    let a_pinv = v.i((.., ..rank)) % u.mapv(|x| T::conj_impl(&x)).t();
    let pinv = a_pinv.into_dim::<Ix2>();

    Ok(PinvResult { pinv, rank })
}

/// n-dim `pinv`. The `rank` scalar reports the largest per-matrix rank in the
/// stack (the array-API drops it).
pub fn faer_impl_pinv_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<PinvResult<Tensor<T, DeviceFaer, IxD>>>
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Send + Sync,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;

    let mut rank = 0;
    let pinv = with_parallel(&device, || {
        let mut out = zeros_f((stack_shape(&batch_shape, &[n, m], order), &device))?;
        map_stack_slices::<T, T, Ix2, _>(a, out.view_mut(), order, |a_slice, mut out_slice| {
            let r = faer_pinv_ix2(a_slice, atol, rtol)?;
            rank = Ord::max(rank, r.rank);
            out_slice.assign_f(r.pinv)
        })?;
        Ok(out)
    })?;

    Ok(PinvResult { pinv, rank })
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
    type Out = PinvResult<Tensor<T, DeviceFaer, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        let result = faer_impl_pinv_f(a.view().to_dyn(), Some(atol), Some(rtol))?;
        Ok(PinvResult { pinv: result.pinv.into_dim::<D>(), rank: result.rank })
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
    type Out = PinvResult<Tensor<T, DeviceFaer, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        let result = faer_impl_pinv_f(a.view().to_dyn(), None, None)?;
        Ok(PinvResult { pinv: result.pinv.into_dim::<D>(), rank: result.rank })
    }
}
