use crate::faer_impl::batch::{map_stack_slices3, stack_shape, with_parallel};
use crate::linalg_util::batch_and_matrix_shape;
use crate::traits_def::{SVDResult, SVDAPI};
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Num;
use rstsr_core::prelude_dev::*;

/// SVD of a single 2-D matrix (no parallel-mode handling).
fn faer_svd_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    full_matrices: bool,
) -> Result<SVDResult<Tensor<T, DeviceFaer, Ix2>, Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let faer_a = a.into_faer();

    let svd_result = match full_matrices {
        true => faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?,
        false => faer_a.thin_svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?,
    };
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    let u = u.into_rstsr().into_owned();
    let s = s.column_vector().into_rstsr();
    let v = v.into_rstsr();

    Ok(SVDResult {
        u: u.into_contig(device.default_order()),
        s: s.mapv(|v| T::real_part_impl(&v)).into_contig(device.default_order()),
        vt: v.into_reverse_axes().into_contig(device.default_order()),
    })
}

pub fn faer_impl_svd_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    full_matrices: bool,
) -> Result<SVDResult<Tensor<T, DeviceFaer, IxD>, Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    let k = Ord::min(m, n);
    let (u_cols, vt_rows) = if full_matrices { (m, n) } else { (k, k) };

    with_parallel(&device, || {
        let mut u = zeros_f((stack_shape(&batch_shape, &[m, u_cols], order), &device))?;
        let mut s = zeros_f((stack_shape(&batch_shape, &[k], order), &device))?;
        let mut vt = zeros_f((stack_shape(&batch_shape, &[vt_rows, n], order), &device))?;
        map_stack_slices3::<T, T, Ix2, T::Real, Ix1, T, Ix2, _>(
            a,
            u.view_mut(),
            s.view_mut(),
            vt.view_mut(),
            order,
            |a_slice, mut u_slice, mut s_slice, mut vt_slice| {
                let r = faer_svd_ix2(a_slice, full_matrices)?;
                u_slice.assign_f(r.u)?;
                s_slice.assign_f(r.s)?;
                vt_slice.assign_f(r.vt)
            },
        )?;
        Ok(SVDResult { u, s, vt })
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> SVDAPI<DeviceFaer> for (Tr, bool)
where
    T: ComplexField + Num,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out =
        SVDResult<Tensor<T, DeviceFaer, D>, Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn svd_f(self) -> Result<Self::Out> {
        let (a, full_matrices) = self;
        let result = faer_impl_svd_f(a.view().to_dyn(), full_matrices)?;
        Ok(SVDResult {
            u: result.u.into_dim::<D>(),
            s: result.s.into_dim::<D::SmallerOne>(),
            vt: result.vt.into_dim::<D>(),
        })
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
    T: ComplexField + Num,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out =
        SVDResult<Tensor<T, DeviceFaer, D>, Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn svd_f(self) -> Result<Self::Out> {
        SVDAPI::<DeviceFaer>::svd_f((self, true))
    }
}
