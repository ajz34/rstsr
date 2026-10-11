use crate::faer_impl::batch::{map_stack_slices, stack_shape, with_parallel};
use crate::linalg_util::batch_and_matrix_shape;
use crate::traits_def::SVDvalsAPI;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

/// Singular values of a single 2-D matrix (no parallel-mode handling).
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

pub fn faer_impl_svdvals_f<T>(a: TensorView<'_, T, DeviceFaer, IxD>) -> Result<Tensor<T::Real, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    let k = Ord::min(m, n);

    with_parallel(&device, || {
        let mut out = zeros_f((stack_shape(&batch_shape, &[k], order), &device))?;
        map_stack_slices::<T, T::Real, Ix1, _>(a, out.view_mut(), order, |a_slice, mut out_slice| {
            out_slice.assign_f(faer_svdvals_ix2(a_slice)?)
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
impl<ImplType> SVDvalsAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn svdvals_f(self) -> Result<Self::Out> {
        let a = self;
        let result = faer_impl_svdvals_f(a.view().to_dyn())?;
        Ok(result.into_dim::<D::SmallerOne>())
    }
}
