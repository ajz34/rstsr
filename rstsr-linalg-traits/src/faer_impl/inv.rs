use crate::faer_impl::batch::batch_and_matrix_shape;
use crate::faer_impl::batch::{map_stack_slices, stack_shape, with_parallel};
use crate::traits_def::InvAPI;
use faer::linalg::solvers::DenseSolveCore;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Num;
use rstsr_core::prelude_dev::*;

/// Inverse of a single 2-D matrix (no parallel-mode handling).
fn faer_inv_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<Tensor<T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let faer_a = a.into_faer();
    let svd_result = faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let result = svd_result.inverse();
    // `into_rstsr` homes the result on `DeviceFaer::default()`, so the device
    // must be changed back to the input's
    result.as_ref().into_rstsr().into_contig(device.default_order()).change_device_f(&device)
}

pub fn faer_impl_inv_f<T>(a: TensorView<'_, T, DeviceFaer, IxD>) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "inv: the matrix must be square, got {m}x{n}")?;

    with_parallel(&device, || {
        let mut out = zeros_f((stack_shape(&batch_shape, &[m, m], order), &device))?;
        map_stack_slices::<T, T, Ix2, _>(a, out.view_mut(), order, |a_slice, mut out_slice| {
            out_slice.assign_f(faer_inv_ix2(a_slice)?)
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
impl<ImplType> InvAPI<DeviceFaer> for Tr
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn inv_f(self) -> Result<Self::Out> {
        let a = self;
        let result = faer_impl_inv_f(a.view().to_dyn())?;
        Ok(result.into_dim::<D>())
    }
}
