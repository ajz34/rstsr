use crate::faer_impl::batch::{batch_and_matrix_shape, map_stack_scalars, with_parallel};
use crate::traits_def::DetAPI;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Num;
use rstsr_core::prelude_dev::*;

/// Determinant of a single 2-D matrix (no parallel-mode handling).
fn faer_det_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<T>
where
    T: ComplexField,
{
    Ok(a.into_faer().determinant())
}

/// n-dim `det`: the determinant of every `(..., M, M)` matrix slice, with the
/// batch shape.
pub fn faer_impl_det_f<T>(a: TensorView<'_, T, DeviceFaer, IxD>) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "det: the matrix must be square, got {m}x{n}")?;

    with_parallel(&device, || {
        let mut out = zeros_f((batch_shape, &device))?;
        map_stack_scalars(a, out.view_mut(), order, |a_slice, o| {
            *o = faer_det_ix2(a_slice)?;
            Ok(())
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
impl<ImplType> DetAPI<DeviceFaer> for Tr
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn det_f(self) -> Result<Self::Out> {
        faer_impl_det_f(self.to_dyn())
    }
}
