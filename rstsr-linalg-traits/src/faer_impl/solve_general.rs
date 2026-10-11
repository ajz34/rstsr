use crate::faer_impl::batch::{map_stack_slices_inplace, solve_plan, with_parallel};
use crate::traits_def::SolveGeneralAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_blas_traits::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// n-dim `solve_general`: solve `a x = b` for a stack `a` of `(..., M, M)`
/// matrices and right-hand side `b` of `(..., M, K)` matrices or `(..., M)`
/// vectors. The two batch shapes broadcast against each other.
///
/// An owned or contiguous mutable `b` is solved in place (its own buffer is the
/// output, so no data is copied); only the non-contiguous mutable and the
/// immutable cases allocate a work buffer. Because an in-place solve cannot grow,
/// a batched `a` whose batch does not fit into `b`'s requires an immutable `b`
/// (the allocating form).
pub fn faer_impl_solve_general_f<'b, T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorReference<'b, T, DeviceFaer, IxD>,
) -> Result<TensorMutable<'b, T, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let plan = solve_plan(a.shape(), b.shape(), order, "solve_general")?;
    let a_b = a.to_broadcast_f(plan.a_shape(order))?;
    let out_shape = plan.out_shape(order);

    with_parallel(&device, || {
        let b_mut = overwritable_convert(b)?;
        let target = if b_mut.view().shape().as_slice() == out_shape.as_slice() {
            b_mut
        } else {
            match b_mut {
                // the solution batch outgrows b: only an allocating (owned) b can hold it
                TensorMutable::Owned(t) => TensorMutable::Owned(t.to_broadcast_f(out_shape)?.to_owned()),
                _ => {
                    return rstsr_raise!(
                        InvalidLayout,
                        "solve_general: an in-place solve needs the solution shape to equal b's shape"
                    )
                },
            }
        };
        let done = map_stack_slices_inplace(a_b, target, order, |a_slice, b_slice| {
            // solve linear system
            let faer_a = a_slice.into_faer();
            let faer_b = b_slice.into_faer();
            let svd_result = faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SVD error: {e:?}"))?;
            svd_result.solve_in_place(faer_b);
            Ok(())
        })?;
        Ok(done.clone_to_mut())
    })
}

#[duplicate_item(
    ImplType                                                            TrA                                 TrB                              ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, DA>] [&TensorAny<Rb, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, DA> ] [TensorView<'_, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, DA>] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                                                       ] [TensorView<'_, T, DeviceFaer, DA>] [TensorView<'_, T, DeviceFaer, DB>];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let a_dyn = a.to_dyn();
        let b_dyn = b.to_dyn();
        let result = faer_impl_solve_general_f(a_dyn, b_dyn.into())?;
        Ok(result.into_owned())
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, DA> ] [TensorMut<'b, T, DeviceFaer, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceFaer, DA>] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, DA> ] [Tensor<T, DeviceFaer, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceFaer, DA>] [Tensor<T, DeviceFaer, DB>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, mut b) = self;
        let a_dyn = a.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = faer_impl_solve_general_f(a_dyn, b_ref.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType                               TrA                                TrB                               ;
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [TensorMut<'_, T, DeviceFaer, DA>] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                          ] [TensorMut<'_, T, DeviceFaer, DA>] [TensorView<'_, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [Tensor<T, DeviceFaer, DA>       ] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                          ] [Tensor<T, DeviceFaer, DA>       ] [TensorView<'_, T, DeviceFaer, DB>];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let a_dyn = a.to_dyn();
        let b_dyn = b.to_dyn();
        let result = faer_impl_solve_general_f(a_dyn, b_dyn.into())?;
        Ok(result.into_owned())
    }
}

#[duplicate_item(
    ImplType        TrA                               TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceFaer, DA>] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceFaer, DA>] [Tensor<T, DeviceFaer, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceFaer, DA>       ] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB] [Tensor<T, DeviceFaer, DA>       ] [Tensor<T, DeviceFaer, DB>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, mut b) = self;
        let a_dyn = a.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = faer_impl_solve_general_f(a_dyn, b_ref.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}
