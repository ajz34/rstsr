use crate::faer_impl::batch::{check_solve_shapes, map_stack_slices_inplace};
use crate::traits_def::SolveTriangularAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_blas_traits::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// n-dim `solve_triangular`: solve `a x = b` for a stack `a` of `(..., M, M)`
/// triangular matrices and right-hand side `b` of `(..., M, K)` matrices or
/// `(..., M)` vectors.
///
/// An owned or contiguous mutable `b` is solved in place (its own buffer is the
/// output, so no data is copied); only the non-contiguous mutable and the
/// immutable cases allocate a contiguous work buffer, as selected by
/// [`overwritable_convert`].
pub fn faer_impl_solve_triangular_f<'b, T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorReference<'b, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<'b, T, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let faer_par = device.get_current_pool().map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    let uplo = uplo.unwrap_or(match order {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    check_solve_shapes(a.shape(), b.shape(), order, "solve_triangular")?;

    let b_mut = overwritable_convert(b)?;
    let done = map_stack_slices_inplace(a, b_mut, order, |a_slice, b_slice| {
        let faer_a = a_slice.into_faer();
        let faer_b = b_slice.into_faer();
        match uplo {
            Lower => faer::linalg::triangular_solve::solve_lower_triangular_in_place(faer_a, faer_b, faer_par),
            Upper => faer::linalg::triangular_solve::solve_upper_triangular_in_place(faer_a, faer_b, faer_par),
        }
        Ok(())
    })?;
    Ok(done.clone_to_mut())
}

/* #region full-args */

#[duplicate_item(
    ImplType                                                            TrA                                 TrB                               ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, DA>] [&TensorAny<Rb, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, DA> ] [TensorView<'_, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, DA>] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                                                       ] [TensorView<'_, T, DeviceFaer, DA>] [TensorView<'_, T, DeviceFaer, DB>];
)]
impl<ImplType> SolveTriangularAPI<DeviceFaer> for (TrA, TrB, Option<FlagUpLo>)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, DB>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        let a_dyn = a.to_dyn();
        let b_dyn = b.to_dyn();
        let result = faer_impl_solve_triangular_f(a_dyn, b_dyn.into(), uplo)?;
        Ok(result.into_owned().into_dim::<DB>())
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, DA> ] [TensorMut<'b, T, DeviceFaer, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceFaer, DA>] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, DA> ] [Tensor<T, DeviceFaer, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceFaer, DA>] [Tensor<T, DeviceFaer, DB>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceFaer> for (TrA, TrB, Option<FlagUpLo>)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = TrB;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        let a_dyn = a.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = faer_impl_solve_triangular_f(a_dyn, b_ref.into(), uplo)?;
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
impl<ImplType> SolveTriangularAPI<DeviceFaer> for (TrA, TrB, Option<FlagUpLo>)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, DB>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        let a_dyn = a.to_dyn();
        let b_dyn = b.to_dyn();
        let result = faer_impl_solve_triangular_f(a_dyn, b_dyn.into(), uplo)?;
        Ok(result.into_owned().into_dim::<DB>())
    }
}

#[duplicate_item(
    ImplType        TrA                                TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceFaer, DA>] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceFaer, DA>] [Tensor<T, DeviceFaer, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceFaer, DA>       ] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB] [Tensor<T, DeviceFaer, DA>       ] [Tensor<T, DeviceFaer, DB>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceFaer> for (TrA, TrB, Option<FlagUpLo>)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = TrB;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        let a_dyn = a.to_dyn();
        let b_ref = b.view_mut().into_dim::<IxD>();
        let result = faer_impl_solve_triangular_f(a_dyn, b_ref.into(), uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct             args_tuple     internal_tuple     ;
   [(TrA, TrB, FlagUpLo)] [(a, b, uplo)] [(a, b, Some(uplo))];
   [(TrA, TrB,         )] [(a, b,     )] [(a, b, None      )];
)]
impl<TrA, TrB> SolveTriangularAPI<DeviceFaer> for ImplStruct
where
    (TrA, TrB, Option<FlagUpLo>): SolveTriangularAPI<DeviceFaer>,
{
    type Out = <(TrA, TrB, Option<FlagUpLo>) as SolveTriangularAPI<DeviceFaer>>::Out;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        SolveTriangularAPI::<DeviceFaer>::solve_triangular_f(internal_tuple)
    }
}

/* #endregion */
