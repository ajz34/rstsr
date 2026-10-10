use crate::traits_def::SolveTriangularAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_blas_traits::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_solve_triangular_f<'b, T>(
    a: TensorReference<'_, T, DeviceFaer, Ix2>,
    b: TensorReference<'b, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<'b, T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    // set parallel mode
    let device = a.device().clone();
    let pool = device.get_current_pool();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));

    let uplo = uplo.unwrap_or_else(|| match device.default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_a = a.view().into_faer();
    let mut b = overwritable_convert(b)?;
    let faer_b = b.view_mut().into_faer();

    match uplo {
        Lower => faer::linalg::triangular_solve::solve_lower_triangular_in_place(faer_a, faer_b, faer_par),
        Upper => faer::linalg::triangular_solve::solve_upper_triangular_in_place(faer_a, faer_b, faer_par),
    }

    Ok(b.clone_to_mut())
}

/// Solve a stack of triangular systems `a x = b` (see [`faer_impl_solve_triangular_f`]).
fn faer_impl_solve_triangular_nd_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorView<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let order = a.device().default_order();
    crate::linalg_util::map_batch_solve_into_output(a, b, order, &mut |a2, b2| {
        faer_impl_solve_triangular_f(a2.into(), b2.into(), uplo)?;
        Ok(())
    })
}

/// n-dim in-place triangular solve: each slice of `b` is overwritten in place.
fn faer_impl_solve_triangular_inplace_nd_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorMut<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<()>
where
    T: ComplexField,
{
    let order = a.device().default_order();
    crate::linalg_util::map_batch_solve_inplace(a, b, order, &mut |a2, b2| {
        faer_impl_solve_triangular_f(a2.into(), b2.into(), uplo)?;
        Ok(())
    })
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
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        faer_impl_solve_triangular_nd_f(a.to_dyn(), b.to_dyn(), uplo)
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
        if a.ndim() > 2 || b.ndim() > 2 {
            let b_view = b.view_mut().into_dim::<IxD>();
            faer_impl_solve_triangular_inplace_nd_f(a.to_dyn(), b_view, uplo)?;
            return Ok(b);
        }
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
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
    type Out = Tensor<T, DeviceFaer, IxD>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        faer_impl_solve_triangular_nd_f(a.to_dyn(), b.to_dyn(), uplo)
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
        let (mut a, mut b, uplo) = self;
        if a.ndim() > 2 || b.ndim() > 2 {
            let b_view = b.view_mut().into_dim::<IxD>();
            faer_impl_solve_triangular_inplace_nd_f(a.to_dyn(), b_view, uplo)?;
            return Ok(b);
        }
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
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
