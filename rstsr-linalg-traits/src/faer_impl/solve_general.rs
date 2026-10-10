use crate::traits_def::SolveGeneralAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_blas_traits::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_solve_general_f<'b, T>(
    a: TensorReference<'_, T, DeviceFaer, Ix2>,
    b: TensorReference<'b, T, DeviceFaer, Ix2>,
) -> Result<TensorMutable<'b, T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    // set parallel mode
    let device = a.device().clone();
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    if let Some(pool) = pool {
        faer::set_global_parallelism(Par::rayon(pool.current_num_threads()));
    }

    let faer_a = a.view().into_faer();

    // solve linear system
    let svd_result = faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SVD error: {e:?}"))?;

    // handle b for mutable
    let mut b = overwritable_convert(b)?;
    let b_view = b.view_mut().into_dim::<Ix2>();
    let faer_b = b_view.into_faer();

    svd_result.solve_in_place(faer_b);

    // restore parallel mode
    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    Ok(b.clone_to_mut())
}

/// Solve a stack of linear systems `a x = b`.
///
/// `a` is `(..., M, M)` and `b` is `(..., M)` (a stack of vectors) or
/// `(..., M, K)`; the batch dims of `a` and `b` are broadcast against each other.
/// A 1-D `b` is treated as an `(..., M, 1)` stack and the trailing axis is
/// dropped on output, giving `(..., M)`.
fn faer_impl_solve_general_nd_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorView<'_, T, DeviceFaer, IxD>,
) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_out, m, k, is_vec, mats) = crate::linalg_util::map_batch_solve(a, b, order, &mut |a2, b2| {
        Ok(faer_impl_solve_general_f(a2.into(), b2.into())?.into_owned())
    })?;

    let result = crate::linalg_util::assemble_batch_matrices_f(mats, &batch_out, &[m, k], order, &device)?;
    if is_vec {
        let mut shape = batch_out;
        shape.push(m);
        Ok(result.into_shape(shape))
    } else {
        Ok(result)
    }
}

/// n-dim in-place `solve`.
///
/// `a` and `b` share one batch shape (no broadcast) and every slice of `b` is
/// overwritten by its solution — neither operand is copied and the output buffer
/// is `b` itself.
fn faer_impl_solve_general_inplace_nd_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorMut<'_, T, DeviceFaer, IxD>,
) -> Result<()>
where
    T: ComplexField,
{
    let order = a.device().default_order();
    let (batch_a, [m, m2]) = crate::linalg_util::batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, m2, InvalidLayout, "solve: matrix a must be square, got {m}x{m2}")?;
    let (batch_b, [bm, _bk]) = crate::linalg_util::batch_and_matrix_shape(b.shape(), order)?;
    rstsr_assert_eq!(bm, m, InvalidLayout, "solve: b's row dimension must match a")?;
    rstsr_assert_eq!(batch_a, batch_b, InvalidLayout, "solve: in-place batching requires matching batch dims")?;
    let mut out = Vec::new();
    crate::linalg_util::map_batch_matrices2_mut(
        a,
        b,
        order,
        &mut |a2, b2| {
            faer_impl_solve_general_f(a2.into(), b2.into())?;
            Ok(())
        },
        &mut out,
    )
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
        faer_impl_solve_general_nd_f(a.to_dyn(), b.to_dyn())
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
        if a.ndim() > 2 || b.ndim() > 2 {
            let b_view = b.view_mut().into_dim::<IxD>();
            faer_impl_solve_general_inplace_nd_f(a.to_dyn(), b_view)?;
            return Ok(b);
        }
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
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
        faer_impl_solve_general_nd_f(a.to_dyn(), b.to_dyn())
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
        let (mut a, mut b) = self;
        if a.ndim() > 2 || b.ndim() > 2 {
            let b_view = b.view_mut().into_dim::<IxD>();
            faer_impl_solve_general_inplace_nd_f(a.to_dyn(), b_view)?;
            return Ok(b);
        }
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}
