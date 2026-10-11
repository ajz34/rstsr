use crate::faer_impl::batch::batch_and_matrix_shape;
use crate::faer_impl::batch::{map_stack_slices2, map_stack_slices2x2, stack_shape, with_parallel};
use crate::traits_def::{EighAPI, EighResult};
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Num;
use rstsr_core::prelude_dev::*;
use rstsr_dtype_traits::ExtNum;

/// Eigendecomposition of a single 2-D symmetric/Hermitian matrix (no
/// parallel-mode handling).
fn faer_eigh_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<(Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_a = a.into_faer();
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    let result = faer_a
        .self_adjoint_eigen(faer_uplo)
        .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;

    // convert eigenvalues to real
    let eigenvalues: TensorView<T, DeviceFaer, _> = result.S().column_vector().into_rstsr();
    let eigenvalues = eigenvalues.mapv(|v| T::real_part_impl(&v));
    let eigenvectors = result.U().into_rstsr().into_contig(device.default_order());

    Ok((eigenvalues, eigenvectors))
}

pub fn faer_impl_standard_eigh_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<(Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>)>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "eigh: the matrix must be square, got {m}x{n}")?;

    with_parallel(&device, || {
        let mut w = zeros_f((stack_shape(&batch_shape, &[m], order), &device))?;
        let mut v = zeros_f((stack_shape(&batch_shape, &[m, m], order), &device))?;
        map_stack_slices2::<T, T::Real, Ix1, T, Ix2, _>(
            a,
            w.view_mut(),
            v.view_mut(),
            order,
            |a_slice, mut w_slice, mut v_slice| {
                let (ew, ev) = faer_eigh_ix2(a_slice, uplo)?;
                w_slice.assign_f(ew)?;
                v_slice.assign_f(ev)
            },
        )?;
        Ok((w, v))
    })
}

/// n-dim generalized `eigh`: solve `a v = λ b v` for a stack of `(..., M, M)`
/// symmetric/Hermitian `a` and positive-definite `b`; the two batch shapes
/// broadcast against each other.
pub fn faer_impl_generalized_eigh_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    b: TensorView<'_, T, DeviceFaer, IxD>,
    uplo: Option<FlagUpLo>,
    itype: i32,
) -> Result<(Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>)>
where
    T: ComplexField + Num,
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    rstsr_pattern!(itype, 1..=3, InvalidValue, "itype in generalized eigen must be 1, 2 or 3")?;

    let device = a.device().clone();
    let order = device.default_order();
    let (batch_a, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "generalized eigh: matrix a must be square, got {m}x{n}")?;
    let (batch_b, [bm, bn]) = batch_and_matrix_shape(b.shape(), order)?;
    rstsr_assert_eq!(
        (bm, bn),
        (m, n),
        InvalidLayout,
        "generalized eigh: a and b must share the matrix shape, got {m}x{n} vs {bm}x{bn}"
    )?;
    let batch_out = broadcast_shapes_f(&[batch_a, batch_b], order)?;

    let mut full = batch_out.clone();
    full.extend_from_slice(&[m, m]);
    let a_b = a.broadcast_to(full.clone());
    let b_b = b.broadcast_to(full);
    let faer_par = device.get_current_pool().map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));

    with_parallel(&device, || {
        let mut w = zeros_f((stack_shape(&batch_out, &[m], order), &device))?;
        let mut v = zeros_f((stack_shape(&batch_out, &[m, m], order), &device))?;
        map_stack_slices2x2::<T, T, T::Real, Ix1, T, Ix2, _>(
            a_b,
            b_b,
            w.view_mut(),
            v.view_mut(),
            order,
            |a_slice, b_slice, mut w_slice, mut v_slice| {
                let (ew, ev) = faer_generalized_eigh_ix2(a_slice, b_slice, uplo, itype, faer_par)?;
                w_slice.assign_f(ew)?;
                v_slice.assign_f(ev)
            },
        )?;
        Ok((w, v))
    })
}

/// Eigendecomposition of a single 2-D generalized symmetric/Hermitian matrix
/// (no parallel-mode handling; `faer_par` is threaded from the batched caller).
fn faer_generalized_eigh_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    b: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
    itype: i32,
    faer_par: Par,
) -> Result<(Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    // perform symmetrize on matrix a
    // TODO: a better implementation
    let mut a = a.to_owned();
    let n = a.nrow();
    match uplo {
        Lower => {
            for i in 0..n {
                for j in 0..i {
                    a[[j, i]] = T::conj_impl(&a[[i, j]]);
                }
                a[[i, i]] = T::from_real_impl(&T::real_part_impl(&a[[i, i]]));
            }
        },
        Upper => {
            for i in 0..n {
                for j in (i + 1)..n {
                    a[[j, i]] = T::conj_impl(&a[[i, j]]);
                }
                a[[i, i]] = T::from_real_impl(&T::real_part_impl(&a[[i, i]]));
            }
        },
    }

    // perform cholesky on matrix b
    let b = b.into_faer();
    let b_llt = b.llt(faer_uplo).map_err(|e| rstsr_error!(FaerError, "Faer Cholesky error: {e:?}"))?;
    let l = b_llt.L();

    let result = match itype {
        1 => {
            // inv(l) @ a @ inv(l.t.conj)
            let mut a = a.view_mut().into_faer();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l.as_ref(), a.as_mut(), faer_par);
            let mut a = a.transpose_mut();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l.conjugate(), a.as_mut(), faer_par);
            let a = a.transpose_mut();

            // eig(a)
            let eig_a = a
                .self_adjoint_eigen(faer_uplo)
                .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;
            let e = eig_a.S().column_vector().into_rstsr().mapv(|v| T::real_part_impl(&v));

            // inv(l.t.conj) @ c
            let mut c = eig_a.U().to_owned();
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l.adjoint(), c.as_mut(), faer_par);
            let c = c.into_rstsr().into_contig(device.default_order());

            (e, c)
        },
        2 | 3 => {
            // inv(l)
            let mut l_inv = Mat::identity(n, n);
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l.as_ref(), l_inv.as_mut(), faer_par);

            // l.t.conj @ a @ l
            let mut a = a.view_mut().into_faer();
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l_inv.adjoint(), a.as_mut(), faer_par);
            let mut a = a.transpose_mut();
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l_inv.transpose(), a.as_mut(), faer_par);
            let a = a.transpose_mut();

            // eig(a)
            let eig_a = a
                .self_adjoint_eigen(faer_uplo)
                .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;
            let e = eig_a.S().column_vector().into_rstsr().mapv(|v| T::real_part_impl(&v));

            let mut c = eig_a.U().to_owned();
            match itype {
                2 => faer::linalg::triangular_solve::solve_upper_triangular_in_place(l.adjoint(), c.as_mut(), faer_par),
                3 => faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    l_inv.as_ref(),
                    c.as_mut(),
                    faer_par,
                ),
                _ => unreachable!(),
            };
            let c = c.into_rstsr().into_contig(device.default_order());

            (e, c)
        },
        _ => unreachable!(),
    };

    Ok(result)
}

/* #region standard eigh */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField + Num,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        let (vals, vecs) = faer_impl_standard_eigh_f(a.view().to_dyn(), uplo)?;
        Ok(EighResult { eigenvalues: vals.into_dim::<D::SmallerOne>(), eigenvectors: vecs.into_dim::<D>() })
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField + ExtNum<AbsOut = T::Real> + Num,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        EighAPI::<DeviceFaer>::eigh_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for Tr
where
    T: ComplexField + ExtNum<AbsOut = T::Real> + Num,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let a = self;
        EighAPI::<DeviceFaer>::eigh_f((a, None))
    }
}

/* #endregion */

/* #region generalized eig */

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> EighAPI<DeviceFaer> for (TrA, TrB, FlagUpLo, i32)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo, eig_type) = self;
        let (vals, vecs) = faer_impl_generalized_eigh_f(a.to_dyn(), b.to_dyn(), Some(uplo), eig_type)?;
        Ok(EighResult { eigenvalues: vals, eigenvectors: vecs })
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> EighAPI<DeviceFaer> for (TrA, TrB, FlagUpLo)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        EighAPI::<DeviceFaer>::eigh_f((a, b, uplo, 1))
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> EighAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EighAPI::<DeviceFaer>::eigh_f((a, b, uplo, 1))
    }
}

/* #endregion */
