//! Matrix and vector norms on the faer device (the array-API `linalg` norms).
//!
//! `matrix_norm` follows NumPy: the matrix axes are the device-order pair (the
//! last two under `RowMajor`, the first two under `ColMajor`); `'fro'` is the L2
//! norm over both axes, `1`/`-1`/`inf`/`-inf` are abs -> sum -> max/min, and
//! `2`/`-2`/`'nuc'` are the max/min/sum of the singular values (via
//! [`faer_impl_svdvals_f`]). `vector_norm` reduces `|x|` over the requested axes
//! with the L2 reduction for `p = 2`, a plain sum for `p = 1`, a count of
//! nonzeros for `p = 0`, and the general `(sum |x|^p)^(1/p)` otherwise. Empty
//! normed axes yield `0` for every order.
//!
//! `TR` is the real scalar (`T`'s `ComplexField::Real`, equal to `ExtNum::AbsOut`);
//! tying both projections to `TR` lets the faer entries and the core reductions
//! share one element type.

use crate::faer_impl::batch::{batch_and_matrix_shape, with_parallel};
use crate::faer_impl::svdvals::faer_impl_svdvals_f;
use crate::traits_def::{MatrixNormAPI, MatrixNormArgs, NormOrd, VectorNormAPI, VectorNormArgs};
use faer::traits::ComplexField;
use num::complex::ComplexFloat;
use num::{FromPrimitive, Num};
use rstsr_core::prelude_dev::*;

/// The real scalar shared by the faer linalg entries and the core reductions:
/// `T`'s real type, usable as an element of the reduction and `pow` ops.
pub trait NormReal:
    ExtNum<AbsOut = Self> + ExtReal + ComplexFloat<Real = Self> + FromPrimitive + Num + Send + Sync + 'static
{
}

impl<R> NormReal for R where
    R: ExtNum<AbsOut = Self> + ExtReal + ComplexFloat<Real = Self> + FromPrimitive + Num + Send + Sync + 'static
{
}

/// Output shape of a matrix norm: the batch shape, with the two matrix axes
/// re-inserted as singletons under `keepdims` (trailing under `RowMajor`,
/// leading under `ColMajor`).
fn matrix_out_shape(batch_shape: &[usize], order: FlagOrder, keepdims: bool) -> Vec<usize> {
    let ones = [1usize, 1];
    match order {
        RowMajor if keepdims => [batch_shape, &ones].concat(),
        ColMajor if keepdims => [&ones[..], batch_shape].concat(),
        _ => batch_shape.to_vec(),
    }
}

/// Mark `axis` (possibly negative) as reduced; out-of-range values are ignored
/// (the reduction itself reports those).
fn set_axis(is_reduced: &mut [bool], axis: isize, ndim: usize) {
    let i = if axis < 0 { axis + ndim as isize } else { axis };
    if i >= 0 && (i as usize) < ndim {
        is_reduced[i as usize] = true;
    }
}

pub fn faer_impl_matrix_norm_f<T, TR>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    ord: NormOrd,
    keepdims: bool,
) -> Result<Tensor<TR, DeviceFaer, IxD>>
where
    T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + Clone + Send + Sync + 'static,
    TR: NormReal + DTypePromoteAPI<TR, Res = TR>,
{
    let ord = ord.resolve_matrix()?;
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;

    // an empty input yields an empty (or, when only a matrix axis vanished, all
    // zero) result: NumPy's `initial=0`, and `max`/`min` reject a zero-size input
    if m == 0 || n == 0 || batch_shape.contains(&0) {
        let out: Tensor<TR, DeviceFaer, IxD> = zeros_f((matrix_out_shape(&batch_shape, order, keepdims), &device))?;
        return Ok(out);
    }

    // matrix axis indices, and the surviving axis after the first reduction
    let (row_ax, col_ax, after_row, after_col, k_ax): (isize, isize, isize, isize, isize) = match order {
        RowMajor => (-2, -1, -1, -1, -1),
        ColMajor => (0, 1, 0, 0, 0),
    };

    let reduced = with_parallel(&device, || match ord {
        NormOrd::Fro => l2_norm_with_args_f(abs_f(&a)?, ([row_ax, col_ax], false)),
        NormOrd::Nuc => sum_with_args_f(faer_impl_svdvals_f(a)?, (k_ax, false)),
        NormOrd::Inf => {
            let t = sum_with_args_f(abs_f(&a)?, (col_ax, false))?;
            max_with_args_f(&t, (after_col, false))
        },
        NormOrd::NegInf => {
            let t = sum_with_args_f(abs_f(&a)?, (col_ax, false))?;
            min_with_args_f(&t, (after_col, false))
        },
        // the numeric orders a matrix has a norm for: ±1 (abs -> sum -> max/min)
        // and ±2 (largest/smallest singular value); `resolve_matrix` fixed the set
        NormOrd::P(p) => {
            if p == 1.0 {
                let t = sum_with_args_f(abs_f(&a)?, (row_ax, false))?;
                max_with_args_f(&t, (after_row, false))
            } else if p == -1.0 {
                let t = sum_with_args_f(abs_f(&a)?, (row_ax, false))?;
                min_with_args_f(&t, (after_row, false))
            } else if p == 2.0 {
                max_with_args_f(faer_impl_svdvals_f(a)?, (k_ax, false))
            } else if p == -2.0 {
                min_with_args_f(faer_impl_svdvals_f(a)?, (k_ax, false))
            } else {
                Err(rstsr_error!(InvalidValue, "matrix_norm: invalid norm order"))
            }
        },
        _ => Err(rstsr_error!(InvalidValue, "matrix_norm: invalid norm order")),
    })?;

    if keepdims {
        let shape = matrix_out_shape(&batch_shape, order, true).into_iter().map(|v| v as isize).collect::<Vec<_>>();
        Ok(reduced.into_shape_f(shape)?)
    } else {
        Ok(reduced)
    }
}

pub fn faer_impl_vector_norm_f<T, TR>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    ord: NormOrd,
    axes: AxesIndex<isize>,
    keepdims: bool,
) -> Result<Tensor<TR, DeviceFaer, IxD>>
where
    T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + Clone + Send + Sync + 'static,
    TR: NormReal + DTypePromoteAPI<TR, Res = TR>,
{
    let ord = ord.resolve_vector()?;
    let device = a.device().clone();

    // an empty input yields an empty (or, when only a reduced axis vanished, all
    // zero) result: NumPy's `initial=0`, and `max`/`min` reject a zero-size input
    let ndim = a.ndim();
    let mut is_reduced = vec![false; ndim];
    match &axes {
        AxesIndex::None => is_reduced.iter_mut().for_each(|r| *r = true),
        AxesIndex::Val(v) => set_axis(&mut is_reduced, *v, ndim),
        AxesIndex::Vec(vs) => vs.iter().for_each(|v| set_axis(&mut is_reduced, *v, ndim)),
    }
    if a.shape().contains(&0) {
        let out_shape = is_reduced
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| (!r).then(|| a.shape()[i]).or_else(|| keepdims.then_some(1)))
            .collect::<Vec<_>>();
        let out: Tensor<TR, DeviceFaer, IxD> = zeros_f((out_shape, &device))?;
        return Ok(out);
    }

    let abs_a = abs_f(&a)?;

    with_parallel(&device, || match ord {
        NormOrd::Inf => max_with_args_f(&abs_a, (axes.clone(), keepdims)),
        NormOrd::NegInf => min_with_args_f(&abs_a, (axes.clone(), keepdims)),
        NormOrd::P(p) => {
            // p = 2 (the default) is the L2 reduction; p = 1 a plain sum; p = 0 a
            // count of nonzeros (`sign(|x|)` is 1 exactly on nonzero entries);
            // anything else is the general (sum |x|^p)^(1/p)
            if p == 2.0 {
                l2_norm_with_args_f(&abs_a, (axes.clone(), keepdims))
            } else if p == 1.0 {
                sum_with_args_f(&abs_a, (axes.clone(), keepdims))
            } else if p == 0.0 {
                sum_with_args_f(sign_f(&abs_a)?, (axes.clone(), keepdims))
            } else {
                let out_of_range = || rstsr_error!(InvalidValue, "vector_norm: order {p} out of range");
                let exp = TR::from_f64(p).ok_or_else(out_of_range)?;
                let inv_exp = TR::from_f64(1.0 / p).ok_or_else(out_of_range)?;
                let t = pow_f(&abs_a, exp)?;
                let s = sum_with_args_f(&t, (axes.clone(), keepdims))?;
                pow_f(&s, inv_exp)
            }
        },
        _ => Err(rstsr_error!(InvalidValue, "vector_norm: invalid norm order")),
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType, TR> MatrixNormAPI<DeviceFaer> for Tr
where
    T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + Clone + Send + Sync + 'static,
    TR: NormReal + DTypePromoteAPI<TR, Res = TR>,
    D: DimAPI,
{
    type Out = Tensor<TR, DeviceFaer, IxD>;
    fn matrix_norm_f(self) -> Result<Self::Out> {
        faer_impl_matrix_norm_f(self.view().to_dyn(), NormOrd::None, false)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType, TR, A> MatrixNormAPI<DeviceFaer> for (Tr, A)
where
    T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + Clone + Send + Sync + 'static,
    TR: NormReal + DTypePromoteAPI<TR, Res = TR>,
    D: DimAPI,
    A: TryInto<MatrixNormArgs>,
    A::Error: Into<Error>,
{
    type Out = Tensor<TR, DeviceFaer, IxD>;
    fn matrix_norm_f(self) -> Result<Self::Out> {
        let (a, args) = self;
        let MatrixNormArgs { ord, keepdims } = args.try_into().map_err(Into::into)?;
        faer_impl_matrix_norm_f(a.view().to_dyn(), ord, keepdims)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType, TR> VectorNormAPI<DeviceFaer> for Tr
where
    T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + Clone + Send + Sync + 'static,
    TR: NormReal + DTypePromoteAPI<TR, Res = TR>,
    D: DimAPI,
{
    type Out = Tensor<TR, DeviceFaer, IxD>;
    fn vector_norm_f(self) -> Result<Self::Out> {
        faer_impl_vector_norm_f(self.view().to_dyn(), NormOrd::None, AxesIndex::None, false)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType, TR, A> VectorNormAPI<DeviceFaer> for (Tr, A)
where
    T: ComplexField<Real = TR> + ExtNum<AbsOut = TR> + Clone + Send + Sync + 'static,
    TR: NormReal + DTypePromoteAPI<TR, Res = TR>,
    D: DimAPI,
    A: TryInto<VectorNormArgs>,
    A::Error: Into<Error>,
{
    type Out = Tensor<TR, DeviceFaer, IxD>;
    fn vector_norm_f(self) -> Result<Self::Out> {
        let (a, args) = self;
        let VectorNormArgs { ord, axes, keepdims } = args.try_into().map_err(Into::into)?;
        faer_impl_vector_norm_f(a.view().to_dyn(), ord, axes, keepdims)
    }
}
