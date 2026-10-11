use crate::faer_impl::batch::{batch_and_matrix_shape, map_stack_scalars2, with_parallel};
use crate::traits_def::{SLogDetAPI, SLogDetResult};
use faer::traits::{ComplexField, IndexCore};
use faer_ext::IntoFaer;

use num::complex::ComplexFloat;
use num::Zero;
use rstsr_core::prelude_dev::*;

// Neutral bounds (no BlasFloat): ComplexFloat provides Num (one/zero/*/-) and
// `Real: Float` (ln); ComplexField is faer's own. The bridge keeps both `Real`s equal.

/// `slogdet` of a single 2-D matrix (LU with partial pivoting); no parallel-mode handling.
fn faer_slogdet_ix2<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<(T, <T as ComplexField>::Real)>
where
    T: ComplexFloat + ComplexField<Real = <T as ComplexFloat>::Real>,
    <T as ComplexField>::Real: Zero,
{
    let faer_a = a.into_faer();

    // LU factorization with partial (row) pivoting: P A = L U. The U diagonal
    // carries the magnitude, the row permutation carries the sign.
    let lu = faer_a.partial_piv_lu();
    let u = lu.U();
    let n = u.nrows();

    let mut sign = T::one();
    let mut logabsdet = <T as ComplexField>::Real::zero();
    for i in 0..n {
        let diag = u[(i, i)];
        let mag: <T as ComplexField>::Real = T::abs_impl(&diag);
        logabsdet += mag.ln();
        // a zero pivot (singular matrix) drives the sign to zero, as NumPy does
        let phase = if mag == <T as ComplexField>::Real::zero() { T::zero() } else { diag / T::from_real_impl(&mag) };
        sign *= phase;
    }

    // det(P) = (-1)^(number of transpositions), and transpositions = n - cycles
    let (forward, _inverse) = lu.P().arrays();
    let mut visited = vec![false; n];
    let mut n_cycles = 0;
    for i in 0..n {
        if !visited[i] {
            n_cycles += 1;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = forward[j].zx();
            }
        }
    }
    if (n - n_cycles) % 2 == 1 {
        sign = T::zero() - sign;
    }

    Ok((sign, logabsdet))
}

/// n-dim `slogdet` over the batch dims.
///
/// The two matrix axes are the last two under `RowMajor` and the first two
/// under `ColMajor` (device default order); all remaining dims form the batch.
/// The outputs (`sign`, `logabsdet`) have the batch shape.
pub fn faer_impl_slogdet_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
) -> Result<(Tensor<T, DeviceFaer, IxD>, Tensor<<T as ComplexField>::Real, DeviceFaer, IxD>)>
where
    T: ComplexFloat + ComplexField<Real = <T as ComplexFloat>::Real>,
    <T as ComplexField>::Real: Zero,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "slogdet: the matrix must be square, got {m}x{n}")?;

    with_parallel(&device, || {
        let mut sign = zeros_f((batch_shape.clone(), &device))?;
        let mut logabsdet = zeros_f((batch_shape.clone(), &device))?;
        map_stack_scalars2(a, sign.view_mut(), logabsdet.view_mut(), order, |a_slice, s, l| {
            let (sv, lv) = faer_slogdet_ix2(a_slice)?;
            *s = sv;
            *l = lv;
            Ok(())
        })?;
        Ok((sign, logabsdet))
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   ['a, T, D                       ] [TensorMut<'a, T, DeviceFaer, D> ];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> SLogDetAPI<DeviceFaer> for Tr
where
    T: ComplexFloat + ComplexField<Real = <T as ComplexFloat>::Real>,
    <T as ComplexField>::Real: Zero,
    D: DimAPI,
{
    type Out = SLogDetResult<Tensor<T, DeviceFaer, IxD>, Tensor<<T as ComplexField>::Real, DeviceFaer, IxD>>;
    fn slogdet_f(self) -> Result<Self::Out> {
        let a = self;
        let a_view = a.to_dyn();
        let (sign, logabsdet) = faer_impl_slogdet_f(a_view)?;
        Ok(SLogDetResult { sign, logabsdet })
    }
}
