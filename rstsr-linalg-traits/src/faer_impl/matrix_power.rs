//! Matrix power on a stack of square matrices (faer device).
//!
//! Mirrors NumPy's `matrix_power`: `n = 0` yields the identity, `n < 0`
//! inverts the base once, then the base is raised to `|n|` by
//! exponentiation-by-squaring over the bits of `|n|` (LSB to MSB). The inverse
//! is the faer one, which does not screen for singularity, so `n < 0` on a
//! singular base returns `inf`/`NaN` entries rather than an error.

use crate::faer_impl::batch::{batch_and_matrix_shape, map_stack_slices, stack_shape, with_parallel};
use crate::faer_impl::inv::faer_impl_inv_f;
use crate::traits_def::MatrixPowerAPI;
use faer::traits::ComplexField;
use num::Num;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_matrix_power_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    n: isize,
) -> Result<Tensor<T, DeviceFaer, IxD>>
where
    T: ComplexField + Num + Clone + Send + Sync + 'static,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, k]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, k, InvalidLayout, "matrix_power: the matrix must be square, got {m}x{k}")?;

    if n == 0 {
        return with_parallel(&device, || {
            let mut out = zeros_f((stack_shape(&batch_shape, &[m, m], order), &device))?;
            map_stack_slices::<T, T, Ix2, _>(a, out.view_mut(), order, |_a_slice, mut out_slice| {
                let eye: Tensor<T, DeviceFaer, Ix2> = eye_f((m, &device))?.into_dim::<Ix2>();
                out_slice.assign_f(eye)
            })?;
            Ok(out)
        });
    }

    // negative exponent: invert once, then raise the inverse to |n|
    let base = if n < 0 { faer_impl_inv_f(a)? } else { a.to_owned() };

    let e = n.unsigned_abs();
    match e {
        1 => Ok(base),
        2 => matmul_f(&base, &base),
        3 => {
            let base2 = matmul_f(&base, &base)?;
            matmul_f(&base2, &base)
        },
        _ => {
            // binary exponentiation over the bits of `e`, LSB to MSB: `z` holds
            // the running square, the set bits accumulate into `result`
            let mut z = base;
            let mut result: Option<Tensor<T, DeviceFaer, IxD>> = None;
            let mut e = e;
            while e > 0 {
                if e & 1 == 1 {
                    result = Some(match result {
                        Some(acc) => matmul_f(&acc, &z)?,
                        None => z.clone(),
                    });
                }
                e >>= 1;
                if e > 0 {
                    z = matmul_f(&z, &z)?;
                }
            }
            result.ok_or_else(|| rstsr_error!(RuntimeError, "matrix_power: empty exponent result"))
        },
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> MatrixPowerAPI<DeviceFaer> for (Tr, isize)
where
    T: ComplexField + Num + Clone + Send + Sync + 'static,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn matrix_power_f(self) -> Result<Self::Out> {
        let (a, n) = self;
        let result = faer_impl_matrix_power_f(a.view().to_dyn(), n)?;
        Ok(result.into_dim::<D>())
    }
}
