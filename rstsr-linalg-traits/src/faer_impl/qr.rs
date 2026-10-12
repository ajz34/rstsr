//! QR decomposition on the faer device (the array-API `linalg.qr`).
//!
//! `mode` selects the array-API `'reduced'` (default) or `'complete'`
//! factorization. `Q` and `R` keep the input's dtype; with `K = min(M, N)`
//! their trailing shapes are `(M, K)`/`(K, N)` for `'reduced'` and
//! `(M, M)`/`(M, N)` for `'complete'`. The decomposition is not unique — each
//! column of `Q` is fixed only up to a unit factor — so callers compare results
//! after canonicalizing the phase of `R`'s diagonal.

use crate::faer_impl::batch::{batch_and_matrix_shape, map_stack_slices2, stack_shape, with_parallel};
use crate::traits_def::{QrAPI, QrMode, QrResult};
use faer::traits::ComplexField;
use faer::Mat;
use faer_ext::IntoFaer;
use num::Num;
use rstsr_core::prelude_dev::*;

/// QR factorization of a single 2-D matrix (no parallel-mode handling).
fn faer_qr_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    mode: QrMode,
) -> Result<(Tensor<T, DeviceFaer, Ix2>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let [m, n] = *a.shape();
    let k = usize::min(m, n);

    let qr = a.into_faer().qr();
    let (q, r) = match mode {
        QrMode::Reduced => (qr.compute_thin_Q(), qr.R().to_owned()),
        QrMode::Complete => {
            // faer stores only the (k, n) thin R; the full (m, n) R zero-pads
            // the rows below the k-th
            let mut r = Mat::<T>::zeros(m, n);
            r.submatrix_mut(0, 0, k, n).copy_from(qr.R());
            (qr.compute_Q(), r)
        },
    };

    // `into_rstsr` homes the results on `DeviceFaer::default()`, so each device
    // must be changed back to the input's
    let q = q.into_rstsr().into_contig(device.default_order()).change_device_f(&device)?;
    let r = r.into_rstsr().into_contig(device.default_order()).change_device_f(&device)?;
    Ok((q, r))
}

pub fn faer_impl_qr_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    mode: QrMode,
) -> Result<QrResult<Tensor<T, DeviceFaer, IxD>, Tensor<T, DeviceFaer, IxD>>>
where
    T: ComplexField + Num,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    let k = usize::min(m, n);
    // Q is (m, q_cols), R is (r_rows, n)
    let (q_cols, r_rows) = match mode {
        QrMode::Reduced => (k, k),
        QrMode::Complete => (m, m),
    };

    // a zero-size matrix axis or batch yields empty outputs (faer is not called)
    if m == 0 || n == 0 || batch_shape.contains(&0) {
        let q = zeros_f((stack_shape(&batch_shape, &[m, q_cols], order), &device))?;
        let r = zeros_f((stack_shape(&batch_shape, &[r_rows, n], order), &device))?;
        return Ok(QrResult { q, r });
    }

    with_parallel(&device, || {
        let mut q = zeros_f((stack_shape(&batch_shape, &[m, q_cols], order), &device))?;
        let mut r = zeros_f((stack_shape(&batch_shape, &[r_rows, n], order), &device))?;
        map_stack_slices2::<T, T, Ix2, T, Ix2, _>(
            a,
            q.view_mut(),
            r.view_mut(),
            order,
            |a_slice, mut q_slice, mut r_slice| {
                let (qi, ri) = faer_qr_ix2(a_slice, mode)?;
                q_slice.assign_f(qi)?;
                r_slice.assign_f(ri)
            },
        )?;
        Ok(QrResult { q, r })
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> QrAPI<DeviceFaer> for (Tr, QrMode)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = QrResult<Tensor<T, DeviceFaer, D>, Tensor<T, DeviceFaer, D>>;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode) = self;
        let result = faer_impl_qr_f(a.view().to_dyn(), mode)?;
        Ok(QrResult { q: result.q.into_dim::<D>(), r: result.r.into_dim::<D>() })
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> QrAPI<DeviceFaer> for (Tr, &str)
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = QrResult<Tensor<T, DeviceFaer, D>, Tensor<T, DeviceFaer, D>>;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode) = self;
        QrAPI::<DeviceFaer>::qr_f((a, QrMode::try_from(mode)?))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> QrAPI<DeviceFaer> for Tr
where
    T: ComplexField + Num,
    D: DimAPI,
{
    type Out = QrResult<Tensor<T, DeviceFaer, D>, Tensor<T, DeviceFaer, D>>;
    fn qr_f(self) -> Result<Self::Out> {
        QrAPI::<DeviceFaer>::qr_f((self, QrMode::Reduced))
    }
}
