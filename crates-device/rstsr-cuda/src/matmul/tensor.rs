//! The tensor-level matmul entries that pass the compute mode per call.

use super::mode::CudaMatMulMode;
use crate::prelude_dev::*;

/// [`matmul`] and its GEMM forms with the compute mode of the **output** given per call.
///
/// `rstsr-core` cannot name the CUDA modes, so the entries that take one live here; they follow
/// the core functions rule for rule, and read nothing from the device flags. The mode is that of
/// the **output** type: `<TC as CudaMatMulMode>::Mode`.
///
/// # Panics
///
/// As [`matmul`], [`matmul_with_output`] and [`matmul_from`]: on shapes that do not follow the
/// rules, on mismatched devices, and (like their `.rstsr_unwrap()`) on an error of the device.
pub trait TensorCudaMatMulAPI<R, T, D>
where
    T: CudaMatMulMode + num::Zero,
    D: DimAPI,
    R: DataAPI<Data = CudaBuffer<T>>,
{
    /// [`matmul`] under `mode`, allocating the output: `TC` is named by the turbofish
    /// (`a.matmul_with_mode::<TB, TC, _, _>(&b, mode)`).
    fn matmul_with_mode<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        mode: <TC as CudaMatMulMode>::Mode,
    ) -> Tensor<TC, DeviceCuda, DC>
    where
        TB: CudaMatMulMode + num::Zero,
        TC: CudaMatMulMode + num::Zero + num::One,
        DB: DimAPI,
        DC: DimAPI,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>;

    /// [`matmul_with_output`] under `mode` (the mode of the output `c`).
    fn matmul_with_output_mode<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        c: impl TensorViewMutAPI<Type = TC, Backend = DeviceCuda, Dim = DC>,
        mode: <TC as CudaMatMulMode>::Mode,
    ) where
        TB: CudaMatMulMode + num::Zero,
        TC: CudaMatMulMode + num::Zero + num::One,
        DB: DimAPI,
        DC: DimAPI;

    /// [`matmul_from`] under `mode` (this tensor is the output `c`):
    /// `self = alpha * (a @ b) + beta * self`.
    fn matmul_from_mode<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        alpha: T,
        beta: T,
        mode: <T as CudaMatMulMode>::Mode,
    ) where
        TA: CudaMatMulMode + num::Zero,
        TB: CudaMatMulMode + num::Zero,
        DA: DimAPI,
        DB: DimAPI,
        R: DataMutAPI<Data = CudaBuffer<T>>;
}

impl<R, T, D> TensorCudaMatMulAPI<R, T, D> for TensorAny<R, T, DeviceCuda, D>
where
    T: CudaMatMulMode + num::Zero,
    D: DimAPI,
    R: DataAPI<Data = CudaBuffer<T>>,
{
    fn matmul_with_mode<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        mode: <TC as CudaMatMulMode>::Mode,
    ) -> Tensor<TC, DeviceCuda, DC>
    where
        TB: CudaMatMulMode + num::Zero,
        TC: CudaMatMulMode + num::Zero + num::One,
        DB: DimAPI,
        DC: DimAPI,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
    {
        matmul_with_mode_f(self.view(), rhs, TC::one(), mode).rstsr_unwrap()
    }

    fn matmul_with_output_mode<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        c: impl TensorViewMutAPI<Type = TC, Backend = DeviceCuda, Dim = DC>,
        mode: <TC as CudaMatMulMode>::Mode,
    ) where
        TB: CudaMatMulMode + num::Zero,
        TC: CudaMatMulMode + num::Zero + num::One,
        DB: DimAPI,
        DC: DimAPI,
    {
        with_mode(c, self.view(), rhs, TC::one(), TC::zero(), mode).rstsr_unwrap()
    }

    fn matmul_from_mode<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        alpha: T,
        beta: T,
        mode: <T as CudaMatMulMode>::Mode,
    ) where
        TA: CudaMatMulMode + num::Zero,
        TB: CudaMatMulMode + num::Zero,
        DA: DimAPI,
        DB: DimAPI,
        R: DataMutAPI<Data = CudaBuffer<T>>,
    {
        with_mode(self.view_mut(), a, b, alpha, beta, mode).rstsr_unwrap()
    }
}

/// The fallible free function of [`TensorCudaMatMulAPI::matmul_with_mode`].
pub fn matmul_with_mode_f<TA, TB, TC, DA, DB, DC>(
    a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
    alpha: TC,
    mode: <TC as CudaMatMulMode>::Mode,
) -> Result<Tensor<TC, DeviceCuda, DC>>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: CudaMatMulMode + num::Zero,
    TB: CudaMatMulMode + num::Zero,
    TC: CudaMatMulMode + num::Zero,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
{
    let (a, b) = (a.view(), b.view());
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let cfg = LayoutMatMulConfig::<DA, DB>::layout_matmul(a.layout(), b.layout(), a.device().default_order())?;
    let lc = cfg.lc;
    // the fresh-output path of `rstsr_core`'s `op_refa_refb_matmul`
    let device = a.device().clone();
    let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
    crate::matmul::inner::matmul_with_mode_inner(
        &device,
        storage_c.raw_mut(),
        &lc,
        a.raw(),
        a.layout(),
        b.raw(),
        b.layout(),
        alpha,
        <TC as num::Zero>::zero(),
        mode,
    )?;
    // SAFETY: `matmul_with_mode_inner` initialized every element of `lc` (`beta == 0`: the whole
    // of the `matmul_uninit` contract it implements).
    let storage_c = unsafe { DeviceCuda::assume_init_impl(storage_c)? };
    Tensor::new_f(storage_c, lc)
}

/// The free function of the GEMM forms under a mode.
pub(crate) fn with_mode<TA, TB, TC, DA, DB, DC>(
    mut c: impl TensorViewMutAPI<Type = TC, Backend = DeviceCuda, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
    alpha: TC,
    beta: TC,
    mode: <TC as CudaMatMulMode>::Mode,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: CudaMatMulMode + num::Zero,
    TB: CudaMatMulMode + num::Zero,
    TC: CudaMatMulMode + num::Zero,
{
    let (a, b, mut c) = (a.view(), b.view(), c.view_mut());
    rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
    // writing through a broadcast layout would alias elements
    rstsr_assert!(!c.layout().is_broadcasted(), InvalidLayout, "cannot matmul into broadcasted tensor")?;
    let device = c.device().clone();
    let lc = c.layout().clone();
    crate::matmul::inner::matmul_with_mode_inner(
        &device,
        c.raw_mut(),
        &lc,
        a.raw(),
        a.layout(),
        b.raw(),
        b.layout(),
        alpha,
        beta,
        mode,
    )
}
