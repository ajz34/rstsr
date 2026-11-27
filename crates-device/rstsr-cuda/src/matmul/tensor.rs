//! The tensor-level matmul entries that pass the compute mode per call.

use super::mode::CudaMatMulMode;
use crate::prelude_dev::*;

/// [`matmul`] and its GEMM forms with the compute mode of the **output** given per call.
///
/// `rstsr-core` cannot name the CUDA modes, so the entries that take one live here; they follow
/// the core functions rule for rule. Only the mode moves to the call: `matmul_drivers`,
/// `matmul_use_syrk`, `fixedpoint_mantissa_bits` and `trace_matmul` still come from the device
/// flags. The mode is that of
/// the **output** type: `<TC as CudaMatMulMode>::Mode`. `matmul_from_mode` and
/// `matmul_with_output_mode` get the output type from `c`; `matmul_with_mode` allocates it, so
/// its result waits for the type to be named: `a.matmul_with_mode(&b, mode).out::<TC>()` (the
/// output is always an `IxD` tensor). Every entry has a fallible `_f` twin that returns
/// [`Result`]; the infallible forms are its `.rstsr_unwrap()`.
///
/// # Panics
///
/// The infallible entries panic as [`matmul`], [`matmul_with_output`] and [`matmul_from`]: on
/// shapes that do not follow the rules, on mismatched devices, and on an error of the device
/// (exactly the [`Result`] their `_f` twin carries).
pub trait TensorCudaMatMulAPI<R, T, D>
where
    T: CudaMatMulMode + num::Zero,
    D: DimAPI,
    R: DataAPI<Data = CudaBuffer<T>>,
{
    /// [`matmul`] under `mode`, allocating the output: a pending product
    /// ([`WithMode`]), completed by `out::<TC>()`. The mode is read as
    /// `<TC as CudaMatMulMode>::Mode` (so it may differ from the operands' family, as for `f16`
    /// operands into an `f32` output); `TC` is the only type to name. The output is an `IxD`
    /// tensor. The operands take any tensor argument, as [`Self::matmul_with_output_mode`] does.
    fn matmul_with_mode<'a, M, TB, DB>(
        &'a self,
        rhs: &'a impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        mode: M,
    ) -> WithMode<'a, T, D, TB, DB, M>
    where
        TB: CudaMatMulMode + num::Zero,
        DB: DimAPI;

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

    /// The fallible form of [`Self::matmul_with_output_mode`].
    fn matmul_with_output_mode_f<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        c: impl TensorViewMutAPI<Type = TC, Backend = DeviceCuda, Dim = DC>,
        mode: <TC as CudaMatMulMode>::Mode,
    ) -> Result<()>
    where
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

    /// The fallible form of [`Self::matmul_from_mode`].
    fn matmul_from_mode_f<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        alpha: T,
        beta: T,
        mode: <T as CudaMatMulMode>::Mode,
    ) -> Result<()>
    where
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
    fn matmul_with_mode<'a, M, TB, DB>(
        &'a self,
        rhs: &'a impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        mode: M,
    ) -> WithMode<'a, T, D, TB, DB, M>
    where
        TB: CudaMatMulMode + num::Zero,
        DB: DimAPI,
    {
        WithMode { a: self.view(), b: rhs.view(), mode }
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
        self.matmul_with_output_mode_f(rhs, c, mode).rstsr_unwrap()
    }

    fn matmul_with_output_mode_f<TB, TC, DB, DC>(
        &self,
        rhs: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        c: impl TensorViewMutAPI<Type = TC, Backend = DeviceCuda, Dim = DC>,
        mode: <TC as CudaMatMulMode>::Mode,
    ) -> Result<()>
    where
        TB: CudaMatMulMode + num::Zero,
        TC: CudaMatMulMode + num::Zero + num::One,
        DB: DimAPI,
        DC: DimAPI,
    {
        with_mode(c, self.view(), rhs, TC::one(), TC::zero(), mode)
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
        self.matmul_from_mode_f(a, b, alpha, beta, mode).rstsr_unwrap()
    }

    fn matmul_from_mode_f<TA, TB, DA, DB>(
        &mut self,
        a: impl TensorViewAPI<Type = TA, Backend = DeviceCuda, Dim = DA>,
        b: impl TensorViewAPI<Type = TB, Backend = DeviceCuda, Dim = DB>,
        alpha: T,
        beta: T,
        mode: <T as CudaMatMulMode>::Mode,
    ) -> Result<()>
    where
        TA: CudaMatMulMode + num::Zero,
        TB: CudaMatMulMode + num::Zero,
        DA: DimAPI,
        DB: DimAPI,
        R: DataMutAPI<Data = CudaBuffer<T>>,
    {
        with_mode(self.view_mut(), a, b, alpha, beta, mode)
    }
}

/// A product waiting for its output type: `a.matmul_with_mode(&b, mode)`, completed by
/// [`Self::out`] (or its fallible form [`Self::out_f`]).
///
/// The mode's type is that of the output, so `out::<TC>()` is what makes it mean something: a
/// `ComputeType32F` is the mode of an `f32` output, of a `Complex<f32>` output, or of an `f32`
/// output of `f16` operands alike.
pub struct WithMode<'a, TA: 'a, DA, TB: 'a, DB, M>
where
    DA: DimAPI,
    DB: DimAPI,
{
    a: TensorView<'a, TA, DeviceCuda, DA>,
    b: TensorView<'a, TB, DeviceCuda, DB>,
    mode: M,
}

impl<TA, DA, TB, DB, M> WithMode<'_, TA, DA, TB, DB, M>
where
    DA: DimAPI,
    DB: DimAPI,
{
    /// The product, with the output stored as `TC` (`TC: CudaMatMulMode<Mode = M>`) in an
    /// `IxD` tensor: `out::<f32>()`.
    pub fn out<TC>(self) -> Tensor<TC, DeviceCuda, IxD>
    where
        TC: CudaMatMulMode<Mode = M> + num::Zero + num::One,
        TA: CudaMatMulMode + num::Zero,
        TB: CudaMatMulMode + num::Zero,
        LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = IxD>,
    {
        self.out_f().rstsr_unwrap()
    }

    /// The fallible form of [`Self::out`].
    pub fn out_f<TC>(self) -> Result<Tensor<TC, DeviceCuda, IxD>>
    where
        TC: CudaMatMulMode<Mode = M> + num::Zero + num::One,
        TA: CudaMatMulMode + num::Zero,
        TB: CudaMatMulMode + num::Zero,
        LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = IxD>,
    {
        matmul_with_mode_f(self.a, self.b, TC::one(), self.mode)
    }
}

/// The fallible free function of [`WithMode::out`]: `alpha * (a @ b)` into a fresh output.
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
