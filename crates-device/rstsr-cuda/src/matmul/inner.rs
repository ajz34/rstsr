//! The device entries: the inner function the drivers are dispatched from, and `DeviceMatMulAPI`.

use crate::matmul::drivers::{self, NaiveScalar, Problem};
use crate::matmul::geom::resolve_geom;
use crate::matmul::mode::CudaMatMulMode;
use crate::prelude_dev::*;

/// `c = alpha * (a @ b) + beta * c`, with the compute `mode` of the output type given by the
/// caller instead of read from the device flags.
///
/// `c` is the output buffer: `TC` for the `beta` path, `MaybeUninit<TC>` for the fresh one. With
/// `beta == 0` it is never read, which is the whole of the `matmul_uninit` contract.
///
/// The drivers are tried in the chain order, filtered by `flags.matmul_drivers`; the first one
/// that takes the problem runs it, and an error is returned only when none did.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_with_mode_inner<TA, TB, TC, DA, DB, DC, TBuf>(
    device: &DeviceCuda,
    c: &CudaBuffer<TBuf>,
    lc: &Layout<DC>,
    a: &CudaBuffer<TA>,
    la: &Layout<DA>,
    b: &CudaBuffer<TB>,
    lb: &Layout<DB>,
    alpha: TC,
    beta: TC,
    mode: <TC as CudaMatMulMode>::Mode,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: NaiveScalar,
    TB: NaiveScalar,
    TC: NaiveScalar + CudaMatMulMode,
    TBuf: 'static,
{
    let geom = resolve_geom(device.default_order(), &lc.to_dim::<IxD>()?, &la.to_dim::<IxD>()?, &lb.to_dim::<IxD>()?)?;
    if geom.m == 0 || geom.n == 0 || geom.n_batch == 0 {
        return Ok(());
    }
    let problem = Problem {
        device,
        geom,
        c,
        a,
        b,
        alpha,
        beta,
        // `beta == 0` means "do not read c": the `MaybeUninit` slots are written, never touched.
        read_c: !beta.is_zero(),
    };
    drivers::dispatch(&problem, &device.flags(), &mode)
}

impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceCuda
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<TC, Raw = CudaBuffer<TC>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    // Every triple is accepted: the accelerated drivers serve the ones they can, the naive driver
    // takes the rest (integers, complex, half, mixed operands).
    TA: NaiveScalar,
    TB: NaiveScalar,
    TC: NaiveScalar + CudaMatMulMode,
{
    fn matmul(
        &self,
        c: &mut CudaBuffer<TC>,
        lc: &Layout<DC>,
        a: &CudaBuffer<TA>,
        la: &Layout<DA>,
        b: &CudaBuffer<TB>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        let mode = TC::matmul_mode(&self.flags());
        matmul_with_mode_inner(self, c, lc, a, la, b, lb, alpha, beta, mode)
    }

    fn matmul_uninit(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &CudaBuffer<TA>,
        la: &Layout<DA>,
        b: &CudaBuffer<TB>,
        lb: &Layout<DB>,
        alpha: TC,
    ) -> Result<()> {
        let mode = TC::matmul_mode(&self.flags());
        matmul_with_mode_inner(self, c, lc, a, la, b, lb, alpha, <TC as num::Zero>::zero(), mode)
    }
}
