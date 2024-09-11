//! Matrix, vector multiplication and related operations.

#![allow(clippy::too_many_arguments)]

use core::ops::{Add, Mul};

use crate::prelude_dev::*;

pub trait DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn matmul(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<DC>,
        a: &Storage<TA, Self>,
        la: &Layout<DA>,
        b: &Storage<TB, Self>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceGEMMAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn gemm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix2>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceSYMMAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn symm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix2>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        side: TensorSide,
        uplo: TensorUpLo,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceSYRKAPI<TA, TC>
where
    TA: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TC>,
{
    fn syrk(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix2>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        uplo: TensorUpLo,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceHERKAPI<TA, TC>
where
    TA: Mul<TA, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TC>,
{
    fn herk(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix2>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        uplo: TensorUpLo,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceGEMVAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn gemv(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix1>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;

    fn gevm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix1>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix1>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}

pub trait DeviceInnerDotAPI<TA, TB, TC>
where
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn inner_dot(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix0>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix1>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()>;
}
