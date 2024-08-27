use core::ops::Add;

use crate::prelude_dev::*;

pub trait OpAssignAPI<T, D1, D2>
where
    D1: DimAPI,
    D2: DimAPI,
    Self: DeviceAPI<T>,
{
    /// Assign values from `b` to `a` with arbitary layout.
    ///
    /// This function does not
    fn assign_arbitary_layout(
        &self,
        a: &mut Storage<T, Self>,
        la: &Layout<D1>,
        b: &Storage<T, Self>,
        lb: &Layout<D2>,
    ) -> Result<()>;
}

pub trait OpAddAPI<TA, TB, TC, DA, DB, DC>
where
    TA: Add<TB, Output = TC>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceRawVecAPI<TA> + DeviceRawVecAPI<TB> + DeviceRawVecAPI<TC>,
{
    fn add(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<DC>,
        a: &mut Storage<TA, Self>,
        la: &Layout<DA>,
        b: &mut Storage<TB, Self>,
        lb: &Layout<DB>,
    ) -> Result<()>;
}
