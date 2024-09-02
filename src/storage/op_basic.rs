use core::ops::Add;

use num::Zero;

use crate::prelude_dev::*;

pub trait OpAssignAPI<T, DC, DA>
where
    DC: DimAPI,
    DA: DimAPI,
    Self: DeviceAPI<T>,
{
    fn assign_arbitary_layout(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()>;
}

pub trait OpAddAPI<T, TB, D>
where
    T: Add<TB, Output = T> + Clone,
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TB>,
{
    fn add_ternary(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<D>,
        a: &Storage<T, Self>,
        la: &Layout<D>,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
    ) -> Result<()>;
}

pub trait OpSumAPI<T, D>
where
    T: Zero + Add<Output = T> + Clone,
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
