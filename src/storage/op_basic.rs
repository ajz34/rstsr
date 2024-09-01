use core::ops::Add;

use num::{traits::NumAssign, Num, Zero};

use crate::prelude_dev::*;

pub trait OpAssignAPI<T, DC, DA>
where
    DC: DimAPI,
    DA: DimAPI,
    Self: DeviceRawVecAPI<T>,
{
    fn assign_arbitary_layout(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()>;
}

pub trait OpAddAPI<T, D>
where
    T: Add<Output = T> + Clone,
    D: DimAPI,
    Self: DeviceRawVecAPI<T>,
{
    fn ternary_add(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<D>,
        a: &Storage<T, Self>,
        la: &Layout<D>,
        b: &Storage<T, Self>,
        lb: &Layout<D>,
    ) -> Result<()>;
}

pub trait OpSumAPI<T, D>
where
    T: Zero + Add<Output = T> + Clone,
    D: DimAPI,
    Self: DeviceRawVecAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
