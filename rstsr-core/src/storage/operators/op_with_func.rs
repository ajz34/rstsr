use crate::prelude_dev::*;
use num::Zero;

/* #region op_func */

#[allow(non_camel_case_types)]
#[allow(clippy::too_many_arguments)]
pub trait DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB),
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<D>,
        a: &Storage<TA, Self>,
        la: &Layout<D>,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
        f: F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutC_RefA_NumB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB),
    Self: DeviceAPI<TA> + DeviceAPI<TC>,
{
    fn op_mutc_refa_numb_func(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<D>,
        a: &Storage<TA, Self>,
        la: &Layout<D>,
        b: TB,
        f: F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutC_NumA_RefB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB),
    Self: DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn op_mutc_numa_refb_func(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<D>,
        a: TA,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
        f: F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutA_RefB_API<TA, TB, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TA, &TB),
    Self: DeviceAPI<TA> + DeviceAPI<TB>,
{
    fn op_muta_refb_func(
        &self,
        a: &mut Storage<TA, Self>,
        la: &Layout<D>,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
        f: F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutA_NumB_API<TA, TB, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TA, &TB),
    Self: DeviceAPI<TA>,
{
    fn op_muta_numb_func(
        &self,
        a: &mut Storage<TA, Self>,
        la: &Layout<D>,
        b: TB,
        f: F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
pub trait DeviceOp_MutA_API<T, D, F>
where
    D: DimAPI,
    F: FnMut(&mut T),
    Self: DeviceAPI<T>,
{
    fn op_muta_func(&self, a: &mut Storage<T, Self>, la: &Layout<D>, f: F) -> Result<()>;
}

/* #endregion */

pub trait OpSumAPI<T, D>
where
    T: Zero + core::ops::Add<Output = T>,
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
