//! Cross product with dtype promotion for the faer device.

use crate::prelude_dev::*;
use core::ops::{Mul, Sub};
use rstsr_dtype_traits::DTypePromoteAPI;

impl<TA, TB, TC, D> DeviceExtCrossAPI<TA, TB, TC, D> for DeviceFaer
where
    D: DimAPI,
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Mul<TC, Output = TC> + Sub<TC, Output = TC>,
    TA: DTypePromoteAPI<TB, Res = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn ext_cross(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        axis: usize,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        cross_ext_naive_cpu_rayon(c, lc, a, la, b, lb, axis, pool)
    }
}
