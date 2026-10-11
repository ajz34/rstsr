use crate::prelude_dev::*;
use core::ops::{Mul, Sub};

impl<TA, TB, TC, D> DeviceCrossAPI<TA, TB, TC, D> for DeviceRayonAutoImpl
where
    D: DimAPI,
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Send + Sync + Sub<TC, Output = TC>,
    TA: Mul<TB, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn cross(
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
        cross_naive_cpu_rayon(c, lc, a, la, b, lb, axis, pool)
    }
}
