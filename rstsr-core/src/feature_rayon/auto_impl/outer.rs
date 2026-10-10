use crate::prelude_dev::*;

impl<TA, TB, TC> DeviceOuterAPI<TA, TB, TC> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Send + Sync,
    TA: Mul<TB, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn outer(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<Ix2>,
        a: &Vec<TA>,
        la: &Layout<Ix1>,
        b: &Vec<TB>,
        lb: &Layout<Ix1>,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        outer_naive_cpu_rayon(c, lc, a, la, b, lb, pool)
    }
}
