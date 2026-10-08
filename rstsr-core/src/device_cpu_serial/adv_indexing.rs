use crate::prelude_dev::*;

impl<T, D> DeviceIndexSelectAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    fn index_select(
        &self,
        c: &mut Vec<MaybeUninit<T>>,
        lc: &Layout<D>,
        a: &Vec<T>,
        la: &Layout<D>,
        axis: usize,
        indices: &[usize],
    ) -> Result<()> {
        index_select_cpu_serial(c, lc, a, la, axis, indices)
    }
}

impl<TA, DA, DM> DeviceMaskIndexAPI<TA, DA, DM> for DeviceCpuSerial
where
    TA: Clone,
    DA: DimAPI,
    DM: DimAPI,
{
    fn mask_select(
        &self,
        c: &mut Vec<MaybeUninit<TA>>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        mask: &Vec<bool>,
        lm: &Layout<DM>,
    ) -> Result<()> {
        mask_select_cpu_serial(c, a, &la.to_dim::<IxD>()?, mask, &lm.to_dim::<IxD>()?, self.default_order())
    }

    fn mask_fill(&self, a: &mut Vec<TA>, la: &Layout<DA>, mask: &Vec<bool>, lm: &Layout<DM>, value: TA) -> Result<()> {
        mask_fill_cpu_serial(a, &la.to_dim::<IxD>()?, mask, &lm.to_dim::<IxD>()?, value, self.default_order())
    }
}
