use crate::prelude_dev::*;

impl<T, D> DeviceIndexSelectAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
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
        let pool = self.get_current_pool();
        index_select_cpu_rayon(c, lc, a, la, axis, indices, pool)
    }
}

impl<TA, DA, DM> DeviceMaskIndexAPI<TA, DA, DM> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync,
    DA: DimAPI,
    DM: DimAPI,
{
    /// Serial gather / scatter (the mask prefix-sum needed for a parallel fill
    /// can follow with the perf pass, as for `nonzero`).
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
