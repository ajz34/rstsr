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
    fn mask_select(
        &self,
        c: &mut Vec<MaybeUninit<TA>>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        mask: &Vec<bool>,
        lm: &Layout<DM>,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        mask_select_cpu_rayon(c, a, &la.to_dim::<IxD>()?, mask, &lm.to_dim::<IxD>()?, self.default_order(), pool)
    }

    fn mask_fill(&self, a: &mut Vec<TA>, la: &Layout<DA>, mask: &Vec<bool>, lm: &Layout<DM>, value: TA) -> Result<()> {
        let pool = self.get_current_pool();
        mask_fill_cpu_rayon(a, &la.to_dim::<IxD>()?, mask, &lm.to_dim::<IxD>()?, value, self.default_order(), pool)
    }
}

impl<TC, TA> DeviceIndexPutAPI<TC, TA> for DeviceRayonAutoImpl
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
{
    fn index_put(
        &self,
        a: &mut Vec<TC>,
        la: &Layout<IxD>,
        axis: usize,
        indices: &[usize],
        value: &Vec<TA>,
        lvalue: &Layout<IxD>,
    ) -> Result<()> {
        // the scatter is serial: duplicate indices make disjoint writes
        // impossible, so the serial kernel is called directly
        index_put_promote_cpu_serial(a, la, axis, indices, value, lvalue)
    }
}
