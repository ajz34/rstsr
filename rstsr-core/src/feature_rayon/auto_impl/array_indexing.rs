//! Array indexing device impls for [`DeviceRayonAutoImpl`] (generic rayon
//! device; also `DeviceFaer`'s impl through the `rayon_auto_impl` symlink).

use crate::prelude_dev::*;

impl<T> DeviceArrayIndexAPI<T> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
{
    fn array_index(
        &self,
        c: &mut Vec<MaybeUninit<T>>,
        lc: &Layout<IxD>,
        a: &Vec<T>,
        la: &Layout<IxD>,
        base_layout: &Layout<IxD>,
        indexers: &[ArrayAuxIndexer<'_, Self>],
        consec: usize,
        order: FlagOrder,
    ) -> Result<()> {
        let pool = self.get_current_pool();
        // this device's raw buffer of `usize` is a host vector
        let indexers: Vec<(usize, &[usize], Layout<IxD>)> =
            indexers.iter().map(|ix| (ix.src_axis, ix.indices.as_slice(), ix.layout.clone())).collect();
        array_index_cpu_rayon(c, lc, a, la, base_layout, &indexers, consec, order, pool)
    }
}

impl<TC, TA> DeviceArrayIndexAssignAPI<TC, TA> for DeviceRayonAutoImpl
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
{
    fn array_index_assign(
        &self,
        a: &mut Vec<TC>,
        la: &Layout<IxD>,
        base_layout: &Layout<IxD>,
        indexers: &[ArrayAuxIndexer<'_, Self>],
        value: &Vec<TA>,
        lvalue: &Layout<IxD>,
        consec: usize,
        order: FlagOrder,
    ) -> Result<()> {
        // the scatter is serial: duplicate index targets make disjoint writes
        // impossible, so no rayon kernel (the serial kernel is called directly)
        let indexers: Vec<(usize, &[usize], Layout<IxD>)> =
            indexers.iter().map(|ix| (ix.src_axis, ix.indices.as_slice(), ix.layout.clone())).collect();
        array_index_assign_promote_cpu_serial(a, la, base_layout, &indexers, value, lvalue, consec, order)
    }
}
