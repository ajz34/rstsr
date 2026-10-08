//! Array indexing device impls for [`DeviceRayonAutoImpl`] (generic rayon
//! device; also [`DeviceFaer`](crate::device_faer::DeviceFaer)'s impl through
//! the `rayon_auto_impl` symlink). Delegates to the serial kernel (a parallel
//! split over the output can follow with the perf pass).

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
        indexers: &[ArrayAuxIndexer<'_>],
        consec: usize,
    ) -> Result<()> {
        let indexers: Vec<(usize, &[usize], Layout<IxD>)> =
            indexers.iter().map(|ix| (ix.src_axis, ix.indices, ix.layout.clone())).collect();
        array_index_cpu_serial(c, lc, a, la, base_layout, &indexers, consec)
    }
}
