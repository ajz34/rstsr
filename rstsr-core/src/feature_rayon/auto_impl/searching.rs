//! Searchsorted device impls for [`DeviceRayonAutoImpl`] (generic rayon
//! device; also [`DeviceFaer`]'s impl through the `rayon_auto_impl` symlink).

use rstsr_dtype_traits::ExtSortCmp;

use crate::prelude_dev::*;

impl<T, D2> OpSearchSortedAPI<T, T, D2> for DeviceRayonAutoImpl
where
    T: Clone + ExtSortCmp + Send + Sync,
    D2: DimAPI,
{
    fn searchsorted(
        &self,
        x1: &Vec<T>,
        l1: &Layout<IxD>,
        x2: &Vec<T>,
        l2: &Layout<D2>,
        side_right: bool,
        sorter: Option<&[usize]>,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)> {
        let shape: Vec<usize> = l2.shape().as_ref().to_vec();
        let layout_c = shape.new_contig(None, self.default_order());
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        let pool = self.get_current_pool();
        searchsorted_cpu_rayon(
            storage.raw_mut(),
            &layout_c,
            x1,
            l1,
            x2,
            l2,
            !side_right,
            sorter,
            &|x: &T| x.ext_is_nan(),
            pool,
        )?;
        // SAFETY: `searchsorted_cpu_rayon` above wrote every element of the
        // fresh storage exactly once (one position per x2 element).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<usize>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}
