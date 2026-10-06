//! Sorting device impls for [`DeviceRayonAutoImpl`] (generic rayon device;
//! also [`DeviceFaer`]'s impl through the `rayon_auto_impl` symlink).

use crate::prelude_dev::*;

impl<T, D> OpSortAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + ExtSortCmp + Send + Sync,
    D: DimAPI,
{
    fn sort_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axis: usize,
        descending: bool,
        _stable: bool,
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)> {
        let shape: Vec<usize> = la.shape().as_ref().to_vec();
        let layout_c = shape.new_contig(None, self.default_order());
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        let pool = self.get_current_pool();
        let f = |x: &T, y: &T| x.ext_total_cmp(y);
        sort_axes_cpu_rayon(
            Some(storage.raw_mut()),
            None,
            &layout_c,
            a,
            la,
            axis,
            &f,
            descending,
            &|x: &T| x.ext_is_nan(),
            pool,
        )?;
        // SAFETY: `sort_axes_cpu_rayon` above wrote every element of the fresh
        // storage exactly once (disjoint per-line output blocks).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}

impl<T, D> OpArgSortAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + ExtSortCmp + Send + Sync,
    D: DimAPI,
{
    fn argsort_axes(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axis: usize,
        descending: bool,
        _stable: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)> {
        let shape: Vec<usize> = la.shape().as_ref().to_vec();
        let layout_c = shape.new_contig(None, self.default_order());
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        let pool = self.get_current_pool();
        let f = |x: &T, y: &T| x.ext_total_cmp(y);
        sort_axes_cpu_rayon(
            None,
            Some(storage.raw_mut()),
            &layout_c,
            a,
            la,
            axis,
            &f,
            descending,
            &|x: &T| x.ext_is_nan(),
            pool,
        )?;
        // SAFETY: see the `OpSortAPI` impl (same write contract).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<usize>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}

impl<T, D> OpSortCustomAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    fn sort_axes_custom<F>(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axis: usize,
        f: F,
    ) -> Result<(Storage<DataOwned<Vec<T>>, T, Self>, Layout<IxD>)>
    where
        F: Fn(&T, &T) -> core::cmp::Ordering + Send + Sync,
    {
        let shape: Vec<usize> = la.shape().as_ref().to_vec();
        let layout_c = shape.new_contig(None, self.default_order());
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        let pool = self.get_current_pool();
        sort_axes_cpu_rayon(Some(storage.raw_mut()), None, &layout_c, a, la, axis, &f, false, &|_| false, pool)?;
        // SAFETY: see the `OpSortAPI` impl (same write contract).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }

    fn argsort_axes_custom<F>(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        axis: usize,
        f: F,
    ) -> Result<(Storage<DataOwned<Vec<usize>>, usize, Self>, Layout<IxD>)>
    where
        F: Fn(&T, &T) -> core::cmp::Ordering + Send + Sync,
    {
        let shape: Vec<usize> = la.shape().as_ref().to_vec();
        let layout_c = shape.new_contig(None, self.default_order());
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        let pool = self.get_current_pool();
        sort_axes_cpu_rayon(None, Some(storage.raw_mut()), &layout_c, a, la, axis, &f, false, &|_| false, pool)?;
        // SAFETY: see the `OpSortAPI` impl (same write contract).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<usize>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}
