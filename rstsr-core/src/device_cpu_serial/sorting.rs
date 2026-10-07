//! Sorting device impls for [`DeviceCpuSerial`].

use crate::prelude_dev::*;

impl<T, D> OpSortAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ExtSortCmp,
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
        let f = |x: &T, y: &T| x.ext_total_cmp(y);
        sort_axes_cpu_serial(Some(storage.raw_mut()), None, &layout_c, a, la, axis, &f, descending, &|x: &T| {
            x.ext_is_nan()
        })?;
        // SAFETY: `sort_axes_cpu_serial` above wrote every element of the
        // fresh storage exactly once (each output position belongs to exactly
        // one line at exactly one sorted slot).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}

impl<T, D> OpArgSortAPI<T, D> for DeviceCpuSerial
where
    T: Clone + ExtSortCmp,
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
        let f = |x: &T, y: &T| x.ext_total_cmp(y);
        sort_axes_cpu_serial(None, Some(storage.raw_mut()), &layout_c, a, la, axis, &f, descending, &|x: &T| {
            x.ext_is_nan()
        })?;
        // SAFETY: see the `OpSortAPI` impl (same write contract).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<usize>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}

impl<T, D> OpSortCustomAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
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
        sort_axes_cpu_serial(Some(storage.raw_mut()), None, &layout_c, a, la, axis, &f, false, &|_| false)?;
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
        sort_axes_cpu_serial(None, Some(storage.raw_mut()), &layout_c, a, la, axis, &f, false, &|_| false)?;
        // SAFETY: see the `OpSortAPI` impl (same write contract).
        let storage = unsafe { <Self as DeviceCreationAnyAPI<usize>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}
