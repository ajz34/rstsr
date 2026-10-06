//! Set-operation device impls for [`DeviceRayonAutoImpl`] (generic rayon
//! device; also [`DeviceFaer`]'s impl through the `rayon_auto_impl`
//! symlink).
//!
//! The unique sweep is a single host-side scan with data-dependent control
//! flow (not yet parallelized); the isin binary search parallelizes per x1
//! value. Both delegate to the serial kernels below the parallel switch.

use core::any::TypeId;

use rstsr_dtype_traits::ExtSortCmp;

use crate::prelude_dev::*;

/// Dtype check mirroring [`crate::device_cpu_serial::set::use_fast_path`];
/// kept local so the device crates' symlinked module stays self-contained.
fn use_fast_path<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
        || TypeId::of::<T>() == TypeId::of::<f64>()
        || TypeId::of::<T>() == TypeId::of::<bool>()
        || TypeId::of::<T>() == TypeId::of::<i8>()
        || TypeId::of::<T>() == TypeId::of::<i16>()
        || TypeId::of::<T>() == TypeId::of::<i32>()
        || TypeId::of::<T>() == TypeId::of::<i64>()
        || TypeId::of::<T>() == TypeId::of::<isize>()
        || TypeId::of::<T>() == TypeId::of::<u8>()
        || TypeId::of::<T>() == TypeId::of::<u16>()
        || TypeId::of::<T>() == TypeId::of::<u32>()
        || TypeId::of::<T>() == TypeId::of::<u64>()
        || TypeId::of::<T>() == TypeId::of::<usize>()
}

impl<T, D> OpUniqueAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    D: DimAPI,
{
    fn unique_values(&self, a: &Vec<T>, la: &Layout<D>, values: &mut Vec<MaybeUninit<T>>) -> Result<usize> {
        let access = LineAccess::new(a, la)?;
        if use_fast_path::<T>() {
            let is_nan = |x: &T| x.ext_is_nan();
            unique_values_sorted_cpu_serial(values, &access, &is_nan)
        } else {
            unique_values_naive_cpu_serial(values, &access)
        }
    }

    fn unique_all(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        values: &mut Vec<MaybeUninit<T>>,
        indices: &mut Vec<MaybeUninit<usize>>,
        inverse: &mut Vec<MaybeUninit<usize>>,
        counts: &mut Vec<MaybeUninit<usize>>,
    ) -> Result<usize> {
        let access = LineAccess::new(a, la)?;
        let flat_c = |i: usize| -> usize { i };
        if use_fast_path::<T>() {
            let is_nan = |x: &T| x.ext_is_nan();
            unique_all_sorted_cpu_serial(values, indices, inverse, counts, &access, &flat_c, &is_nan)
        } else {
            unique_all_naive_cpu_serial(values, indices, inverse, counts, &access, &flat_c)
        }
    }
}

impl<T, D1> OpIsinAPI<T, D1> for DeviceRayonAutoImpl
where
    T: Clone + PartialEq + ExtSortCmp + Send + Sync + 'static,
    D1: DimAPI,
{
    fn isin(
        &self,
        x1: &Vec<T>,
        l1: &Layout<D1>,
        x2: &Vec<T>,
        l2: &Layout<IxD>,
        invert: bool,
    ) -> Result<(Storage<DataOwned<Vec<bool>>, bool, Self>, Layout<IxD>)> {
        // sort a copy of x2's values (registered efficiency exception)
        let access2 = LineAccess::new(x2, l2)?;
        let n2 = access2.len();
        let mut x2_sorted: Vec<T> = Vec::with_capacity(n2);
        for i in 0..n2 {
            x2_sorted.push(access2.get(i).clone());
        }
        x2_sorted.sort_by(|a, b| a.ext_total_cmp(b));
        x2_sorted.dedup_by(|a, b| a == b);

        let shape: Vec<usize> = l1.shape().as_ref().to_vec();
        let layout_c = shape.new_contig(None, self.default_order());
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        let access1 = LineAccess::new(x1, l1)?;
        let is_nan = |x: &T| x.ext_is_nan();
        isin_cpu_serial(storage.raw_mut(), &x2_sorted, &access1, &is_nan)?;
        if invert {
            for slot in storage.raw_mut().iter_mut() {
                // SAFETY: every slot was written above.
                let v = unsafe { slot.assume_init_mut() };
                *v = !*v;
            }
        }
        // SAFETY: `isin_cpu_serial` (+ the invert flip above) wrote every
        // element of the fresh storage exactly once.
        let storage = unsafe { <Self as DeviceCreationAnyAPI<bool>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}
