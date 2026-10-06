//! Nonzero device impls for [`DeviceRayonAutoImpl`] (generic rayon device;
//! also [`DeviceFaer`]'s impl through the `rayon_auto_impl` symlink).
//! The two passes are reductions/gathers with data-dependent output length;
//! a chunked-parallel count + prefix-sum fill can follow with the perf pass.

use crate::prelude_dev::*;

impl<T, D> OpNonzeroAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    fn nonzero_count(&self, a: &Vec<T>, la: &Layout<D>, is_nonzero: &dyn Fn(&T) -> bool) -> Result<usize> {
        nonzero_count_cpu_serial(a, la, is_nonzero)
    }

    fn nonzero_fill(
        &self,
        out: &mut Vec<MaybeUninit<usize>>,
        a: &Vec<T>,
        la: &Layout<D>,
        is_nonzero: &dyn Fn(&T) -> bool,
    ) -> Result<usize> {
        nonzero_fill_cpu_serial(out, a, la, is_nonzero)
    }
}
