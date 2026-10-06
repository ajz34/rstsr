//! Nonzero device impls for [`DeviceCpuSerial`].

use crate::prelude_dev::*;

impl<T, D> OpNonzeroAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
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
