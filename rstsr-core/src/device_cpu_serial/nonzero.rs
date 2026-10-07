//! Nonzero device impls for [`DeviceCpuSerial`].

use rstsr_dtype_traits::ExtZero;

use crate::prelude_dev::*;

impl<T, D> OpNonzeroAPI<T, D> for DeviceCpuSerial
where
    T: ExtZero + PartialEq,
    D: DimAPI,
{
    fn nonzero_count(&self, a: &Vec<T>, la: &Layout<D>) -> Result<usize> {
        nonzero_count_cpu_serial(a, la)
    }

    fn nonzero_fill(&self, out: &mut [&mut Vec<MaybeUninit<usize>>], a: &Vec<T>, la: &Layout<D>) -> Result<()> {
        nonzero_fill_cpu_serial(out, a, la)
    }
}
