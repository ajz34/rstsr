//! take_along_axis device impls for [`DeviceCpuSerial`].

use crate::prelude_dev::*;

impl<T, DA, DI> DeviceTakeAlongAxisAPI<T, DA, DI> for DeviceCpuSerial
where
    T: Clone,
    DA: DimAPI,
    DI: DimAPI,
{
    fn take_along_axis(
        &self,
        c: &mut Vec<MaybeUninit<T>>,
        layout_c: &Layout<IxD>,
        a: &Vec<T>,
        la: &Layout<DA>,
        idx: &Vec<usize>,
        lidx: &Layout<DI>,
        axis: usize,
    ) -> Result<()> {
        let stride_ref: &[isize] = layout_c.stride().as_ref();
        let out_strides: Vec<usize> = stride_ref.iter().map(|&s| s.unsigned_abs()).collect();
        take_along_axis_cpu_serial(c, &out_strides, a, la, idx, lidx, axis)
    }
}
