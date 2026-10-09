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

impl<TC, TA, DA, DI> DevicePutAlongAxisAPI<TC, DA, DI, TA> for DeviceCpuSerial
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
    DA: DimAPI,
    DI: DimAPI,
{
    fn put_along_axis(
        &self,
        a: &mut Vec<TC>,
        la: &Layout<DA>,
        idx: &Vec<usize>,
        lidx: &Layout<DI>,
        values: &Vec<TA>,
        lvalues: &Layout<DI>,
        axis: usize,
    ) -> Result<()> {
        put_along_axis_promote_cpu_serial(a, la, idx, lidx, values, lvalues, axis)
    }
}
