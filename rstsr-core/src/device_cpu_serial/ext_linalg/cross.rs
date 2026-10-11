//! Cross product with dtype promotion for the serial CPU device.

use crate::prelude_dev::*;
use core::ops::{Mul, Sub};
use rstsr_dtype_traits::DTypePromoteAPI;

impl<TA, TB, TC, D> DeviceExtCrossAPI<TA, TB, TC, D> for DeviceCpuSerial
where
    D: DimAPI,
    TA: Clone,
    TB: Clone,
    TC: Clone + Mul<TC, Output = TC> + Sub<TC, Output = TC>,
    TA: DTypePromoteAPI<TB, Res = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn ext_cross(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        axis: usize,
    ) -> Result<()> {
        cross_ext_naive_cpu_serial(c, lc, a, la, b, lb, axis)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_cross_mixed_dtypes() {
        // i8 x i16 -> i16
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1i8, 2, 3], &device);
        let b = rt::tensor_from_nested!([4i16, 5, 6], &device);
        let c = rt::ext_cross(&a, &b, -1);
        // the promoted dtype is `i16` (the `to_vec` binding pins it at compile time)
        assert_eq!(c.to_vec(), vec![-3i16, 6, -3]);
    }

    #[test]
    fn test_ext_cross_same_dtype_matches_cross() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
        let b = rt::tensor_from_nested!([4.0f64, 5., 6.], &device);
        assert_eq!(format!("{}", rt::ext_cross(&a, &b, -1)), format!("{}", rt::cross(&a, &b, -1)));
    }
}
