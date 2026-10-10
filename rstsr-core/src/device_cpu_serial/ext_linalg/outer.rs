//! Outer product with dtype promotion for the serial CPU device.

use crate::prelude_dev::*;
use core::ops::Mul;
use rstsr_dtype_traits::DTypePromoteAPI;

impl<TA, TB, TC> DeviceExtOuterAPI<TA, TB, TC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Mul<TC, Output = TC>,
    TA: DTypePromoteAPI<TB, Res = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn ext_outer(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<Ix2>,
        a: &Vec<TA>,
        la: &Layout<Ix1>,
        b: &Vec<TB>,
        lb: &Layout<Ix1>,
    ) -> Result<()> {
        outer_ext_naive_cpu_serial(c, lc, a, la, b, lb)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_outer_mixed_dtypes() {
        // u8 x u16 -> u16 under both default orders
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceCpuSerial::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([1u8, 2, 3], &device);
            let b = rt::tensor_from_nested!([4u16, 5], &device);
            let c = rt::ext_outer(&a, &b);
            assert_eq!(c.shape(), &[3, 2]);
            assert_eq!(format!("{c}"), "[[ 4 5]\n [ 8 10]\n [ 12 15]]");
        }
    }

    #[test]
    fn test_ext_outer_same_dtype_matches_outer() {
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceCpuSerial::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
            let b = rt::tensor_from_nested!([4.0f64, 5.], &device);
            assert_eq!(format!("{}", rt::ext_outer(&a, &b)), format!("{}", rt::outer(&a, &b)));
        }
    }
}
