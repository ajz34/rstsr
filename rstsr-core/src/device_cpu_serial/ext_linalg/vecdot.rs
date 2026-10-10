//! Vector dot product with dtype promotion for the serial CPU device.

use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;
use rstsr_dtype_traits::{DTypePromoteAPI, ExtNum};

impl<TA, TB, TC, DA, DB, DC> DeviceExtVecdotAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero + ExtNum + Mul<TC, Output = TC>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn ext_vecdot(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()> {
        vecdot_ext_naive_cpu_serial(c, lc, a, la, b, lb, axes_a, axes_b, self.default_order())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_vecdot_mixed_dtypes() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1u8, 2, 3], &device);
        let b = rt::tensor_from_nested!([4u16, 5, 6], &device);
        let c = rt::ext_vecdot(&a, &b, None);
        assert_eq!(format!("{c}"), "32");
    }

    #[test]
    fn test_ext_vecdot_conjugates_first_arg() {
        // complex first operand against real second: conj(x1) * x2, both promoted
        // to complex128
        use num::complex::c64;
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([c64(1., 2.), c64(0., 1.)], &device);
        let b = rt::tensor_from_nested!([2.0f64, 3.0], &device);
        let c = rt::ext_vecdot(&a, &b, None);
        // conj(1+2i)*2 + conj(i)*3 = (1-2i)*2 - 3i = 2 - 7i
        assert_eq!(format!("{c}"), "2-7i");
    }

    #[test]
    fn test_ext_vecdot_same_dtype_matches_vecdot() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([[1.0f64, 2.], [3., 4.]], &device);
        let b = rt::tensor_from_nested!([[5.0f64, 6.], [7., 8.]], &device);
        assert_eq!(format!("{}", rt::ext_vecdot(&a, &b, None)), format!("{}", rt::vecdot(&a, &b, None)));
    }
}
