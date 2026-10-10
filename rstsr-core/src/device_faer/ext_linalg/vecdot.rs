//! Vector dot product with dtype promotion for the faer device.

use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;
use rstsr_dtype_traits::{DTypePromoteAPI, ExtNum};

impl<TA, TB, TC, DA, DB, DC> DeviceExtVecdotAPI<TA, TB, TC, DA, DB, DC> for DeviceFaer
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Zero + ExtNum + Mul<TC, Output = TC>,
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
        let pool = self.get_current_pool();
        vecdot_ext_naive_cpu_rayon(c, lc, a, la, b, lb, axes_a, axes_b, self.default_order(), pool)
    }
}

#[cfg(test)]
mod test_ext {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_vecdot_mixed_dtypes() {
        // u8 . u16 -> u16 under both default orders
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceFaer::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([1u8, 2, 3], &device);
            let b = rt::tensor_from_nested!([4u16, 5, 6], &device);
            let c = rt::ext_vecdot(&a, &b, None);
            assert_eq!(format!("{c}"), "32");
        }
    }

    #[test]
    fn test_ext_vecdot_same_dtype_matches_vecdot() {
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceFaer::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([[1.0f64, 2.], [3., 4.]], &device);
            let b = rt::tensor_from_nested!([[5.0f64, 6.], [7., 8.]], &device);
            assert_eq!(format!("{}", rt::ext_vecdot(&a, &b, None)), format!("{}", rt::vecdot(&a, &b, None)));
        }
    }
}
