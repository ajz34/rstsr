//! Tensor contraction with dtype promotion for the faer device.

use crate::prelude_dev::*;
use core::ops::Mul;
use num::{One, Zero};
use rstsr_dtype_traits::DTypePromoteAPI;

impl<TA, TB, TC, DA, DB, DC> DeviceExtTensordotAPI<TA, TB, TC, DA, DB, DC> for DeviceFaer
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static + Zero + One + Mul<TC, Output = TC>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
    Self: DeviceAPI<MaybeUninit<TC>, Raw = Vec<MaybeUninit<TC>>>,
    Self: DeviceExtMatMulAPI<TA, TB, TC, Ix2, Ix2, Ix2>,
{
    fn ext_tensordot(
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
        let order = self.default_order();
        // View-only GEMM fast path; never copies to enable GEMM.
        if let Some((la2, lb2, lc2)) = tensordot_gemm_layouts(la, axes_a, lb, axes_b, lc, order)? {
            return self.ext_matmul_uninit(c, &lc2, a, &la2, b, &lb2, TC::one());
        }
        let pool = self.get_current_pool();
        tensordot_ext_naive_cpu_rayon(c, lc, a, la, b, lb, axes_a, axes_b, pool)
    }
}

#[cfg(test)]
mod test_ext {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_tensordot_mixed_dtypes() {
        // u8 x u16 -> u16 (matrix product through the ext GEMM fast path) under
        // both default orders
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceFaer::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([[1u8, 2], [3, 4]], &device);
            let b = rt::tensor_from_nested!([[5u16, 6], [7, 8]], &device);
            let c = rt::ext_tensordot(&a, &b, 1);
            assert_eq!(format!("{c}"), "[[ 19 22]\n [ 43 50]]");
        }
    }

    #[test]
    fn test_ext_tensordot_same_dtype_matches_tensordot() {
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceFaer::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([[1.0f64, 2.], [3., 4.]], &device);
            let b = rt::tensor_from_nested!([[5.0f64, 6.], [7., 8.]], &device);
            assert_eq!(format!("{}", rt::ext_tensordot(&a, &b, 1)), format!("{}", rt::tensordot(&a, &b, 1)));
        }
    }
}
