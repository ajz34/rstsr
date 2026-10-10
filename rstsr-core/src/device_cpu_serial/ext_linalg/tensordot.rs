//! Tensor contraction with dtype promotion for the serial CPU device.

use crate::prelude_dev::*;
use core::ops::Mul;
use num::{One, Zero};
use rstsr_dtype_traits::DTypePromoteAPI;

impl<TA, TB, TC, DA, DB, DC> DeviceExtTensordotAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero + One + Mul<TC, Output = TC>,
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
        tensordot_ext_naive_cpu_serial(c, lc, a, la, b, lb, axes_a, axes_b)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_tensordot_mixed_dtypes() {
        // u8 x u16 -> u16: outer (axes=0), matrix product via the GEMM fast path
        // (axes=1), and full contraction (axes=2)
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[1u8, 2], [3, 4]], &device);
        let b = rt::tensor_from_nested!([[5u16, 6], [7, 8]], &device);

        let c0 = rt::ext_tensordot(&a, &b, 0);
        assert_eq!(c0.shape(), &[2, 2, 2, 2]);

        let c1 = rt::ext_tensordot(&a, &b, 1);
        assert_eq!(format!("{c1}"), "[[ 19 22]\n [ 43 50]]");

        let c2 = rt::ext_tensordot(&a, &b, 2);
        assert_eq!(format!("{c2}"), "70");
    }

    #[test]
    fn test_ext_tensordot_matches_prepromoted() {
        // mixed i32 x f64 agrees with the same-dtype contraction of the
        // already-promoted operands
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a_i = rt::tensor_from_nested!([[1i32, 2], [3, 4]], &device);
        let a_f = rt::tensor_from_nested!([[1.0f64, 2.0], [3.0, 4.0]], &device);
        let b_f = rt::tensor_from_nested!([[0.5f64, 1.5], [2.0, 0.25]], &device);
        assert_eq!(format!("{}", rt::ext_tensordot(&a_i, &b_f, 1)), format!("{}", rt::tensordot(&a_f, &b_f, 1)));
    }
}
