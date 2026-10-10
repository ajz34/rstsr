//! Matrix multiplication with dtype promotion for the serial CPU device.

use core::ops::{Add, Mul};
use num::Zero;

use crate::prelude_dev::*;

impl<TA, TB, TC, DA, DB, DC> DeviceExtMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    // the column-major branch swaps the operands, so both promotion directions
    // are needed (the promotion table is symmetric in its result type)
    TB: DTypePromoteAPI<TA, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
    Self: DeviceAPI<MaybeUninit<TC>, Raw = Vec<MaybeUninit<TC>>>,
{
    fn ext_matmul_uninit(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        alpha: TC,
    ) -> Result<()> {
        let default_order = self.default_order();
        match default_order {
            RowMajor => matmul_ext_naive_uninit_cpu_serial(c, lc, a, la, b, lb, alpha),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let lc = lc.reverse_axes();
                matmul_ext_naive_uninit_cpu_serial(c, &lc, b, &lb, a, &la, alpha)
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_ext_matmul_mixed_dtypes() {
        // u8 @ u16 -> u16 under both default orders; column-major exercises the
        // swap-and-reverse branch of the promoting device impl
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceCpuSerial::default();
            device.set_default_order(order);
            let a = rt::tensor_from_nested!([[1u8, 2], [3, 4]], &device);
            let b = rt::tensor_from_nested!([[5u16, 6], [7, 8]], &device);
            let c = rt::ext_matmul(&a, &b);
            assert_eq!(format!("{c}"), "[[ 19 22]\n [ 43 50]]");
        }
    }

    #[test]
    fn test_ext_matmul_matches_prepromoted() {
        // mixed i32 x f64 and stacked i8 @ i16 agree with the same-dtype product
        // of the already-promoted operands
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);

        let a_i = rt::tensor_from_nested!([[1i32, 2], [3, 4]], &device);
        let a_f = rt::tensor_from_nested!([[1.0f64, 2.0], [3.0, 4.0]], &device);
        let b_f = rt::tensor_from_nested!([[0.5f64, 1.5], [2.0, 0.25]], &device);
        assert_eq!(format!("{}", rt::ext_matmul(&a_i, &b_f)), format!("{}", rt::matmul(&a_f, &b_f)));

        let s_i = rt::tensor_from_nested!([[[1i8, 2], [3, 4]], [[1, 0], [1, 1]]], &device);
        let s_j = rt::tensor_from_nested!([[[1i16, 2], [3, 4]], [[1, 0], [1, 1]]], &device);
        let b_j = rt::tensor_from_nested!([[1i16, 0], [1, 1]], &device);
        assert_eq!(format!("{}", rt::ext_matmul(&s_i, &b_j)), format!("{}", rt::matmul(&s_j, &b_j)));
    }

    #[test]
    fn test_ext_matmul_same_dtype_matches_matmul() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::tensor_from_nested!([[1.0f64, 2.0], [3.0, 4.0]], &device);
        let b = rt::tensor_from_nested!([[5.0f64, 6.0], [7.0, 8.0]], &device);
        assert_eq!(format!("{}", rt::ext_matmul(&a, &b)), format!("{}", rt::matmul(&a, &b)));
    }
}
