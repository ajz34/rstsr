//! Nonzero tests: NumPy-cited coordinate contract (TestNonzero) + custom edges.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_nonzero {
    use super::*;
    static FUNC: &str = "numpy_nonzero";

    #[test]
    fn test_nonzero_trivial() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestNonzero::test_nonzero_trivial (L1638)
        crate::specify_test!("test_nonzero_trivial");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.nonzero(np.array([])) == ([],)
        let empty: Tensor<i32, _> = rt::zeros(([0], &device));
        assert_eq!(rt::nonzero(&empty)[0].to_vec(), Vec::<usize>::new());
        // np.nonzero(np.array([0])) == ([],)
        let z = rt::tensor_from_nested!([0], &device);
        assert_eq!(rt::nonzero(&z)[0].to_vec(), Vec::<usize>::new());
        // np.nonzero(np.array([1])) == ([0],)
        let o = rt::tensor_from_nested!([1], &device);
        assert_eq!(rt::nonzero(&o)[0].to_vec(), vec![0]);
    }

    #[test]
    fn test_nonzero_zerodim() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestNonzero::test_nonzero_zerodim (L1651)
        // np.nonzero(np.array(0)) raises ValueError ("Calling nonzero on 0d
        // arrays is not allowed"); rstsr raises too (nonzero requires ndim > 0).
        crate::specify_test!("test_nonzero_zerodim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i32, _> = rt::full(([], 0, &device));
        assert!(rt::nonzero_f(&a).is_err());
    }

    #[test]
    fn test_nonzero_onedim() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestNonzero::test_nonzero_onedim (L1658)
        crate::specify_test!("test_nonzero_onedim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.nonzero([1, 0, 2, -1, 0, 0, 8]) == ([0, 2, 3, 6],)
        let x = rt::tensor_from_nested!([1, 0, 2, -1, 0, 0, 8], &device);
        assert_eq!(rt::nonzero(&x)[0].to_vec(), vec![0, 2, 3, 6]);
    }

    #[test]
    fn test_nonzero_twodim() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestNonzero::test_nonzero_twodim (L1677)
        crate::specify_test!("test_nonzero_twodim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.nonzero([[0, 1, 0], [2, 0, 3]]) == ([0, 1, 1], [1, 0, 2])
        let x = rt::tensor_from_nested!([[0, 1, 0], [2, 0, 3]], &device);
        let c = rt::nonzero(&x);
        assert_eq!(c[0].to_vec(), vec![0, 1, 1]);
        assert_eq!(c[1].to_vec(), vec![1, 0, 2]);

        // np.nonzero(np.eye(3)) == ([0, 1, 2], [0, 1, 2])
        let e: Tensor<i32, _> = rt::eye((3, &device));
        let c = rt::nonzero(&e);
        assert_eq!(c[0].to_vec(), vec![0, 1, 2]);
        assert_eq!(c[1].to_vec(), vec![0, 1, 2]);
    }

    #[test]
    fn test_nonzero_dtypes() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestNonzero::test_nonzero_integer_dtypes
        // (L1729) NumPy compares nonzero against where(x != 0) over a random array; the
        // dtype sweep (bool/int/uint) is the transferable part.
        crate::specify_test!("test_nonzero_dtypes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a8 = rt::asarray((vec![1_i8, 0, 1, 0], &device));
        assert_eq!(rt::nonzero(&a8)[0].to_vec(), vec![0, 2]);
        let a16 = rt::asarray((vec![1_i16, 0, 2, 0], &device));
        assert_eq!(rt::nonzero(&a16)[0].to_vec(), vec![0, 2]);
        let a32 = rt::asarray((vec![1_i32, 0, 3, 0], &device));
        assert_eq!(rt::nonzero(&a32)[0].to_vec(), vec![0, 2]);
        let a64 = rt::asarray((vec![1_i64, 0, 4, 0], &device));
        assert_eq!(rt::nonzero(&a64)[0].to_vec(), vec![0, 2]);
        let u8 = rt::asarray((vec![1_u8, 0, 5, 0], &device));
        assert_eq!(rt::nonzero(&u8)[0].to_vec(), vec![0, 2]);
        let u32 = rt::asarray((vec![1_u32, 0, 6, 0], &device));
        assert_eq!(rt::nonzero(&u32)[0].to_vec(), vec![0, 2]);
        let b = rt::tensor_from_nested!([true, false, true, false], &device);
        assert_eq!(rt::nonzero(&b)[0].to_vec(), vec![0, 2]);
    }
}

#[cfg(test)]
mod custom_nonzero {
    use super::*;
    static FUNC: &str = "custom_nonzero";

    #[test]
    fn test_nonzero_2d() {
        // np.nonzero([[1, 0, 2], [0, 3, 0]]) -> (array([0, 0, 1]), array([0, 2, 1]))
        crate::specify_test!("test_nonzero_2d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 0, 2], [0, 3, 0]], &device);
        let coords = rt::nonzero(&a);
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0].to_vec(), vec![0, 0, 1]);
        assert_eq!(coords[1].to_vec(), vec![0, 2, 1]);
    }

    #[test]
    fn test_nonzero_row_major_order() {
        // strict row-major element order even for strided input
        // at = arange(6).reshape(2,3).T: content [0 3; 1 4; 2 5]
        // nonzero -> coords ([0,1,1,2,2],[1,0,2,0,2]) per row-major walk
        crate::specify_test!("test_nonzero_row_major_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t();
        let coords = rt::nonzero(&at);
        assert_eq!(coords[0].to_vec(), vec![0, 1, 1, 2, 2]);
        assert_eq!(coords[1].to_vec(), vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_nonzero_bool_complex() {
        // bool: true is nonzero; complex: either component nonzero
        crate::specify_test!("test_nonzero_bool_complex");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([true, false, true], &device);
        let coords = rt::nonzero(&a);
        assert_eq!(coords[0].to_vec(), vec![0, 2]);

        let c = rt::asarray((
            vec![num::Complex::new(0.0_f64, 1.0), num::Complex::new(0.0, 0.0), num::Complex::new(2.0, 0.0)],
            &device,
        ));
        let coords = rt::nonzero(&c);
        assert_eq!(coords[0].to_vec(), vec![0, 2]);
    }

    #[test]
    fn test_nonzero_all_zero_and_all_nonzero() {
        crate::specify_test!("test_nonzero_all_zero_and_all_nonzero");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let z: Tensor<i32, _> = rt::zeros(([2, 2], &device));
        let coords = rt::nonzero(&z);
        assert_eq!(coords[0].shape(), &[0]);

        let o: Tensor<i32, _> = rt::ones(([2, 2], &device));
        let coords = rt::nonzero(&o);
        assert_eq!(coords[0].to_vec(), vec![0, 0, 1, 1]);
        assert_eq!(coords[1].to_vec(), vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_nonzero_1d_and_nan() {
        // 1-d returns one tensor; NaN is nonzero (NaN != 0)
        crate::specify_test!("test_nonzero_1d_and_nan");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([0.0, 5.0, 0.0], &device);
        let coords = rt::nonzero(&a);
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0].to_vec(), vec![1]);

        let n = rt::tensor_from_nested!([f64::NAN, 0.0], &device);
        let coords = rt::nonzero(&n);
        assert_eq!(coords[0].to_vec(), vec![0]);
    }
}
