//! Argsort tests: NumPy-cited cases plus custom stability/NaN edges. Most
//! value-ordering behavior is covered in `test_sort.rs`; this file focuses on
//! the index permutation contract.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_argsort {
    use super::*;
    static FUNC: &str = "numpy_argsort";

    #[test]
    fn test_argsort_basic() {
        // NumPy: np.argsort([3, 1, 2]) == [1, 2, 0]; dtype int64/index
        crate::specify_test!("test_argsort_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([3_i64, 1, 2], &device);
        let expected = rt::tensor_from_nested!([1_usize, 2, 0], &device);
        assert_equal(rt::argsort((&a, ())), &expected, None);
    }

    #[test]
    fn test_argsort_nan_last() {
        // NumPy v2.5.2, TestMethods::test_sort (line 2267), argsort analog:
        // np.argsort([nan, 1, 0]) == [2, 1, 0] (NaN last ascending)
        crate::specify_test!("test_argsort_nan_last");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([f64::NAN, 1.0, 0.0], &device);
        assert_eq!(rt::argsort((&a, ())).to_vec(), vec![2, 1, 0]);
        // descending: values [1, 0, nan] -> indices [1, 2, 0] (NaN last)
        assert_eq!(rt::argsort((&a, (0, true))).to_vec(), vec![1, 2, 0]);
    }
}

#[cfg(test)]
mod custom_argsort {
    use super::*;
    static FUNC: &str = "custom_argsort";

    #[test]
    fn test_argsort_shape_and_axis() {
        crate::specify_test!("test_argsort_shape_and_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        for axis in [-2_isize, -1, 0, 1] {
            let out = a.argsort(axis);
            assert_eq!(out.shape(), a.shape());
        }
    }

    #[test]
    fn test_argsort_gather_roundtrip() {
        // a[i, argsort[i, j]] is non-decreasing along j (ascending sort)
        crate::specify_test!("test_argsort_gather_roundtrip");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = (rt::arange((20, &device)) * 7).mapv(|x| x % 11).into_shape([4, 5]);
        let idx = a.argsort(1);
        for i in 0..4 {
            let mut prev = i32::MIN;
            for j in 0..5 {
                let v = a.i((i, idx.i((i, j)).to_scalar())).to_scalar();
                assert!(v >= prev);
                prev = v;
            }
        }
    }
}
