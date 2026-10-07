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
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_argsort (L2607)
        // np.argsort([3, 1, 2]) == [1, 2, 0]; and arange(101)[::-1] argsorts to
        // its own reversal.
        crate::specify_test!("test_argsort_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([3_i64, 1, 2], &device);
        let expected = rt::tensor_from_nested!([1_usize, 2, 0], &device);
        assert_equal(rt::argsort((&a, ())), &expected, None);

        let b = rt::arange((101, &device));
        assert_eq!(rt::argsort((&b, ())).to_vec(), (0..101).collect::<Vec<usize>>());
        let br = rt::flip(&b, 0);
        assert_eq!(rt::argsort((&br, ())).to_vec(), (0..101).rev().collect::<Vec<usize>>());
    }

    #[test]
    fn test_argsort_nan_last() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_argsort (L2607)
        // argsort analog of the NaN-last sort order:
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
mod numpy_argsort_descending {
    use super::*;
    static FUNC: &str = "numpy_argsort_descending";

    #[test]
    fn test_argsort_descending_signed() {
        // numpy: v2.5.2 |
        // _core/tests/test_multiarray.py::TestMethods::test_argsort_descending_signed (L3001)
        // distinct values: the permutation is the reversal of the ascending one.
        crate::specify_test!("test_argsort_descending_signed");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((-51, 50, &device));
        assert_eq!(rt::argsort((&a, (0, true))).to_vec(), (0..101).rev().collect::<Vec<usize>>());
    }

    #[test]
    fn test_argsort_descending_unsigned() {
        // numpy: v2.5.2 |
        // _core/tests/test_multiarray.py::TestMethods::test_argsort_descending_unsigned (L3008)
        crate::specify_test!("test_argsort_descending_unsigned");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<u32, _> = rt::arange((0_u32, 101, &device));
        assert_eq!(rt::argsort((&a, (0, true))).to_vec(), (0..101).rev().collect::<Vec<usize>>());
    }

    #[test]
    fn test_argsort_descending_floats() {
        // numpy: v2.5.2 |
        // _core/tests/test_multiarray.py::TestMethods::test_argsort_descending_floats (L3039)
        // gathering with the descending permutation yields descending finite
        // values with all NaNs at the end.
        crate::specify_test!("test_argsort_descending_floats");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let mut v = rt::arange((-50.0_f64, 50.0, &device)).to_vec();
        for i in (0..v.len()).step_by(10) {
            v[i] = f64::NAN;
        }
        let a = rt::asarray((v, &device));
        let sorted: Vec<f64> = rt::argsort((&a, (0, true))).to_vec().iter().map(|&i| a.i(i).to_scalar()).collect();
        assert!(sorted[..90].windows(2).all(|w| w[0] >= w[1]));
        assert!(sorted[90..].iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_argsort_stable_bool_int_duplicates() {
        // numpy: v2.5.2 |
        // _core/tests/test_multiarray.py::TestMethods::test_argsort_stable_bool_int_duplicates
        // (L3018) a = [min, 1, max] * 2; expected = sorted(range(n), key=a[i],
        // reverse=descending)
        crate::specify_test!("test_argsort_stable_bool_int_duplicates");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([i8::MIN, 1, i8::MAX, i8::MIN, 1, i8::MAX], &device);
        let asc = rt::tensor_from_nested!([0_usize, 3, 1, 4, 2, 5], &device);
        assert_equal(rt::argsort((&a, (0, false, true))), &asc, None);
        let desc = rt::tensor_from_nested!([2_usize, 5, 1, 4, 0, 3], &device);
        assert_equal(rt::argsort((&a, (0, true, true))), &desc, None);
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
