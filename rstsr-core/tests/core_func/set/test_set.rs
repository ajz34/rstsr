#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod custom_unique {
    use super::*;
    use num::Complex;
    static FUNC: &str = "custom_unique";

    #[test]
    fn test_unique_values_basic() {
        // first-occurrence order over the row-major flatten
        crate::specify_test!("test_unique_values_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // int dtype: sorted ascending (fast path)
        let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);
        let expected = rt::tensor_from_nested!([1, 2, 3], &device);
        assert_equal(rt::unique_values(&a), &expected, None);
    }

    #[test]
    fn test_unique_values_2d_flatten_c_order() {
        // flattening visits row-major even for strided input
        crate::specify_test!("test_unique_values_2d_flatten_c_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t(); // flat C-order: 0 3 1 4 2 5
        let out = rt::unique_values(&at);
        // all distinct; sorted ascending for this dtype
        assert_eq!(out.to_vec(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_all_fields() {
        // indices: first occurrence (flat C-order); inverse: input shape;
        // counts: multiplicities aligned with values
        crate::specify_test!("test_unique_all_fields");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[3, 1, 3], [1, 3, 1]], &device);
        let res = rt::unique_all(&a);
        // sorted ascending: values [1, 3]
        assert_eq!(res.values.to_vec(), vec![1, 3]);
        // first occurrence of 1 is flat 1, of 3 is flat 0
        assert_eq!(res.indices.to_vec(), vec![1, 0]);
        assert_eq!(res.inverse_indices.shape(), a.shape());
        // 3->slot 1, 1->slot 0
        assert_eq!(res.inverse_indices.reshape([-1]).to_vec(), vec![1, 0, 1, 0, 1, 0]);
        assert_eq!(res.counts.to_vec(), vec![3, 3]);
    }

    #[test]
    fn test_unique_nan_distinct() {
        // each NaN is its own unique entry (count 1 each)
        crate::specify_test!("test_unique_nan_distinct");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([f64::NAN, 1.0, f64::NAN], &device);
        let res = rt::unique_all(&a);
        let v = res.values.to_vec();
        assert_eq!(v.len(), 3);
        // NaN tail: 1.0 first, then the two distinct NaNs
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
        assert!(v[2].is_nan());
        assert_eq!(res.counts.to_vec(), vec![1, 1, 1]);
        assert_eq!(res.indices.to_vec(), vec![1, 0, 2]);
    }

    #[test]
    fn test_unique_signed_zero_merge() {
        // -0.0 == 0.0: merged, first-seen encoding kept
        crate::specify_test!("test_unique_signed_zero_merge");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([-0.0_f64, 1.0, 0.0], &device);
        let res = rt::unique_all(&a);
        assert_eq!(res.values.to_vec().len(), 2);
        assert_eq!(res.values.to_vec()[0], 0.0); // -0.0 == 0.0, sorted first
        assert_eq!(res.values.to_vec()[1], 1.0);
        assert_eq!(res.counts.to_vec(), vec![2, 1]);
    }

    #[test]
    fn test_unique_counts_and_inverse() {
        crate::specify_test!("test_unique_counts_and_inverse");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);
        let counts = rt::unique_counts(&a);
        assert_eq!(counts.values.to_vec(), vec![1, 2, 3]);
        assert_eq!(counts.counts.to_vec(), vec![2, 1, 2]);
        let (values, counts_tuple): (Tensor<i32, _, _>, Tensor<usize, _, _>) = rt::unique_counts(&a).into();
        assert_eq!(values.to_vec(), vec![1, 2, 3]);
        assert_eq!(counts_tuple.to_vec(), vec![2, 1, 2]);

        let inv = rt::unique_inverse(&a);
        assert_eq!(inv.values.to_vec(), vec![1, 2, 3]);
        // 3->slot 2, 1->slot 0, 3->2, 2->1, 1->0
        assert_eq!(inv.inverse_indices.to_vec(), vec![2, 0, 2, 1, 0]);
        let (values2, inverse): (Tensor<i32, _, _>, Tensor<usize, _, _>) = rt::unique_inverse(&a).into();
        assert_eq!(values2.to_vec(), vec![1, 2, 3]);
        assert_eq!(inverse.to_vec(), vec![2, 0, 2, 1, 0]);
    }

    #[test]
    fn test_unique_complex_first_occurrence() {
        // complex entries compare with == (PartialEq), first occurrence kept
        crate::specify_test!("test_unique_complex_first_occurrence");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a =
            rt::asarray((vec![Complex::new(1.0_f64, 2.0), Complex::new(1.0, 2.0), Complex::new(0.0, 0.0)], &device));
        let out = rt::unique_values(&a);
        assert_eq!(out.to_vec().len(), 2);
    }

    #[test]
    fn test_unique_empty() {
        crate::specify_test!("test_unique_empty");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i32, _> = rt::zeros(([0], &device));
        let out = rt::unique_values(&a);
        assert_eq!(out.shape(), &[0]);
    }
}

#[cfg(test)]
mod custom_isin {
    use super::*;
    static FUNC: &str = "custom_isin";

    #[test]
    fn test_isin_basic() {
        crate::specify_test!("test_isin_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let b = rt::tensor_from_nested!([2, 4], &device);
        assert_eq!(rt::isin((&a, &b, false)).to_vec(), vec![false, true, false, true]);
    }

    #[test]
    fn test_isin_invert() {
        crate::specify_test!("test_isin_invert");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let b = rt::tensor_from_nested!([2, 4], &device);
        assert_eq!(rt::isin((&a, &b, true)).to_vec(), vec![true, false, true, false]);
    }

    #[test]
    fn test_isin_shape_of_x1() {
        // output has x1's shape; x2 may be any shape
        crate::specify_test!("test_isin_shape_of_x1");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let b = rt::tensor_from_nested!([[1, 3, 5]], &device);
        let out = rt::isin((&a, &b, false));
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![false, true, false, true, false, true]);
    }

    #[test]
    fn test_isin_nan_membership() {
        // membership is value equality: NaN is never a member (NumPy:
        // isin([nan], [nan]) is [false]), including complex NaN-bearing keys
        crate::specify_test!("test_isin_nan_membership");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([f64::NAN, 1.0, f64::NAN], &device);
        let b = rt::tensor_from_nested!([f64::NAN, 1.0], &device);
        let out = rt::isin((&a, &b, false));
        assert_eq!(out.to_vec(), vec![false, true, false]);

        let c1 = rt::asarray((vec![num::Complex::new(0.0_f64, f64::NAN)], &device));
        let c2 = rt::asarray((vec![num::Complex::new(1.0_f64, 0.0), num::Complex::new(2.0, f64::NAN)], &device));
        let out = rt::isin((&c1, &c2, false));
        assert_eq!(out.to_vec(), vec![false]);

        // invert flips the NaN verdict to true
        let out = rt::isin((&a, &b, true));
        assert_eq!(out.to_vec(), vec![true, false, true]);
    }

    #[test]
    fn test_isin_duplicates_in_x2() {
        // duplicated x2 entries are fine; membership is a set question
        crate::specify_test!("test_isin_duplicates_in_x2");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([2, 2, 2], &device);
        let out = rt::isin((&a, &b, false));
        assert_eq!(out.to_vec(), vec![false, true, false]);
    }
}
