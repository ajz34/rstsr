//! Set tests: NumPy-cited unique_*/isin (TestUnique / TestSetOps) + custom edges.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_unique {
    use super::*;
    use num::Complex;
    static FUNC: &str = "numpy_unique";

    #[test]
    fn test_unique_1d() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestUnique::test_unique_1d (L700)
        crate::specify_test!("test_unique_1d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = [5, 7, 1, 2, 1, 5, 7] * 10; b = [1, 2, 5, 7];
        // i1 = [2, 3, 0, 1]; counts = [20, 10, 20, 20]; i2 = [2,3,0,1,0,2,3] * 10
        let base = [5_i32, 7, 1, 2, 1, 5, 7];
        let mut a = Vec::new();
        for _ in 0..10 {
            a.extend_from_slice(&base);
        }
        let a = rt::asarray((a, &device));
        let res = rt::unique_all(&a);
        assert_eq!(res.values.to_vec(), vec![1, 2, 5, 7]);
        assert_eq!(res.indices.to_vec(), vec![2_usize, 3, 0, 1]);
        assert_eq!(res.counts.to_vec(), vec![20_usize, 10, 20, 20]);
        let inv_base = [2_usize, 3, 0, 1, 0, 2, 3];
        let mut inv_expected = Vec::new();
        for _ in 0..10 {
            inv_expected.extend_from_slice(&inv_base);
        }
        assert_eq!(res.inverse_indices.reshape([-1]).to_vec(), inv_expected);
    }

    #[test]
    fn test_unique_zero_sized() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestUnique::test_unique_zero_sized (L822)
        crate::specify_test!("test_unique_zero_sized");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i32, _> = rt::zeros(([0], &device));
        let res = rt::unique_all(&a);
        assert_eq!(res.values.shape(), &[0]);
        assert_eq!(res.indices.shape(), &[0]);
        assert_eq!(res.counts.shape(), &[0]);
        assert_eq!(res.inverse_indices.shape(), &[0]);
    }

    #[test]
    fn test_unique_nanequals() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestUnique::test_unique_nanequals (L1200)
        // np.unique([1,1,nan,nan,nan], equal_nan=False) == [1, nan, nan, nan];
        // rstsr mirrors the array-api aliases (distinct NaNs).
        crate::specify_test!("test_unique_nanequals");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1.0_f64, 1.0, f64::NAN, f64::NAN, f64::NAN], &device);
        let res = rt::unique_all(&a);
        let v = res.values.to_vec();
        assert_eq!(v.len(), 4);
        assert_eq!(v[0], 1.0);
        assert!(v[1..].iter().all(|x| x.is_nan()));
        assert_eq!(res.counts.to_vec(), vec![2_usize, 1, 1, 1]);
    }

    #[test]
    fn test_unique_array_api_functions() {
        // numpy: v2.5.2 |
        // lib/tests/test_arraysetops.py::TestUnique::test_unique_array_api_functions (L1208)
        // The NumPy test compares the array-api aliases against
        // np.unique(..., equal_nan=False); the alias order is not guaranteed
        // (the test sorts before comparing), so rstsr's ascending order with a
        // distinct-NaN tail is checked here.
        crate::specify_test!("test_unique_array_api_functions");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let arr =
            vec![f64::NAN, 1.0, 0.0, 4.0, -f64::NAN, -0.0, 1.0, 3.0, 4.0, f64::NAN, 5.0, -0.0, 1.0, -f64::NAN, 0.0];
        let a = rt::asarray((arr, &device));

        let res = rt::unique_all(&a);
        assert_eq!(res.values.to_vec().len(), 9);
        assert_eq!(res.counts.to_vec(), vec![4_usize, 3, 1, 2, 1, 1, 1, 1, 1]);
        let v = res.values.to_vec();
        assert!(v[5..].iter().all(|x| x.is_nan()));
        let mut finite = v[..5].to_vec();
        finite.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(finite, vec![0.0, 1.0, 3.0, 4.0, 5.0]);

        // unique_counts agrees with unique_all's counts
        assert_eq!(rt::unique_counts(&a).counts.to_vec(), res.counts.to_vec());
    }

    #[test]
    fn test_unique_inverse_shape() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestUnique::test_unique_inverse_shape
        // (L1251) https://github.com/numpy/numpy/issues/25552
        crate::specify_test!("test_unique_inverse_shape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let arr = rt::tensor_from_nested!([[1, 2, 3], [2, 3, 1]], &device);
        for res in [rt::unique_inverse(&arr), {
            let a = rt::unique_all(&arr);
            UniqueInverse { values: a.values, inverse_indices: a.inverse_indices }
        }] {
            assert_eq!(res.values.to_vec(), vec![1, 2, 3]);
            assert_eq!(res.inverse_indices.shape(), &[2, 3]);
            // arr == values[inverse_indices]
            let vals = res.values.to_vec();
            let gathered: Vec<i32> = res.inverse_indices.reshape([-1]).to_vec().iter().map(|&i| vals[i]).collect();
            assert_eq!(gathered, arr.reshape([-1]).to_vec());
        }
    }

    #[test]
    fn test_unique_complex_signed_zeros() {
        // numpy: v2.5.2 |
        // lib/tests/test_arraysetops.py::TestUnique::test_unique_complex_signed_zeros (L1301)
        // z = [0.-1j, -0.-1j, 0]; the two signed-zero-imag entries compare
        // equal, so the unique length is len(values) - 1.
        crate::specify_test!("test_unique_complex_signed_zeros");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let z =
            rt::asarray((vec![Complex::new(0.0_f64, -1.0), Complex::new(-0.0, -1.0), Complex::new(0.0, 0.0)], &device));
        assert_eq!(rt::unique_values(&z).to_vec().len(), 2);
    }
}

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
        // first-seen encoding is -0.0 (the first input element) — assert the sign
        assert_eq!(res.values.to_vec()[0], 0.0);
        assert!(res.values.to_vec()[0].is_sign_negative());
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
        // complex entries compare with == (PartialEq); the naive path keeps
        // first-occurrence order, so [1+2j, 1+2j, 0j] -> [1+2j, 0j]
        crate::specify_test!("test_unique_complex_first_occurrence");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a =
            rt::asarray((vec![Complex::new(1.0_f64, 2.0), Complex::new(1.0, 2.0), Complex::new(0.0, 0.0)], &device));
        let out = rt::unique_values(&a);
        assert_eq!(out.to_vec(), vec![Complex::new(1.0, 2.0), Complex::new(0.0, 0.0)]);
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

    #[test]
    fn test_unique_custom_partial_eq_type() {
        // a user dtype with only `Clone + PartialEq` works through the general
        // first-occurrence path (no ExtSortCmp bound at the API)
        crate::specify_test!("test_unique_custom_partial_eq_type");

        #[derive(Clone, PartialEq, Debug)]
        struct Tag(i32);

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::asarray((vec![Tag(2), Tag(1), Tag(2), Tag(3), Tag(1)], &device));
        assert_eq!(rt::unique_values(&a).to_vec(), vec![Tag(2), Tag(1), Tag(3)]);
        let res = rt::unique_all(&a);
        assert_eq!(res.values.to_vec(), vec![Tag(2), Tag(1), Tag(3)]);
        assert_eq!(res.counts.to_vec(), vec![2_usize, 2, 1]);
        assert_eq!(res.indices.to_vec(), vec![0_usize, 1, 3]);
    }

    #[test]
    fn test_unique_values_ascending_dtype_sweep() {
        // the sorted fast path is unrolled per dtype (TypeId dispatch); pin
        // ascending order across the listed scalar dtypes
        crate::specify_test!("test_unique_values_ascending_dtype_sweep");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        macro_rules! check {
            ($($v:expr),* $(,)?) => {{
                let t = rt::asarray((vec![$($v),*], &device));
                assert_eq!(rt::unique_values(&t).to_vec(), vec![1, 2, 3]);
            }};
        }
        check!(3_i8, 1_i8, 2_i8);
        check!(3_i16, 1_i16, 2_i16);
        check!(3_i32, 1_i32, 2_i32);
        check!(3_i64, 1_i64, 2_i64);
        check!(3_isize, 1_isize, 2_isize);
        check!(3_u8, 1_u8, 2_u8);
        check!(3_u16, 1_u16, 2_u16);
        check!(3_u32, 1_u32, 2_u32);
        check!(3_u64, 1_u64, 2_u64);
        check!(3_usize, 1_usize, 2_usize);
        check!(3_i128, 1_i128, 2_i128);
        check!(3_u128, 1_u128, 2_u128);
        let f32v = rt::asarray((vec![3.0_f32, 1.0, 2.0], &device));
        assert_eq!(rt::unique_values(&f32v).to_vec(), vec![1.0_f32, 2.0, 3.0]);
        let f64v = rt::asarray((vec![3.0_f64, 1.0, 2.0], &device));
        assert_eq!(rt::unique_values(&f64v).to_vec(), vec![1.0_f64, 2.0, 3.0]);
        let b = rt::asarray((vec![true, false, true], &device));
        assert_eq!(rt::unique_values(&b).to_vec(), vec![false, true]);
    }
}

#[cfg(test)]
mod numpy_isin {
    use super::*;
    static FUNC: &str = "numpy_isin";

    #[test]
    fn test_isin() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestSetOps::test_isin (L218)
        // multidimensional arrays in both arguments; empty-array cases.
        crate::specify_test!("test_isin");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        let b = rt::tensor_from_nested!([[10, 20, 30], [0, 1, 3], [11, 22, 33]], &device);
        let out = rt::isin((&a, &b, false));
        assert_eq!(out.shape(), &[2, 3, 4]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![
            true, true, false, true, false, false, false, false, false, false, true, true, false, false, false, false,
            false, false, false, false, true, false, true, false
        ]);

        // empty x1 / empty x2 give all-false
        let empty: Tensor<i32, _> = rt::zeros(([0], &device));
        let ar = rt::tensor_from_nested!([10, 20, 30], &device);
        assert_eq!(rt::isin((&empty, &ar, false)).shape(), &[0]);
        assert_eq!(rt::isin((&ar, &empty, false)).to_vec(), vec![false, false, false]);
    }

    #[test]
    fn test_isin_invert() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestSetOps::test_isin_invert (L347)
        crate::specify_test!("test_isin_invert");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], &device);
        let b = rt::tensor_from_nested!([2, 3, 4], &device);
        let normal = rt::isin((&a, &b, false)).to_vec();
        let inverted = rt::isin((&a, &b, true)).to_vec();
        assert_eq!(normal, vec![
            false, true, false, true, true, true, true, true, true, false, true, false, false, false
        ]);
        assert_eq!(inverted, normal.iter().map(|x| !x).collect::<Vec<_>>());
    }

    #[test]
    fn test_isin_boolean() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestSetOps::test_isin_boolean (L384)
        crate::specify_test!("test_isin_boolean");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([true, false], &device);
        let b = rt::tensor_from_nested!([false, false, false], &device);
        assert_eq!(rt::isin((&a, &b, false)).to_vec(), vec![false, true]);
        assert_eq!(rt::isin((&a, &b, true)).to_vec(), vec![true, false]);
    }

    #[test]
    fn test_isin_errors() {
        // numpy: v2.5.2 | lib/tests/test_arraysetops.py::TestSetOps::test_isin_errors (L539)
        // The `kind=` error cases are N/A (rstsr isin has no `kind`); the
        // non-error overflow case (kind=None -> sort path) transfers.
        crate::specify_test!("test_isin_errors");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let ar1 = rt::tensor_from_nested!([-1, 2, 3, 4, 5], &device);
        let ar2 = rt::tensor_from_nested!([-1, i32::MAX], &device);
        assert_eq!(rt::isin((&ar1, &ar2, false)).to_vec(), vec![true, false, false, false, false]);
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

    #[test]
    fn test_isin_custom_partial_eq_type() {
        // membership is equality only: a user dtype with `Clone + PartialEq`
        // works through the general linear-scan path
        crate::specify_test!("test_isin_custom_partial_eq_type");

        #[derive(Clone, PartialEq, Debug)]
        struct Tag(i32);

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x1 = rt::asarray((vec![Tag(1), Tag(3), Tag(2)], &device));
        let x2 = rt::asarray((vec![Tag(3), Tag(3)], &device));
        assert_eq!(rt::isin((&x1, &x2, false)).to_vec(), vec![false, true, false]);
        assert_eq!(rt::isin((&x1, &x2, true)).to_vec(), vec![true, false, true]);
    }
}

#[cfg(test)]
mod device_order {
    use super::*;
    use num::Complex;
    static FUNC: &str = "device_order";

    #[test]
    fn test_unique_all_first_occurrence_col_major() {
        crate::specify_test!("test_unique_all_first_occurrence_col_major");

        // the first-occurrence visit sequence (and the flat `indices`) follow
        // the device default order
        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        // values [7 7 8 9 8 9] filled in device order: [[7 8 8], [7 9 9]]
        let a = rt::asarray((vec![7_i32, 7, 8, 9, 8, 9], &device)).into_shape([2, 3]);
        let u = rt::unique_all(&a);
        assert_eq!(u.values.to_vec(), vec![7, 8, 9]); // ascending is value-defined
        assert_eq!(u.counts.to_vec(), vec![2, 2, 2]);
        assert_eq!(u.indices.to_vec(), vec![0, 2, 3]); // col-major first occurrence
        assert_eq!(u.inverse_indices.stride(), &[1, 2]); // device-order contig
        assert_eq!(format!("{}", u.inverse_indices), "[[ 0 1 1]\n [ 0 2 2]]");
    }

    #[test]
    fn test_unique_values_complex_col_major() {
        crate::specify_test!("test_unique_values_complex_col_major");

        // complex takes the first-occurrence path (no `ExtSortCmp` order):
        // the sequence follows the visit order, and the two orders differ
        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let vals = (0..6).map(|i| Complex::new(i as f64, 0.0)).collect::<Vec<_>>();
        let a = rt::asarray((vals, &device)).into_shape([2, 3]); // [[0 2 4], [1 3 5]]
        let out = rt::unique_values(&a);
        assert_eq!(out.to_vec(), (0..6).map(|i| Complex::new(i as f64, 0.0)).collect::<Vec<_>>());

        // the same logical tensor filled row-major visits [0 2 4 1 3 5] instead:
        // the first-occurrence sequence genuinely depends on the visit order
        let mut device_rm = TESTCFG.device.clone();
        device_rm.set_default_order(RowMajor);
        let vals_rm: Vec<Complex<f64>> = [0.0, 2.0, 4.0, 1.0, 3.0, 5.0].iter().map(|&i| Complex::new(i, 0.0)).collect();
        let a_rm = rt::asarray((vals_rm, &device_rm)).into_shape([2, 3]);
        assert_eq!(format!("{a}"), format!("{a_rm}"));
        let out_rm = rt::unique_values(&a_rm);
        let expected_rm: Vec<Complex<f64>> =
            [0.0, 2.0, 4.0, 1.0, 3.0, 5.0].iter().map(|&i| Complex::new(i, 0.0)).collect();
        assert_eq!(out_rm.to_vec(), expected_rm);
    }

    #[test]
    fn test_isin_arrangement_col_major() {
        crate::specify_test!("test_isin_arrangement_col_major");

        // values are position-mapped (order-independent); the memory
        // arrangement of the output follows the device default order
        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]); // [[0 2 4], [1 3 5]]
        let b = rt::asarray((vec![2_i32, 3], &device));
        let out = rt::isin((&a, &b, false));
        assert_eq!(out.stride(), &[1, 2]);
        // logical [[false, true, false], [false, true, false]]
        assert_eq!(format!("{out}"), "[[ false true false]\n [ false true false]]");
    }
}
