#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_sort {
    use super::*;
    static FUNC: &str = "numpy_sort";

    #[test]
    fn test_sort_nan_order() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort (L2267)
        // real part: np.sort([nan, 1, 0]) == [nan, 1, 0][::-1] == [0, 1, nan]
        // (NaN sorts to the END of the ascending order)
        crate::specify_test!("test_sort_nan_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([f64::NAN, 1.0, 0.0], &device);
        let v = a.sort(()).to_vec();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 1.0);
        assert!(v[2].is_nan());
    }

    #[test]
    fn test_sort_unsigned() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_unsigned (L2301)
        // a = arange(101); b = a[::-1]; sort(b) == a
        crate::specify_test!("test_sort_unsigned");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((101, &device));
        let b = rt::flip(&a, 0);
        assert_equal(rt::sort((&b, ())), &a, None);
    }

    #[test]
    fn test_sort_2d_axis() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_axis (L2426)
        // sorting is per-line along the given axis
        // np.sort([[3, 1, 2], [6, 4, 5]], axis=0) == [[3, 1, 2], [6, 4, 5]]
        // np.sort(..., axis=1) == [[1, 2, 3], [4, 5, 6]]
        crate::specify_test!("test_sort_2d_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[3, 1, 2], [6, 4, 5]], &device);
        let expected0 = rt::tensor_from_nested!([[3, 1, 2], [6, 4, 5]], &device);
        assert_equal(rt::sort((&a, 0)), &expected0, None);
        let expected1 = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        assert_equal(rt::sort((&a, 1)), &expected1, None);
        let expected_last = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        assert_equal(rt::sort((&a, ())), &expected_last, None);
    }

    #[test]
    fn test_sort_descending() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_descending_floats
        // (L2842) np.sort(a, descending=True) reverses value order but keeps NaN last
        // (numpy_tag.h: "NaN sorts to the end in reverse too")
        crate::specify_test!("test_sort_descending");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.sort([3, nan, 1, 2], descending=True) -> [3, 2, 1, nan]
        let a = rt::tensor_from_nested!([3.0, f64::NAN, 1.0, 2.0], &device);
        let v = a.sort((0, true)).to_vec();
        assert_eq!(v[..3], [3.0, 2.0, 1.0]);
        assert!(v[3].is_nan());

        // np.sort([3, nan, 1, 2]) -> [1, 2, 3, nan]
        let v = a.sort(()).to_vec();
        assert_eq!(v[..3], [1.0, 2.0, 3.0]);
        assert!(v[3].is_nan());
    }

    #[test]
    fn test_sort_signed_negatives() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_signed (L2316)
        // signed values incl. negatives keep numeric order
        crate::specify_test!("test_sort_signed_negatives");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([-5_i32, 3, 0, -1, 7], &device);
        let expected = rt::tensor_from_nested!([-5, -1, 0, 3, 7], &device);
        assert_equal(a.sort(()), &expected, None);
    }
}

#[cfg(test)]
mod numpy_sort_descending {
    use super::*;
    static FUNC: &str = "numpy_sort_descending";

    #[test]
    fn test_sort_descending_signed() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_descending_signed
        // (L2826) ascending input [-51, 50); ascending sort is the identity, descending
        // is its reverse (no NaNs). `stable` does not change distinct values.
        crate::specify_test!("test_sort_descending_signed");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((-51, 50, &device));
        assert_equal(rt::sort((&a, (0, false))), &a, None);
        let rev = rt::flip(&a, 0);
        assert_equal(rt::sort((&a, (0, true))), &rev, None);
        assert_equal(rt::sort((&a, (0, true, true))), &rev, None);
    }

    #[test]
    fn test_sort_descending_unsigned() {
        // numpy: v2.5.2 |
        // _core/tests/test_multiarray.py::TestMethods::test_sort_descending_unsigned (L2833)
        crate::specify_test!("test_sort_descending_unsigned");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<u32, _> = rt::arange((0_u32, 101, &device));
        assert_equal(rt::sort((&a, (0, false))), &a, None);
        assert_equal(rt::sort((&a, (0, true))), rt::flip(&a, 0), None);
    }

    #[test]
    fn test_sort_descending_floats() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_descending_floats
        // (L2842) NaNs sort to the END in both directions.
        crate::specify_test!("test_sort_descending_floats");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let mut v = rt::arange((-50.0_f64, 50.0, &device)).to_vec(); // 100 values
        for i in (0..v.len()).step_by(10) {
            v[i] = f64::NAN;
        }
        let a = rt::asarray((v, &device));

        let asc = rt::sort((&a, ())).to_vec();
        let desc = rt::sort((&a, (0, true))).to_vec();
        assert_eq!(asc.iter().filter(|x| x.is_nan()).count(), 10);
        assert!(asc[..90].windows(2).all(|w| w[0] <= w[1]));
        assert!(asc[90..].iter().all(|x| x.is_nan()));
        assert!(desc[..90].windows(2).all(|w| w[0] >= w[1]));
        assert!(desc[90..].iter().all(|x| x.is_nan()));
        // descending finite part is the reverse of the ascending finite part
        assert_eq!(desc[..90], asc[..90].iter().rev().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn test_sort_size_0() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestMethods::test_sort_size_0 (L2442)
        crate::specify_test!("test_sort_size_0");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::zeros(([0], &device));
        assert_eq!(rt::sort((&a, ())).shape(), &[0]);
    }
}

#[cfg(test)]
mod custom_sort {
    use super::*;
    use num::Complex;
    static FUNC: &str = "custom_sort";

    #[test]
    fn test_stability_ties() {
        // stable sort: ties keep input order; verified through argsort
        // companion (values equal, indices ascending among ties)
        crate::specify_test!("test_stability_ties");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // values [2, 1, 1, 0]: ascending ties are the two 1s in input order
        let a = rt::tensor_from_nested!([2_i32, 1, 1, 0], &device);
        let expected = rt::tensor_from_nested!([3_usize, 1, 2, 0], &device);
        assert_equal(rt::argsort((&a, ())), &expected, None);

        // descending: value comparison flips, ties still input order
        let expected = rt::tensor_from_nested!([0_usize, 1, 2, 3], &device);
        assert_equal(rt::argsort((&a, (0, true))), &expected, None);
    }

    #[test]
    fn test_signed_zero_equal() {
        // -0.0 == 0.0 in the sort order (both between -1 and 1)
        crate::specify_test!("test_signed_zero_equal");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1.0_f64, -0.0, -1.0, 0.0], &device);
        let out = a.sort(());
        let v = out.to_vec();
        assert_eq!(v[0], -1.0);
        assert_eq!(v[3], 1.0);
        // middle two are ±0 in some order; both compare equal to 0.0
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_strided_input() {
        // sorting a transposed (strided) view along its last axis
        crate::specify_test!("test_strided_input");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t(); // shape [3, 2], rows are [0, 3], [1, 4], [2, 5]
        let expected = rt::tensor_from_nested!([[0, 3], [1, 4], [2, 5]], &device);
        assert_equal(rt::sort((&at, ())), &expected, None);
    }

    #[test]
    fn test_axis_none_of_shape() {
        // output shape equals input shape for any axis choice
        crate::specify_test!("test_axis_none_of_shape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
        for axis in [-3_isize, -2, -1, 0, 1, 2] {
            let out = a.sort(axis);
            assert_eq!(out.shape(), a.shape());
        }
    }

    #[test]
    fn test_output_layout_default_order() {
        // output is contiguous in the device default order
        crate::specify_test!("test_output_layout_default_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let out = a.sort(1);
        assert!(out.c_contig());

        let mut device_f = TESTCFG.device.clone();
        device_f.set_default_order(ColMajor);
        let a_f = rt::arange((6, &device_f)).into_shape([2, 3]);
        let out_f = a_f.sort(1);
        assert!(out_f.f_contig());
    }

    #[test]
    fn test_bool_and_int_dtypes() {
        // bool sorts False < True; ints sort numerically
        crate::specify_test!("test_bool_and_int_dtypes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([true, false, true], &device);
        assert_eq!(a.sort(()).to_vec(), vec![false, true, true]);
    }

    #[test]
    fn test_axis_out_of_range() {
        crate::specify_test!("test_axis_out_of_range");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        assert!(a.sort_f(2).is_err());
        assert!(a.sort_f(-3).is_err());
    }

    #[test]
    fn test_empty_rest_dim() {
        // a zero-sized dimension OUTSIDE the sorted axis: no lines to sort,
        // empty output (np.sort(np.zeros((3, 0)), axis=0) has shape (3, 0))
        crate::specify_test!("test_empty_rest_dim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::zeros(([3, 0], &device));
        let out = a.sort(0);
        assert_eq!(out.shape(), &[3, 0]);

        let idx = rt::argsort((&a, 0));
        assert_eq!(idx.shape(), &[3, 0]);

        // zero-sized sorted axis is also an empty result
        let b: Tensor<f64, _> = rt::zeros(([2, 0], &device));
        let out = b.sort(1);
        assert_eq!(out.shape(), &[2, 0]);
    }

    #[test]
    fn test_negative_axis_values() {
        // negative axes normalize to the same result as positive
        crate::specify_test!("test_negative_axis_values");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[3, 1, 2], [6, 4, 5]], &device);
        assert_equal(a.sort(-1), a.sort(1), None);
        assert_equal(a.sort(-2), a.sort(0), None);
    }

    #[test]
    fn test_complex_declined() {
        // complex sort/argsort declined at the tensor layer (array-api
        // restricts sorting to real-valued dtypes); escape hatch sort_custom
        crate::specify_test!("test_complex_declined");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::asarray((vec![Complex::new(1.0_f64, 2.0), Complex::new(0.0, 1.0)], &device));
        assert!(a.sort_f(()).is_err());
        assert!(a.argsort_f(()).is_err());
    }
}

#[cfg(test)]
mod custom_sort_custom {
    use super::*;
    use core::cmp::Ordering;
    use num::Complex;
    static FUNC: &str = "custom_sort_custom";

    #[test]
    fn test_complex_by_norm() {
        // documented escape hatch: complex sorted by squared magnitude
        crate::specify_test!("test_complex_by_norm");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a =
            rt::asarray((vec![Complex::new(3.0_f64, 0.0), Complex::new(1.0, 1.0), Complex::new(0.0, 2.0)], &device));
        let by_norm = |x: &Complex<f64>, y: &Complex<f64>| {
            let n1 = x.re * x.re + x.im * x.im;
            let n2 = y.re * y.re + y.im * y.im;
            n1.partial_cmp(&n2).unwrap_or(Ordering::Equal)
        };
        let out = a.sort_custom(-1, by_norm);
        let v = out.to_vec();
        // |1+1i|^2 = 2 < |2i|^2 = 4 < |3|^2 = 9
        assert_eq!(v[0], Complex::new(1.0, 1.0));
        assert_eq!(v[1], Complex::new(0.0, 2.0));
        assert_eq!(v[2], Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_argsort_custom_matches_sort_custom() {
        crate::specify_test!("test_argsort_custom_matches_sort_custom");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // sort by residue mod 4; the rows are scrambled so the comparator (not
        // the input order) determines the result, and argsort's permutation must
        // reproduce the sorted values when applied through basic indexing.
        let raw = vec![3_i32, 1, 2, 0, 7, 5, 4, 6, 11, 9, 8, 10];
        let a = rt::asarray((raw, [3, 4], &device));
        let by_mod4 = |x: &i32, y: &i32| (x % 4).cmp(&(y % 4));
        let sorted = a.sort_custom(1, by_mod4);
        let idx = a.argsort_custom(1, by_mod4);
        for i in 0..3 {
            let row: Vec<i32> = (0..4).map(|j| sorted.i((i, j)).to_scalar()).collect();
            assert_eq!(row.iter().map(|x| x % 4).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
            for j in 0..4 {
                let vi = sorted.i((i, j)).to_scalar();
                let orig = a.i((i, idx.i((i, j)).to_scalar())).to_scalar();
                assert_eq!(vi, orig);
            }
        }
    }
}
