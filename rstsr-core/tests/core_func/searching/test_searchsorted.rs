#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_searchsorted {
    use super::*;
    static FUNC: &str = "numpy_searchsorted";

    #[test]
    fn test_searchsorted_basic() {
        // NumPy v2.5.2, _core/tests/test_numeric.py, TestNumeric::test_searchsorted
        // (line 276): arr = [-8, -5, -1, 3, 6, 10]; searchsorted(arr, 0) == 3
        crate::specify_test!("test_searchsorted_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let arr = rt::tensor_from_nested!([-8, -5, -1, 3, 6, 10], &device);
        let v = rt::tensor_from_nested!([0], &device);
        assert_eq!(rt::searchsorted((&arr, &v, ())).to_vec(), vec![3]);
    }

    #[test]
    fn test_searchsorted_floats_nan() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py,
        // TestMethods::test_searchsorted_floats (line 3072):
        // a = [0, 1, nan]; a.searchsorted(a, 'left') == [0, 1, 2]
        // a.searchsorted(a, 'right') == [1, 2, 3]
        crate::specify_test!("test_searchsorted_floats_nan");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([0.0_f64, 1.0, f64::NAN], &device);
        let left = rt::searchsorted((&a, &a, "left")).to_vec();
        assert_eq!(left, vec![0, 1, 2]);
        let right = rt::searchsorted((&a, &a, "right")).to_vec();
        assert_eq!(right, vec![1, 2, 3]);
    }

    #[test]
    fn test_searchsorted_with_sorter() {
        // NumPy v2.5.2, _core/tests/test_multiarray.py,
        // TestMethods::test_searchsorted_with_sorter (line 3225), case 2:
        // a = [0, 1, 2, 3, 5] * 20; side='left' + sorter -> [0, 20, 40, 60, 80];
        // side='right' + sorter -> [20, 40, 60, 80, 100]
        crate::specify_test!("test_searchsorted_with_sorter");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = [0, 1, 2, 3, 5] repeated 20 times
        let block = vec![0_usize, 1, 2, 3, 5];
        let mut a = Vec::with_capacity(100);
        for _ in 0..20 {
            a.extend_from_slice(&block);
        }
        let a = rt::asarray((a, &device));
        // stable argsort gives the block-wise permutation
        let s_vec = a.argsort(()).to_vec();
        let k = rt::tensor_from_nested!([0_usize, 1, 2, 3, 5], &device);
        let expected_left = vec![0_usize, 20, 40, 60, 80];
        let expected_right = vec![20_usize, 40, 60, 80, 100];
        assert_eq!(rt::searchsorted((&a, &k, (SearchSide::Left, s_vec.clone()))).to_vec(), expected_left);
        assert_eq!(rt::searchsorted((&a, &k, (SearchSide::Right, s_vec))).to_vec(), expected_right);
    }

    #[test]
    fn test_searchsorted_return_type() {
        // NumPy v2.5.2, TestMethods::test_searchsorted_return_type (line 3304):
        // the result dtype is the default index dtype (usize here)
        crate::specify_test!("test_searchsorted_return_type");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 3, 5], &device);
        let v = rt::tensor_from_nested!([4], &device);
        let out = rt::searchsorted((&a, &v, ()));
        assert_eq!(out.shape(), &[1]);
        assert_eq!(out.to_vec(), vec![2]);
    }
}

#[cfg(test)]
mod custom_searchsorted {
    use super::*;
    static FUNC: &str = "custom_searchsorted";

    #[test]
    fn test_side_left_right_boundaries() {
        // side left/right differ exactly at exact matches
        crate::specify_test!("test_side_left_right_boundaries");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([10, 20, 30, 40], &device);
        let k = rt::tensor_from_nested!([10, 15, 20, 25, 30, 45], &device);
        assert_eq!(rt::searchsorted((&a, &k, "left")).to_vec(), vec![0, 1, 1, 2, 2, 4]);
        assert_eq!(rt::searchsorted((&a, &k, "right")).to_vec(), vec![1, 1, 2, 2, 3, 4]);
    }

    #[test]
    fn test_multidim_v() {
        // output shape equals the values' shape
        crate::specify_test!("test_multidim_v");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 3, 5, 7], &device);
        let k = rt::tensor_from_nested!([[0, 2], [4, 9]], &device);
        let out = rt::searchsorted((&a, &k, ()));
        assert_eq!(out.shape(), &[2, 2]);
        let expected = rt::tensor_from_nested!([[0, 1], [2, 4]], &device);
        assert_equal(out, &expected, None);
    }

    #[test]
    fn test_ndim1_required() {
        // x1 must be one-dimensional
        crate::specify_test!("test_ndim1_required");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let v = rt::tensor_from_nested!([1], &device);
        assert!(rt::searchsorted_f(&a, &v, ()).is_err());
    }

    #[test]
    fn test_invalid_side() {
        crate::specify_test!("test_invalid_side");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 3, 5], &device);
        let v = rt::tensor_from_nested!([4], &device);
        let args: std::result::Result<SearchSortedArgs, _> = "middle".try_into();
        assert!(args.is_err());
        let _ = (&a, &v);
    }

    #[test]
    fn test_empty_inputs() {
        // NumPy v2.5.2, TestMethods::test_searchsorted_n_elements (line 3112):
        // 0-element x1 returns all zeros; empty x2 returns an empty output
        crate::specify_test!("test_empty_inputs");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let empty: Tensor<i32, _> = rt::zeros(([0], &device));
        let v = rt::tensor_from_nested!([1, 2, 3], &device);
        assert_eq!(rt::searchsorted((&empty, &v, ())).to_vec(), vec![0, 0, 0]);

        let a = rt::tensor_from_nested!([1, 3, 5], &device);
        let out = rt::searchsorted((&a, &empty, ()));
        assert_eq!(out.shape(), &[0]);
    }

    #[test]
    fn test_invalid_sorter() {
        // NumPy v2.5.2, TestMethods::test_searchsorted_with_invalid_sorter
        // (line 3211): wrong-length or out-of-range sorter raises
        crate::specify_test!("test_invalid_sorter");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 3, 5], &device);
        let v = rt::tensor_from_nested!([4], &device);
        // wrong length
        assert!(rt::searchsorted_f(&a, &v, (SearchSide::Left, vec![0, 1])).is_err());
        // out-of-range entry
        assert!(rt::searchsorted_f(&a, &v, (SearchSide::Left, vec![0, 1, 5])).is_err());
    }

    #[test]
    fn test_invalid_side_through_f() {
        // invalid side surfaces as Err through searchsorted_f's TryInto
        crate::specify_test!("test_invalid_side_through_f");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 3, 5], &device);
        let v = rt::tensor_from_nested!([4], &device);
        assert!(rt::searchsorted_f(&a, &v, "middle").is_err());
    }

    #[test]
    fn test_strided_x1() {
        // a strided (sliced) x1 works: values are read through the layout
        crate::specify_test!("test_strided_x1");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let base = rt::arange((12, &device)); // 0..11
        let a = base.i((slice!(0, None, 3),)); // [0, 3, 6, 9]
        let k = rt::tensor_from_nested!([2, 3, 4, 7, 9, 10], &device);
        let out = rt::searchsorted((&a, &k, ()));
        // sorted x1 = [0, 3, 6, 9]: 2->1, 3->1, 4->2, 7->3, 9->3, 10->4
        assert_eq!(out.to_vec(), vec![1, 1, 2, 3, 3, 4]);
    }
}
