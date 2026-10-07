#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_repeat {
    use super::*;
    static FUNC: &str = "numpy_repeat";

    #[test]
    fn test_basic() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestRepeat::test_basic (L8440)
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // m = np.array([1, 2, 3, 4, 5, 6])
        // A = np.repeat(m, [1, 3, 2, 1, 1, 2])
        // assert_equal(A, [1, 2, 2, 2, 3, 3, 4, 5, 6, 6])
        let m = rt::tensor_from_nested!([1, 2, 3, 4, 5, 6], &device);
        let expected = rt::tensor_from_nested!([1, 2, 2, 2, 3, 3, 4, 5, 6, 6], &device);
        assert_equal(m.repeat([1, 3, 2, 1, 1, 2], None), &expected, None);
    }

    #[test]
    fn test_broadcast1() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestRepeat::test_broadcast1 (L8446)
        crate::specify_test!("test_broadcast1");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // m = np.array([1, 2, 3, 4, 5, 6])
        // A = np.repeat(m, 2)
        // assert_equal(A, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        let m = rt::tensor_from_nested!([1, 2, 3, 4, 5, 6], &device);
        let expected = rt::tensor_from_nested!([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], &device);
        assert_equal(m.repeat(2, None), &expected, None);
    }

    #[test]
    fn test_axis_spec() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestRepeat::test_axis_spec (L8452)
        crate::specify_test!("test_axis_spec");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // m_rect = np.array([[1, 2, 3], [4, 5, 6]])
        // A = np.repeat(m_rect, [2, 1], axis=0)
        let m_rect = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let expected = rt::tensor_from_nested!([[1, 2, 3], [1, 2, 3], [4, 5, 6]], &device);
        assert_equal(m_rect.repeat([2, 1], 0), &expected, None);

        // A = np.repeat(m_rect, [1, 3, 2], axis=1)
        let expected = rt::tensor_from_nested!([[1, 2, 2, 2, 3, 3], [4, 5, 5, 5, 6, 6]], &device);
        assert_equal(m_rect.repeat([1, 3, 2], 1), &expected, None);
    }

    #[test]
    fn test_broadcast2() {
        // numpy: v2.5.2 | _core/tests/test_multiarray.py::TestRepeat::test_broadcast2 (L8463)
        crate::specify_test!("test_broadcast2");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // m_rect = np.array([[1, 2, 3], [4, 5, 6]])
        // A = np.repeat(m_rect, 2, axis=0)
        let m_rect = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let expected = rt::tensor_from_nested!([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]], &device);
        assert_equal(m_rect.repeat(2, 0), &expected, None);

        // A = np.repeat(m_rect, 2, axis=1)
        let expected = rt::tensor_from_nested!([[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]], &device);
        assert_equal(m_rect.repeat(2, 1), &expected, None);
    }
}

#[cfg(test)]
mod custom_repeat {
    use super::*;
    static FUNC: &str = "custom_repeat";

    #[test]
    fn test_negative_axis() {
        crate::specify_test!("test_negative_axis");

        // negative axis normalization: axis=-1 == axis=1 on a 2-d input
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let m_rect = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let expected = m_rect.repeat(2, 1);
        assert_equal(m_rect.repeat(2, -1), &expected, None);
    }

    #[test]
    fn test_flatten_c_order_strided() {
        crate::specify_test!("test_flatten_c_order_strided");

        // C-order flatten contract: transposed input must flatten row-major
        // (np.repeat(np.arange(6).reshape(2, 3).T, 1) == [0, 3, 1, 4, 2, 5])
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t(); // strided view
        let expected = rt::tensor_from_nested!([0, 3, 1, 4, 2, 5], &device);
        assert_equal(at.repeat(1, None), &expected, None);
    }

    #[test]
    fn test_single_count_list() {
        crate::specify_test!("test_single_count_list");

        // a length-1 repeats list is the scalar form
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((3, &device));
        let expected = rt::tensor_from_nested!([0, 0, 1, 1, 2, 2], &device);
        assert_equal(a.repeat([2], None), &expected, None);
    }

    #[test]
    fn test_zero_count() {
        crate::specify_test!("test_zero_count");

        // repeats of zero drop elements / empty the axis
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let expected = rt::tensor_from_nested!([2, 3], &device);
        assert_equal(a.repeat([0, 1, 1], None), &expected, None);

        let empty = a.repeat(0, None);
        assert_eq!(empty.shape(), &[0]);
    }

    #[test]
    fn test_length_mismatch_error() {
        crate::specify_test!("test_length_mismatch_error");

        // repeats length must be 1 or the axis size
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        assert!(a.repeat_f([1, 2, 3, 4], 0).is_err());
        assert!(a.repeat_f([1, 2], None).is_err());
    }

    #[test]
    fn test_multiple_axes_error() {
        crate::specify_test!("test_multiple_axes_error");

        // at most one axis is accepted
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        assert!(a.repeat_f(1, [0, 1]).is_err());
    }

    #[test]
    fn test_0d() {
        crate::specify_test!("test_0d");

        // 0-d input flattens to a 1-element sequence
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::full(([], 7, &device));
        let expected = rt::tensor_from_nested!([7, 7, 7], &device);
        assert_equal(a.repeat(3, None), &expected, None);
    }

    #[test]
    fn test_empty() {
        crate::specify_test!("test_empty");

        // empty input repeats to empty output
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<i32, _> = rt::zeros(([0], &device));
        let out = a.repeat(3, None);
        assert_eq!(out.shape(), &[0]);
    }
}
