//! diff tests: NumPy-cited (TestDiff) + custom edges.
//!
//! `bool` input is N/A: `rt::diff` requires `T: Sub<Output = T>`, and Rust's
//! `bool` has no `Sub` (NumPy diffs bool arrays as XOR) — see
//! `tracking/numpy_differences.md`.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_diff {
    use super::*;
    static FUNC: &str = "numpy_diff";

    #[test]
    fn test_basic() {
        // numpy: v2.5.2 | lib/tests/test_function_base.py::TestDiff::test_basic (L849)
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = [1, 4, 6, 7, 12]; out = [3, 2, 1, 5]; n=2 -> [-1, -1, 4]; n=3 -> [0, 5]
        let x = rt::tensor_from_nested!([1, 4, 6, 7, 12], &device);
        assert_equal(rt::diff((&x, 0, 1, None, None)), rt::tensor_from_nested!([3, 2, 1, 5], &device), None);
        assert_equal(rt::diff((&x, 0, 2, None, None)), rt::tensor_from_nested!([-1, -1, 4], &device), None);
        assert_equal(rt::diff((&x, 0, 3, None, None)), rt::tensor_from_nested!([0, 5], &device), None);

        // x = [1.1, 2.2, 3.0, -0.2, -0.1]; out = [1.1, 0.8, -3.2, 0.1]
        let xf = rt::tensor_from_nested!([1.1_f64, 2.2, 3.0, -0.2, -0.1], &device);
        let out = rt::tensor_from_nested!([1.1_f64, 0.8, -3.2, 0.1], &device);
        assert_equal(rt::diff((&xf, 0, 1, None, None)), &out, None);
    }

    #[test]
    fn test_axis() {
        // numpy: v2.5.2 | lib/tests/test_function_base.py::TestDiff::test_axis (L868)
        crate::specify_test!("test_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::tensor_from_nested!(
            [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]],
            &device
        );

        // diff(x) and diff(x, axis=-1) -> shape (2, 3, 3)
        let exp_last = rt::tensor_from_nested!(
            [[[1, -1, 1], [-1, 1, -1], [1, -1, 1]], [[-1, 1, -1], [1, -1, 1], [-1, 1, -1]]],
            &device
        );
        assert_equal(rt::diff((&x, -1, 1, None, None)), &exp_last, None);
        // diff(x) defaults to the last axis, same as axis=-1
        assert_equal(rt::diff((&x, -1, 1, None, None)), rt::diff((&x, 2, 1, None, None)), None);

        // axis=1 and axis=-2 agree
        assert_equal(rt::diff((&x, 1, 1, None, None)), rt::diff((&x, -2, 1, None, None)), None);
        // axis=0 -> shape (1, 3, 4)
        assert_eq!(rt::diff((&x, 0, 1, None, None)).shape(), &[1, 3, 4]);

        // out-of-range axes raise
        assert!((&x, 3, 1, None, None).diff_f().is_err());
        assert!((&x, -4, 1, None, None).diff_f().is_err());

        // 0-d input raises (np.diff(np.array(1.1)) -> ValueError)
        let x0 = rt::full(([], 1.1_f64, &device));
        assert!((&x0, 0, 1, None, None).diff_f().is_err());
    }

    #[test]
    fn test_nd() {
        // numpy: v2.5.2 | lib/tests/test_function_base.py::TestDiff::test_nd (L884)
        // n-th order along an axis composes: diff(x, n=2) == diff(diff(x)).
        crate::specify_test!("test_nd");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = (rt::arange((120, &device)) * 7).mapv(|v| v % 11).into_shape([4, 5, 6]);
        let d1 = rt::diff((&x, -1, 1, None, None));
        let d2 = rt::diff((&x, -1, 2, None, None));
        assert_equal(rt::diff((&d1, -1, 1, None, None)), &d2, None);

        let e1 = rt::diff((&x, 0, 1, None, None));
        let e2 = rt::diff((&x, 0, 2, None, None));
        assert_equal(rt::diff((&e1, 0, 1, None, None)), &e2, None);
    }

    #[test]
    fn test_n() {
        // numpy: v2.5.2 | lib/tests/test_function_base.py::TestDiff::test_n (L895)
        // x = [0, 1, 2]; output length is max(0, len(x) - n); n=0 is a copy.
        crate::specify_test!("test_n");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::tensor_from_nested!([0, 1, 2], &device);
        assert_equal(rt::diff((&x, 0, 0, None, None)), &x, None);
        assert_equal(rt::diff((&x, 0, 1, None, None)), rt::tensor_from_nested!([1, 1], &device), None);
        assert_equal(rt::diff((&x, 0, 2, None, None)), rt::tensor_from_nested!([0], &device), None);
        assert_eq!(rt::diff((&x, 0, 3, None, None)).shape(), &[0]);
        assert_eq!(rt::diff((&x, 0, 4, None, None)).shape(), &[0]);
    }

    #[test]
    fn test_prepend() {
        // numpy: v2.5.2 | lib/tests/test_function_base.py::TestDiff::test_prepend (L934)
        // NumPy expands a scalar prepend to length 1 along `axis`; rstsr requires
        // an explicit tensor matching x's shape outside `axis`.
        crate::specify_test!("test_prepend");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = arange(5) + 1; diff(x, prepend=0) == ones(5)
        let x = rt::arange((1, 6, &device));
        let zero1 = rt::full(([1], 0, &device));
        let ones: Tensor<i32, _> = rt::ones(([5], &device));
        assert_equal(rt::diff((&x, 0, 1, Some(&zero1), None)), &ones, None);
        // prepend=[-1, 0] -> length 6, still ones
        let two = rt::tensor_from_nested!([-1, 0], &device);
        let ones6: Tensor<i32, _> = rt::ones(([6], &device));
        assert_equal(rt::diff((&x, 0, 1, Some(&two), None)), &ones6, None);

        // 2-d: np.diff(arange(4).reshape(2,2), axis=1, prepend=0) -> [[0, 1], [2, 1]]
        let m = rt::arange((4, &device)).into_shape([2, 2]);
        let p1 = rt::full(([2, 1], 0, &device));
        let exp = rt::tensor_from_nested!([[0, 1], [2, 1]], &device);
        assert_equal(rt::diff((&m, 1, 1, Some(&p1), None)), &exp, None);
        // axis=0, prepend shape (1, 2) -> [[0, 1], [2, 2]]
        let p0 = rt::full(([1, 2], 0, &device));
        let exp0 = rt::tensor_from_nested!([[0, 1], [2, 2]], &device);
        assert_equal(rt::diff((&m, 0, 1, Some(&p0), None)), &exp0, None);

        // shape mismatch outside axis raises
        let bad: Tensor<i32, _> = rt::zeros(([3, 3], &device));
        assert!((&m, 1, 1, Some(&bad), None).diff_f().is_err());
    }

    #[test]
    fn test_append() {
        // numpy: v2.5.2 | lib/tests/test_function_base.py::TestDiff::test_append (L958)
        crate::specify_test!("test_append");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = arange(5); diff(x, append=0) == [1, 1, 1, 1, -4]
        let x = rt::arange((5, &device));
        let zero1 = rt::full(([1], 0, &device));
        assert_equal(
            rt::diff((&x, 0, 1, None, Some(&zero1))),
            rt::tensor_from_nested!([1, 1, 1, 1, -4], &device),
            None,
        );
        // append=[0, 2] -> [1, 1, 1, 1, -4, 2]
        let two = rt::tensor_from_nested!([0, 2], &device);
        assert_equal(
            rt::diff((&x, 0, 1, None, Some(&two))),
            rt::tensor_from_nested!([1, 1, 1, 1, -4, 2], &device),
            None,
        );
    }
}

#[cfg(test)]
mod custom_diff {
    use super::*;
    static FUNC: &str = "custom_diff";

    #[test]
    fn test_empty_axis_stops() {
        // an n pass with n > axis length stops at the empty axis (no panic)
        crate::specify_test!("test_empty_axis_stops");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::tensor_from_nested!([1, 4, 6], &device);
        assert_eq!(rt::diff((&x, 0, 5, None, None)).shape(), &[0]);
    }

    #[test]
    fn test_negative_axis() {
        crate::specify_test!("test_negative_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::arange((6, &device)).into_shape([2, 3]);
        assert_equal(rt::diff((&x, -1, 1, None, None)), rt::diff((&x, 1, 1, None, None)), None);
    }
}
