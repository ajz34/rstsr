#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_outer {
    use super::*;
    static FUNC: &str = "numpy_outer";

    #[test]
    fn test_outer_basic() {
        // NumPy v2.5.2, lib/_shape_base_impl.py::outer
        //   np.outer(np.array([1, 2, 3]), np.array([4, 5]))
        //   -> [[4, 5], [8, 10], [12, 15]]
        crate::specify_test!("test_outer_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5], &device);
        let actual = rt::outer(&a, &b);
        let expected = rt::tensor_from_nested!([[4, 5], [8, 10], [12, 15]], &device);
        assert_equal(&actual, &expected, None);
        assert_eq!(actual.shape(), &[3, 2]);
    }

    #[test]
    fn test_outer_matches_broadcast_mul() {
        // the array-API contract: outer(x1, x2) == x1[:, None] * x2[None, :]
        crate::specify_test!("test_outer_matches_broadcast_mul");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device));
        let b = rt::arange((4, &device));
        let actual = rt::outer(&a, &b);
        let expected = &a.into_shape([6, 1]) * &b.into_shape([1, 4]);
        assert_equal(&actual, &expected, None);
        assert_eq!(actual.shape(), &[6, 4]);
    }

    #[test]
    fn test_outer_non_contiguous() {
        crate::specify_test!("test_outer_non_contiguous");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // strided first operand: column 0 of a (4, 3) matrix
        let m = rt::arange((12, &device)).into_shape([4, 3]);
        let a = m.i((.., 0));
        let b = rt::tensor_from_nested!([1, 2], &device);
        let actual = rt::outer(&a, &b);

        let expected = rt::tensor_from_nested!([[0, 0], [3, 6], [6, 12], [9, 18]], &device);
        assert_equal(&actual, &expected, None);
    }

    #[test]
    fn test_outer_requires_1d() {
        crate::specify_test!("test_outer_requires_1d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2], &device);
        let b2 = rt::tensor_from_nested!([[1, 2]], &device);
        assert!(rt::outer_f(&a, &b2).is_err());
        assert!(rt::outer_f(&b2, &a).is_err());
    }
}
