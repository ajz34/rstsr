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
    fn test_outer() {
        // numpy: v2.5.2 | linalg/tests/test_linalg.py::TestOuter (L1960)
        crate::specify_test!("test_outer");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // arr1 = np.arange(3)
        // arr2 = np.arange(3)
        // expected = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4]])
        // assert_array_equal(np.linalg.outer(arr1, arr2), expected)
        let arr1 = rt::arange((3, &device));
        let arr2 = rt::arange((3, &device));
        let actual = rt::outer(&arr1, &arr2);
        let expected = rt::tensor_from_nested!([[0, 0, 0], [0, 1, 2], [0, 2, 4]], &device);
        assert_equal(&actual, &expected, None);
        assert_eq!(actual.shape(), &[3, 3]);

        // with assert_raises_regex(ValueError, "Input arrays must be one-dimensional"):
        //     np.linalg.outer(arr1[:, np.newaxis], arr2)
        let arr1_col = arr1.into_shape([3, 1]);
        assert!(rt::outer_f(&arr1_col, &arr2).is_err());
    }
}

#[cfg(test)]
mod custom_outer {
    use super::*;
    static FUNC: &str = "custom_outer";

    #[test]
    fn test_outer_matches_broadcast_mul() {
        // the array-API definition: outer(x1, x2) == x1[:, None] * x2[None, :]
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
}
