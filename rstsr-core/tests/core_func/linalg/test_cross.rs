#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_cross {
    use super::*;
    static FUNC: &str = "numpy_cross";

    #[test]
    fn test_3x3() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestCross::test_3x3 (L3978)
        crate::specify_test!("test_3x3");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // u = [1, 2, 3]
        // v = [4, 5, 6]
        // z = np.array([-3, 6, -3])
        // cp = np.cross(u, v); assert_equal(cp, z)
        // cp = np.cross(v, u); assert_equal(cp, -z)
        let u = rt::tensor_from_nested!([1, 2, 3], &device);
        let v = rt::tensor_from_nested!([4, 5, 6], &device);
        let z = rt::tensor_from_nested!([-3, 6, -3], &device);
        let neg_z = rt::tensor_from_nested!([3, -6, 3], &device);
        assert_equal(rt::cross(&u, &v, -1), &z, None);
        assert_equal(rt::cross(&v, &u, -1), &neg_z, None);
    }

    #[test]
    fn test_broadcasting_shapes() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestCross::test_broadcasting_shapes (L3995)
        crate::specify_test!("test_broadcasting_shapes");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // u = np.ones((2, 1, 3)); v = np.ones((5, 3))
        // assert_equal(np.cross(u, v).shape, (2, 5, 3))
        // (both axes are the default last one here, so one shared `axis` covers it)
        let u: Tensor<f64, _> = rt::ones(([2, 1, 3], &device));
        let v: Tensor<f64, _> = rt::ones(([5, 3], &device));
        assert_eq!(rt::cross(&u, &v, -1).shape(), &[2, 5, 3]);

        // the same shapes raise when the shared axis is not length 3 in both inputs
        let u_bad: Tensor<f64, _> = rt::ones(([2, 1, 2], &device));
        assert!(rt::cross_f(&u_bad, &v, -1).is_err());
    }
}

#[cfg(test)]
mod custom_cross {
    use super::*;
    static FUNC: &str = "custom_cross";

    #[test]
    fn test_cross_explicit_axis() {
        // per-row cross along axis 1
        crate::specify_test!("test_cross_explicit_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let b = rt::tensor_from_nested!([[1, 0, 0], [0, 1, 0]], &device);
        let expected = rt::tensor_from_nested!([[0, 3, -2], [-6, 0, 4]], &device);
        assert_equal(rt::cross(&a, &b, 1), &expected, None);
        // the default axis (last) and `()` agree with the explicit -1
        assert_equal(rt::cross(&a, &b, -1), rt::cross(&a, &b, ()), None);
    }

    #[test]
    fn test_cross_stacked() {
        // the cross axis may be non-trailing
        crate::specify_test!("test_cross_stacked");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a (2, 3): axis 0 (length 2) is not length 3, so it must raise
        let bad: Tensor<f64, _> = rt::ones(([2, 3], &device));
        assert!(rt::cross_f(&bad, &bad, 0).is_err());

        // (3, 2) with axis 0: the cross of each column pair, result keeping the axis
        let a = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
        let b = rt::tensor_from_nested!([[1, 0], [0, 1], [0, 0]], &device);
        let expected = rt::tensor_from_nested!([[0, -6], [5, 0], [-3, 2]], &device);
        assert_equal(rt::cross(&a, &b, 0), &expected, None);
    }

    #[test]
    fn test_cross_non_contiguous() {
        crate::specify_test!("test_cross_non_contiguous");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a non-contiguous operand (transposed view) against a broadcast operand
        let m = rt::arange((9, &device)).into_shape([3, 3]);
        let mt = rt::matrix_transpose(&m);
        let b = rt::tensor_from_nested!([1, 0, 0], &device);
        let actual = rt::cross(&mt, &b, -1);
        let expected = rt::tensor_from_nested!([[0, 6, -3], [0, 7, -4], [0, 8, -5]], &device);
        assert_equal(&actual, &expected, None);
    }

    #[test]
    fn test_cross_negative_stride() {
        crate::specify_test!("test_cross_negative_stride");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a reversed (negative-stride) cross axis: a[:, ::-1] = [[3, 2, 1], [6, 5, 4]]
        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let a_rev = a.i((.., slice!(None, None, -1)));
        let b = rt::tensor_from_nested!([1, 0, 0], &device);
        let actual = rt::cross(&a_rev, &b, -1);
        let expected = rt::tensor_from_nested!([[0, 1, -2], [0, 4, -5]], &device);
        assert_equal(&actual, &expected, None);
    }

    #[test]
    fn test_cross_col_major() {
        // the values are order-independent; only the fresh result's arrangement differs
        crate::specify_test!("test_cross_col_major");

        let u = [1, 2, 3];
        let v = [4, 5, 6];

        let mut device_r = TESTCFG.device.clone();
        device_r.set_default_order(RowMajor);
        let ur = rt::tensor_from_nested!([1, 2, 3], &device_r);
        let vr = rt::tensor_from_nested!([4, 5, 6], &device_r);
        let cr = rt::cross(&ur, &vr, -1);

        let mut device_c = TESTCFG.device.clone();
        device_c.set_default_order(ColMajor);
        let uc = rt::asarray((&u[..], &device_c));
        let vc = rt::asarray((&v[..], &device_c));
        let cc = rt::cross(&uc, &vc, -1);

        assert_eq!(cr.to_vec(), cc.to_vec());
        assert_eq!(cc.to_vec(), vec![-3, 6, -3]);
    }
}
