#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_roll {
    use super::*;
    static FUNC: &str = "numpy_roll";

    #[test]
    fn test_roll1d() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestRoll::test_roll1d (L3761)
        crate::specify_test!("test_roll1d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.arange(10); xr = np.roll(x, 2)
        // assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))
        let x = rt::arange((10, &device));
        let expected = rt::tensor_from_nested!([8, 9, 0, 1, 2, 3, 4, 5, 6, 7], &device);
        assert_equal(x.roll(2, None), &expected, None);
    }

    #[test]
    fn test_roll2d() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestRoll::test_roll2d (L3766)
        crate::specify_test!("test_roll2d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x2 = np.reshape(np.arange(10), (2, 5))
        let x2 = rt::arange((10, &device)).into_shape([2, 5]);

        // x2r = np.roll(x2, 1)
        // assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))
        let expected = rt::tensor_from_nested!([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]], &device);
        assert_equal(x2.roll(1, None), &expected, None);

        // x2r = np.roll(x2, 1, axis=0)
        let expected = rt::tensor_from_nested!([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], &device);
        assert_equal(x2.roll(1, 0), &expected, None);

        // x2r = np.roll(x2, 1, axis=1)
        let expected = rt::tensor_from_nested!([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]], &device);
        assert_equal(x2.roll(1, 1), &expected, None);

        // Roll multiple axes at once.
        // x2r = np.roll(x2, 1, axis=(0, 1))
        let expected = rt::tensor_from_nested!([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]], &device);
        assert_equal(x2.roll(1, [0, 1]), &expected, None);

        // x2r = np.roll(x2, (1, 0), axis=(0, 1))
        let expected = rt::tensor_from_nested!([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], &device);
        assert_equal(x2.roll([1, 0], [0, 1]), &expected, None);

        // x2r = np.roll(x2, (-1, 0), axis=(0, 1))
        let expected = rt::tensor_from_nested!([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], &device);
        assert_equal(x2.roll([-1, 0], [0, 1]), &expected, None);

        // x2r = np.roll(x2, (0, 1), axis=(0, 1))
        let expected = rt::tensor_from_nested!([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]], &device);
        assert_equal(x2.roll([0, 1], [0, 1]), &expected, None);

        // x2r = np.roll(x2, (0, -1), axis=(0, 1))
        let expected = rt::tensor_from_nested!([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]], &device);
        assert_equal(x2.roll([0, -1], [0, 1]), &expected, None);

        // x2r = np.roll(x2, (1, 1), axis=(0, 1))
        let expected = rt::tensor_from_nested!([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]], &device);
        assert_equal(x2.roll([1, 1], [0, 1]), &expected, None);

        // x2r = np.roll(x2, (-1, -1), axis=(0, 1))
        let expected = rt::tensor_from_nested!([[6, 7, 8, 9, 5], [1, 2, 3, 4, 0]], &device);
        assert_equal(x2.roll([-1, -1], [0, 1]), &expected, None);

        // Roll the same axis multiple times.
        // x2r = np.roll(x2, 1, axis=(0, 0))
        let expected = rt::tensor_from_nested!([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], &device);
        assert_equal(x2.roll(1, [0, 0]), &expected, None);

        // x2r = np.roll(x2, 1, axis=(1, 1))
        let expected = rt::tensor_from_nested!([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]], &device);
        assert_equal(x2.roll(1, [1, 1]), &expected, None);

        // Roll more than one turn in either direction.
        // x2r = np.roll(x2, 6, axis=1)
        let expected = rt::tensor_from_nested!([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]], &device);
        assert_equal(x2.roll(6, 1), &expected, None);

        // x2r = np.roll(x2, -4, axis=1)
        let expected = rt::tensor_from_nested!([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]], &device);
        assert_equal(x2.roll(-4, 1), &expected, None);
    }

    #[test]
    fn test_roll_empty() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestRoll::test_roll_empty (L3813)
        crate::specify_test!("test_roll_empty");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.array([]); assert_equal(np.roll(x, 1), np.array([]))
        let x: Tensor<i32, _> = rt::zeros(([0], &device));
        let out = x.roll(1, None);
        assert_eq!(out.shape(), &[0]);
    }

    #[test]
    fn test_roll_big_int() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestRoll::test_roll_big_int (L3825)
        crate::specify_test!("test_roll_big_int");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // x = np.arange(4)
        // assert_equal(np.roll(x, 2**100), x)
        // isize cannot hold 2**100; the same multi-turn wrap path is exercised
        // with shifts that are exact multiples of the axis length (identity)
        let x = rt::arange((4, &device));
        assert_equal(x.roll(4 * 25, None), &x, None);
        // negative multiple turns is identity
        assert_equal(x.roll(-(4 * 3), None), &x, None);
    }
}

#[cfg(test)]
mod custom_roll {
    use super::*;
    static FUNC: &str = "custom_roll";

    #[test]
    fn test_flatten_c_order_strided() {
        crate::specify_test!("test_flatten_c_order_strided");

        // axis=None flatten uses strict row-major order even for strided input;
        // NumPy then restores the input shape:
        // flat(at) = [0, 3, 1, 4, 2, 5]; roll by 1 -> [5, 0, 3, 1, 4, 2]
        // reshaped back to (3, 2) -> [[5, 0], [3, 1], [4, 2]]
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t();
        let expected = rt::tensor_from_nested!([[5, 0], [3, 1], [4, 2]], &device);
        assert_equal(at.roll(1, None), &expected, None);
    }

    #[test]
    fn test_shape_preserved_strided() {
        crate::specify_test!("test_shape_preserved_strided");

        // rolling a transposed view keeps the (transposed) shape
        // at = [[0,3],[1,4],[2,5]]; roll axis 0 by 1 (NumPy: out[i] = in[i-1]):
        // [[2,5],[0,3],[1,4]]
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t();
        let out = at.roll(1, 0);
        assert_eq!(out.shape(), &[3, 2]);
        let expected = rt::tensor_from_nested!([[2, 5], [0, 3], [1, 4]], &device);
        assert_equal(out, &expected, None);
    }

    #[test]
    fn test_shift_length_mismatch() {
        crate::specify_test!("test_shift_length_mismatch");

        // tuple shift and tuple axis must have the same length
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        assert!(a.roll_f([1, 2], [0, 1, 1]).is_err());
    }

    #[test]
    fn test_single_shift_tuple_axis() {
        crate::specify_test!("test_single_shift_tuple_axis");

        // scalar shift + tuple axis: same shift applied to each axis (NumPy)
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.roll(x2, 1, axis=(0, 1)) == np.roll(x2, (1, 1), axis=(0, 1))
        let x2 = rt::arange((10, &device)).into_shape([2, 5]);
        let expected = x2.roll([1, 1], [0, 1]);
        assert_equal(x2.roll(1, [0, 1]), &expected, None);
    }

    #[test]
    fn test_tuple_shift_single_axis_broadcast() {
        crate::specify_test!("test_tuple_shift_single_axis_broadcast");

        // tuple shift on a single axis is summed (NumPy broadcasts the axis);
        // np.roll(x, (1, 2), axis=0) on a size-5 axis == np.roll(x, 3, axis=0)
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::arange((5, &device));
        let expected = x.roll(3, None);
        assert_equal(x.roll([1, 2], None), &expected, None);
        let m = rt::arange((20, &device)).into_shape([4, 5]);
        let expected_m = m.roll(3, 1);
        assert_equal(m.roll([1, 2], 1), &expected_m, None);
    }

    #[test]
    fn test_len1_shift_tuple_axis_broadcast() {
        crate::specify_test!("test_len1_shift_tuple_axis_broadcast");

        // len-1 tuple shift with a tuple axis: that shift on every axis (NumPy)
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::arange((24, &device)).into_shape([2, 3, 4]);
        let expected = x.roll([5, 5], [0, 1]);
        assert_equal(x.roll([5], [0, 1]), &expected, None);
        // int shift with tuple axis behaves identically
        assert_equal(x.roll(5, [0, 1]), &expected, None);
    }

    #[test]
    fn test_0d() {
        crate::specify_test!("test_0d");

        // 0-d tensor rolls to itself (flattened form has one element)
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::full(([], 9, &device));
        let out = a.roll(3, None);
        assert_eq!(out.shape(), &[]);
        assert_equal(out, &a, None);
    }

    #[test]
    fn test_zero_shift_is_copy() {
        crate::specify_test!("test_zero_shift_is_copy");

        // shift 0 still returns a fresh owned tensor (independent data)
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((4, &device));
        let mut b = a.roll(0, None);
        b.i_mut((0,)).fill(100);
        let expected = rt::tensor_from_nested!([0, 1, 2, 3], &device);
        assert_equal(a, &expected, None);
    }
}

#[cfg(test)]
mod device_order {
    use super::*;
    static FUNC: &str = "device_order";

    #[test]
    fn test_flatten_col_major() {
        crate::specify_test!("test_flatten_col_major");

        // the flattened form (`axis = None`) visits elements in the device
        // default order: col-major sequence [0 1 2 3 4 5] rolled by 1 becomes
        // [5 0 1 2 3 4], restored to [2, 3] in col-major order
        let mut device = TESTCFG.device.clone();
        device.set_default_order(ColMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]); // [[0 2 4], [1 3 5]]
        let out = rt::roll((&a, 1, None));
        // the restored shape follows the device order as well (F-contiguous)
        assert_eq!(out.stride(), &[1, 2]);
        let expected = rt::tensor_from_nested!([[5, 1, 3], [0, 2, 4]], &device);
        assert_equal(out, &expected, None);
    }
}
