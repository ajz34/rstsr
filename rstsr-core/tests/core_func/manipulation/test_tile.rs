#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_tile {
    use super::*;
    static FUNC: &str = "numpy_tile";

    #[test]
    fn test_basic() {
        // numpy: v2.5.2 | lib/tests/test_shape_base.py::TestTile::test_basic (L755)
        crate::specify_test!("test_basic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([0, 1, 2])
        let a = rt::tensor_from_nested!([0, 1, 2], &device);
        // b = [[1, 2], [3, 4]]
        let b = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);

        // assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        let expected = rt::tensor_from_nested!([0, 1, 2, 0, 1, 2], &device);
        assert_equal(a.tile(2), &expected, None);

        // assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        let expected = rt::tensor_from_nested!([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]], &device);
        assert_equal(a.tile([2, 2]), &expected, None);

        // assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        let expected = rt::tensor_from_nested!([[0, 1, 2, 0, 1, 2]], &device);
        assert_equal(a.tile([1, 2]), &expected, None);

        // assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        let expected = rt::tensor_from_nested!([[1, 2, 1, 2], [3, 4, 3, 4]], &device);
        assert_equal(b.tile(2), &expected, None);

        // assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        let expected = rt::tensor_from_nested!([[1, 2], [3, 4], [1, 2], [3, 4]], &device);
        assert_equal(b.tile([2, 1]), &expected, None);

        // assert_equal(tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4],
        //                                [1, 2, 1, 2], [3, 4, 3, 4]])
        let expected = rt::tensor_from_nested!([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]], &device);
        assert_equal(b.tile([2, 2]), &expected, None);
    }

    #[test]
    fn test_tile_one_repetition_on_array_gh4679() {
        // numpy: v2.5.2 |
        // lib/tests/test_shape_base.py::TestTile::test_tile_one_repetition_on_array_gh4679 (L766)
        // NumPy checks tile(a, 1) copies (b += 2 does not touch a); rstsr's
        // tile always returns freshly owned data, so the equivalent guarantee
        // is checked by mutating the result.
        crate::specify_test!("test_tile_one_repetition_on_array_gh4679");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((5, &device));
        let mut b = a.tile(1);
        b.i_mut((0,)).fill(100);
        let expected = rt::tensor_from_nested!([0, 1, 2, 3, 4], &device);
        assert_equal(a, &expected, None);
    }

    #[test]
    fn test_empty() {
        // numpy: v2.5.2 | lib/tests/test_shape_base.py::TestTile::test_empty (L772)
        crate::specify_test!("test_empty");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a = np.array([[[]]])
        // d = tile(a, (3, 2, 5)).shape == (3, 2, 0)
        let a: Tensor<i32, _> = rt::zeros(([1, 0], &device));
        let d = a.tile([3, 2, 5]);
        assert_eq!(d.shape(), &[3, 2, 0]);

        // b = np.array([[], []])
        // c = tile(b, 2).shape == (2, 0)
        let b: Tensor<i32, _> = rt::zeros(([2, 0], &device));
        let c = b.tile(2);
        assert_eq!(c.shape(), &[2, 0]);
    }
}

#[cfg(test)]
mod custom_tile {
    use super::*;
    static FUNC: &str = "custom_tile";

    #[test]
    fn test_rank_promotion_up() {
        // repetitions longer than ndim: singleton axes prepended to x
        // np.tile(np.array([0, 1, 2]), (2, 1, 2)) has shape (2, 1, 6)
        crate::specify_test!("test_rank_promotion_up");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([0, 1, 2], &device);
        let t = a.tile([2, 1, 2]);
        assert_eq!(t.shape(), &[2, 1, 6]);
        let expected = rt::tensor_from_nested!([[[0, 1, 2, 0, 1, 2]], [[0, 1, 2, 0, 1, 2]]], &device);
        assert_equal(t, &expected, None);
    }

    #[test]
    fn test_strided_input() {
        crate::specify_test!("test_strided_input");

        // tiling a transposed view visits elements in view order
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.tile(np.arange(6).reshape(2, 3).T, (1, 2))
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t();
        let expected = rt::tensor_from_nested!([[0, 3, 0, 3], [1, 4, 1, 4], [2, 5, 2, 5]], &device);
        assert_equal(at.tile([1, 2]), &expected, None);
    }

    #[test]
    fn test_zero_repetition() {
        crate::specify_test!("test_zero_repetition");

        // repetition of zero gives an empty axis
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let out = a.tile([1, 0]);
        assert_eq!(out.shape(), &[2, 0]);
    }

    #[test]
    fn test_zero_size_input_axis() {
        // tiling along a zero-size input axis must not panic: layout offsets
        // may point past the empty source storage (G-072)
        crate::specify_test!("test_zero_size_input_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::zeros(([0, 4], &device));
        let out = a.tile([2, 2]);
        assert_eq!(out.shape(), &[0, 8]);

        let b: Tensor<f64, _> = rt::zeros(([2, 0], &device));
        let out = b.tile([3]);
        assert_eq!(out.shape(), &[2, 0]);
    }

    #[test]
    fn test_0d() {
        crate::specify_test!("test_0d");

        // tiling a 0-d tensor promotes it to 1-d
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::full(([], 5, &device));
        let out = a.tile(3);
        assert_eq!(out.shape(), &[3]);
        let expected = rt::tensor_from_nested!([5, 5, 5], &device);
        assert_equal(out, &expected, None);
    }

    #[test]
    fn test_3d() {
        crate::specify_test!("test_3d");

        // multi-axis tiling of a 3-d tensor
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let out = a.tile([1, 2, 1]);
        assert_eq!(out.shape(), &[2, 4, 2]);
        let expected =
            rt::tensor_from_nested!([[[0, 1], [2, 3], [0, 1], [2, 3]], [[4, 5], [6, 7], [4, 5], [6, 7]]], &device);
        assert_equal(out, &expected, None);
    }
}
