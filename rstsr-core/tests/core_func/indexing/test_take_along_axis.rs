//! take_along_axis tests: NumPy-cited gather contract (TestTakeAlongAxis) + edges.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_take_along_axis {
    use super::*;
    static FUNC: &str = "numpy_take_along_axis";

    #[test]
    fn test_argequivalent() {
        // numpy: v2.5.2 | lib/tests/test_shape_base.py::TestTakeAlongAxis::test_argequivalent (L41)
        // take_along_axis(a, argsort(a, axis), axis) == sort(a, axis) for every axis.
        crate::specify_test!("test_argequivalent");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = (rt::arange((60, &device)) * 7).mapv(|x| x % 13).into_shape([3, 4, 5]);
        for axis in 0..a.ndim() {
            let idx = a.argsort(axis as isize).mapv(|v| v as isize);
            let gathered = a.take_along_axis(&idx, axis as isize);
            assert_equal(gathered, rt::sort((&a, axis as isize)), None);
        }
    }

    #[test]
    fn test_invalid() {
        // numpy: v2.5.2 | lib/tests/test_shape_base.py::TestTakeAlongAxis::test_invalid (L59)
        // bool/float index dtypes and axis=None are type-level or API-level N/A
        // (rstsr indices are `isize`, axis is required); the dimensional and
        // axis-range errors transfer.
        crate::specify_test!("test_invalid");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::ones(([10, 10], &device));
        // not enough indices (0-d index tensor)
        let idx0 = rt::full(([], 1_isize, &device));
        assert!(rt::take_along_axis_f(&a, &idx0, 1).is_err());
        // invalid axis
        let ai = rt::full(([10, 2], 1_isize, &device));
        assert!(rt::take_along_axis_f(&a, &ai, 10).is_err());
    }

    #[test]
    fn test_empty() {
        // numpy: v2.5.2 | lib/tests/test_shape_base.py::TestTakeAlongAxis::test_empty (L78)
        crate::specify_test!("test_empty");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::ones(([3, 4, 5], &device));
        let ai: Tensor<isize, _> = rt::ones(([3, 0, 5], &device));
        let actual = rt::take_along_axis((&a, &ai, 1));
        assert_eq!(actual.shape(), &[3, 0, 5]);
    }

    #[test]
    fn test_broadcast() {
        // numpy: v2.5.2 | lib/tests/test_shape_base.py::TestTakeAlongAxis::test_broadcast (L86)
        // non-indexing dimensions broadcast in both directions: the tensor may
        // own the size-1 dimension (not only the indices).
        crate::specify_test!("test_broadcast");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a: Tensor<f64, _> = rt::ones(([3, 4, 1], &device));
        let ai: Tensor<isize, _> = rt::ones(([1, 2, 5], &device));
        let actual = rt::take_along_axis((&a, &ai, 1));
        assert_eq!(actual.shape(), &[3, 2, 5]);
    }
}

#[cfg(test)]
mod custom_take_along_axis {
    use super::*;
    static FUNC: &str = "custom_take_along_axis";

    #[test]
    fn test_basic_2d_last_axis() {
        // np.take_along_axis([[10,20,30],[40,50,60]], [[2,0],[1,1]], axis=-1)
        // == [[30,10],[50,50]]
        crate::specify_test!("test_basic_2d_last_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[2_isize, 0], [1, 1]], &device);
        let out = rt::take_along_axis((&a, &idx, -1));
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![30, 10, 50, 50]);
    }

    #[test]
    fn test_axis_0() {
        // gathering along axis 0: out[j, i] = a[indices[j, i], i]
        crate::specify_test!("test_axis_0");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[1_isize, 0, 1]], &device);
        let out = rt::take_along_axis((&a, &idx, 0));
        assert_eq!(out.shape(), &[1, 3]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![40, 20, 60]);
    }

    #[test]
    fn test_argsort_roundtrip() {
        // take_along_axis(a, argsort(a, axis), axis) == sort(a, axis)
        crate::specify_test!("test_argsort_roundtrip");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = (rt::arange((20, &device)) * 7).mapv(|x| x % 11).into_shape([4, 5]);
        let idx = a.argsort(1).mapv(|v| v as isize);
        let gathered = a.take_along_axis(&idx, 1);
        let sorted = a.sort(1);
        assert_equal(gathered, &sorted, None);
    }

    #[test]
    fn test_strided_input() {
        crate::specify_test!("test_strided_input");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t(); // [[0,3],[1,4],[2,5]]
        let idx = rt::tensor_from_nested!([[1_isize, 0], [0, 1], [1, 1]], &device);
        let out = rt::take_along_axis((&at, &idx, -1));
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![3, 0, 1, 4, 5, 5]);
    }

    #[test]
    fn test_negative_indices() {
        // negatives count from the back (NumPy: [[2,-3],[1,1]] == [[2,0],[1,1]])
        crate::specify_test!("test_negative_indices");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[2_isize, -3], [1, 1]], &device);
        let out = rt::take_along_axis((&a, &idx, -1));
        assert_eq!(out.reshape([-1]).to_vec(), vec![30, 10, 50, 50]);
    }

    #[test]
    fn test_index_out_of_range() {
        crate::specify_test!("test_index_out_of_range");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[3_isize, 0]], &device);
        assert!(rt::take_along_axis_f(&a, &idx, -1).is_err());
    }

    #[test]
    fn test_shape_mismatch_outside_axis() {
        // non-broadcast-compatible shapes outside the axis raise (3 vs 2);
        // dim-1 rows would be legal (broadcast per the array-api standard)
        crate::specify_test!("test_shape_mismatch_outside_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[2_isize, 0, 1], [0, 1, 2], [1, 2, 0]], &device);
        assert!(rt::take_along_axis_f(&a, &idx, -1).is_err());
    }

    #[test]
    fn test_broadcast_indices() {
        // array-api 2025.12: indices broadcast against x outside the axis;
        // output shape follows that broadcasting (NumPy parity)
        crate::specify_test!("test_broadcast_indices");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // np.take_along_axis(arange(12).reshape(3,4), [[3,0]], axis=1)
        // -> [[3,0],[7,4],[11,8]]
        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let idx = rt::tensor_from_nested!([[3_isize, 0]], &device);
        let out = rt::take_along_axis((&a, &idx, -1));
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![3, 0, 7, 4, 11, 8]);
    }

    #[test]
    fn test_transposed_and_broadcast_index_views() {
        // index tensors as views (transposed / broadcast / sliced) are read
        // in logical order, not storage order (external review finding 2)
        crate::specify_test!("test_transposed_and_broadcast_index_views");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((12, &device)).into_shape([3, 4]);
        // transposed index view: idx (4,3) transposed to (3,4); each row of
        // the logical (3,4) index reads column-wise from storage
        // idx_t[i][j] = idx[j][i] = j % 4 (a rotation); numpy reference:
        // np.take_along_axis(a, np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]]).T, axis=1)
        let idx_mat = rt::tensor_from_nested!([[0_isize, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], &device);
        let idx_t = idx_mat.t();
        let out = rt::take_along_axis((&a, &idx_t, -1));
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        // broadcast (stride-0) index view: (1,2) -> (3,2)
        let idx_b = rt::tensor_from_nested!([[3_isize, 0]], &device);
        let idx_bc: TensorView<isize, _, Vec<usize>> = idx_b.broadcast_to(vec![3, 2]);
        let out = rt::take_along_axis((&a, &idx_bc, -1));
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![3, 0, 7, 4, 11, 8]);
        // sliced index view (non-zero storage offset): (3,4) sliced to (3,2)
        let idx_full =
            rt::tensor_from_nested!([[9_isize, 9, 9, 9], [0_isize, 1, 9, 9], [2, 1, 9, 9], [3, 0, 9, 9]], &device);
        let idx_sliced = idx_full.i((1.., 0..2));
        let out2 = rt::take_along_axis((&a, &idx_sliced, -1));
        assert_eq!(out2.shape(), &[3, 2]);
        // idx rows [0,1],[2,1],[3,0]: a[0,0],a[0,1] | a[1,2],a[1,1] | a[2,3],a[2,0]
        assert_eq!(out2.reshape([-1]).to_vec(), vec![0, 1, 6, 5, 11, 8]);
    }

    #[test]
    fn test_zero_length_axis() {
        // indices with a 0-length indexed axis give an empty output
        crate::specify_test!("test_zero_length_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx: Tensor<isize, _> = rt::zeros(([2, 0], &device));
        let out = rt::take_along_axis((&a, &idx, -1));
        assert_eq!(out.shape(), &[2, 0]);
    }
}
