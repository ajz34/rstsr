//! take_along_axis tests: NumPy-cited gather contract + edges.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

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
        let idx = rt::tensor_from_nested!([[2_usize, 0], [1, 1]], &device);
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
        let idx = rt::tensor_from_nested!([[1_usize, 0, 1]], &device);
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
        let idx = a.argsort(1);
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
        let idx = rt::tensor_from_nested!([[1_usize, 0], [0, 1], [1, 1]], &device);
        let out = rt::take_along_axis((&at, &idx, -1));
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.reshape([-1]).to_vec(), vec![3, 0, 1, 4, 5, 5]);
    }

    #[test]
    fn test_index_out_of_range() {
        crate::specify_test!("test_index_out_of_range");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[3_usize, 0]], &device);
        assert!(rt::take_along_axis_f(&a, &idx, -1).is_err());
    }

    #[test]
    fn test_shape_mismatch_outside_axis() {
        crate::specify_test!("test_shape_mismatch_outside_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx = rt::tensor_from_nested!([[2_usize, 0, 1]], &device);
        assert!(rt::take_along_axis_f(&a, &idx, -1).is_err());
    }

    #[test]
    fn test_zero_length_axis() {
        // indices with a 0-length indexed axis give an empty output
        crate::specify_test!("test_zero_length_axis");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[10, 20, 30], [40, 50, 60]], &device);
        let idx: Tensor<usize, _> = rt::zeros(([2, 0], &device));
        let out = rt::take_along_axis((&a, &idx, -1));
        assert_eq!(out.shape(), &[2, 0]);
    }
}
