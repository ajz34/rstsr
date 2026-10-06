//! Nonzero tests: NumPy-cited row-major coordinate contract + edges.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod custom_nonzero {
    use super::*;
    static FUNC: &str = "custom_nonzero";

    #[test]
    fn test_nonzero_2d() {
        // np.nonzero([[1, 0, 2], [0, 3, 0]]) -> (array([0, 0, 1]), array([0, 2, 1]))
        crate::specify_test!("test_nonzero_2d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 0, 2], [0, 3, 0]], &device);
        let coords = rt::nonzero(&a);
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0].to_vec(), vec![0, 0, 1]);
        assert_eq!(coords[1].to_vec(), vec![0, 2, 1]);
    }

    #[test]
    fn test_nonzero_row_major_order() {
        // strict row-major element order even for strided input
        // at = arange(6).reshape(2,3).T: content [0 3; 1 4; 2 5]
        // nonzero -> coords ([0,1,1,2,2],[1,0,2,0,2]) per row-major walk
        crate::specify_test!("test_nonzero_row_major_order");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let at = a.t();
        let coords = rt::nonzero(&at);
        assert_eq!(coords[0].to_vec(), vec![0, 1, 1, 2, 2]);
        assert_eq!(coords[1].to_vec(), vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_nonzero_bool_complex() {
        // bool: true is nonzero; complex: either component nonzero
        crate::specify_test!("test_nonzero_bool_complex");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([true, false, true], &device);
        let coords = rt::nonzero(&a);
        assert_eq!(coords[0].to_vec(), vec![0, 2]);

        let c = rt::asarray((
            vec![num::Complex::new(0.0_f64, 1.0), num::Complex::new(0.0, 0.0), num::Complex::new(2.0, 0.0)],
            &device,
        ));
        let coords = rt::nonzero(&c);
        assert_eq!(coords[0].to_vec(), vec![0, 2]);
    }

    #[test]
    fn test_nonzero_all_zero_and_all_nonzero() {
        crate::specify_test!("test_nonzero_all_zero_and_all_nonzero");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let z: Tensor<i32, _> = rt::zeros(([2, 2], &device));
        let coords = rt::nonzero(&z);
        assert_eq!(coords[0].shape(), &[0]);

        let o: Tensor<i32, _> = rt::ones(([2, 2], &device));
        let coords = rt::nonzero(&o);
        assert_eq!(coords[0].to_vec(), vec![0, 0, 1, 1]);
        assert_eq!(coords[1].to_vec(), vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_nonzero_1d_and_nan() {
        // 1-d returns one tensor; NaN is nonzero (NaN != 0)
        crate::specify_test!("test_nonzero_1d_and_nan");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([0.0, 5.0, 0.0], &device);
        let coords = rt::nonzero(&a);
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0].to_vec(), vec![1]);

        let n = rt::tensor_from_nested!([f64::NAN, 0.0], &device);
        let coords = rt::nonzero(&n);
        assert_eq!(coords[0].to_vec(), vec![0]);
    }

    #[test]
    fn test_nonzero_0d_errors() {
        // 0-d input raises (no axis to index); NumPy succeeds returning empty
        // coordinate arrays — deviation registered in DECISIONS
        crate::specify_test!("test_nonzero_0d_errors");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::full(([], 1, &device));
        assert!(rt::nonzero_f(&a).is_err());
    }
}
