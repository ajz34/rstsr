use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

mod doc_moveaxis {
    use super::*;
    static FUNC: &str = "doc_moveaxis";

    #[test]
    fn test_doc() {
        // Test that the documentation examples for moveaxis work correctly.
        // Based on NumPy v2.4.2 moveaxis docstring examples.
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Move a single axis to a new position
        let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
        let result = x.moveaxis(0, -1);
        println!("{:?}", result.shape());
        // [4, 5, 3]
        assert_eq!(result.shape(), &[4, 5, 3]);

        // Move multiple axes to new positions
        let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
        let result = x.moveaxis([0, 1], [-1, -2]);
        println!("{:?}", result.shape());
        // [5, 4, 3]
        assert_eq!(result.shape(), &[5, 4, 3]);

        // Using negative indices
        let x: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
        let result = x.moveaxis(-1, 0);
        println!("{:?}", result.shape());
        // [5, 3, 4]
        assert_eq!(result.shape(), &[5, 3, 4]);
    }

    #[test]
    fn overload_list() {
        crate::specify_test!("overload_list");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // single integer source and destination: move one axis
        let x = rt::arange((6, &device)).into_shape([2, 3]);
        println!("{}", rt::moveaxis(&x, 0, 1));
        // [[ 0 3]
        //  [ 1 4]
        //  [ 2 5]]
        assert_eq!(format!("{}", rt::moveaxis(&x, 0, 1)), "[[ 0 3]\n [ 1 4]\n [ 2 5]]");

        // list source and destination: move several axes
        let y: Tensor<f64, _> = rt::zeros(([3, 4, 5], &device));
        println!("{:?}", rt::moveaxis(&y, [0, 1], [-1, -2]).shape());
        // [5, 4, 3]
        assert_eq!(rt::moveaxis(&y, [0, 1], [-1, -2]).shape(), &[5, 4, 3]);

        // negative indices count from the back
        println!("{:?}", rt::moveaxis(&y, -1, 0).shape());
        // [5, 3, 4]
        assert_eq!(rt::moveaxis(&y, -1, 0).shape(), &[5, 3, 4]);
    }

    #[test]
    fn test_data_integrity() {
        // Test that moveaxis preserves data correctly
        crate::specify_test!("test_data_integrity");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // Create a tensor with sequential data
        let x: Tensor<i32, _> = rt::arange((24, &device)).into_shape([2, 3, 4]);
        println!("Original:\n{:?}", x);

        // Move axis 0 to the end
        let result = x.moveaxis(0, -1);
        println!("After moveaxis(0, -1):\n{:?}", result);
        assert_eq!(result.shape(), &[3, 4, 2]);

        // Verify data by checking specific elements
        // Original shape [2, 3, 4] -> [3, 4, 2]
        // result[0, 0, 0] should be x[0, 0, 0] = 0
        // result[0, 0, 1] should be x[1, 0, 0] = 12
        // result[1, 2, 0] should be x[0, 1, 2] = 6
        assert_eq!(result[[0, 0, 0]], 0);
        assert_eq!(result[[0, 0, 1]], 12);
        assert_eq!(result[[1, 2, 0]], 6);

        // Move axis 2 to the beginning
        let result2 = x.moveaxis(2, 0);
        println!("After moveaxis(2, 0):\n{:?}", result2);
        assert_eq!(result2.shape(), &[4, 2, 3]);

        // Verify data
        // result2[0, 0, 0] should be x[0, 0, 0] = 0
        // result2[1, 0, 0] should be x[0, 0, 1] = 1
        // result2[3, 1, 2] should be x[1, 2, 3] = 23
        assert_eq!(result2[[0, 0, 0]], 0);
        assert_eq!(result2[[1, 0, 0]], 1);
        assert_eq!(result2[[3, 1, 2]], 23);
    }
}
