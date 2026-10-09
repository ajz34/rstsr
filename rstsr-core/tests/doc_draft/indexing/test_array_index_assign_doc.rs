//! doc_draft for `array_index_assign` (array-indexing assignment / setter).

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_array_index_assign {
    use super::*;
    static FUNC: &str = "doc_array_index_assign";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a host list indexes the first axis; the value is the two matching rows
        let mut a = rt::arange((12, &device)).into_shape([3, 4]);
        let rows = rt::tensor_from_nested!([[100, 101, 102, 103], [-1, -2, -3, -4]], &device);
        a.array_index_assign([2, 0], &rows);
        println!("{a}");
        // [[ -1 -2 -3 -4]
        //  [ 4 5 6 7]
        //  [ 100 101 102 103]]
        assert_eq!(format!("{a}"), "[[ -1 -2 -3 -4]\n [ 4 5 6 7]\n [ 100 101 102 103]]");
        assert_eq!(a.into_shape([-1]).to_vec(), vec![-1, -2, -3, -4, 4, 5, 6, 7, 100, 101, 102, 103]);

        // a boolean mask selects the rows it keeps (the value is cast to f64)
        let mut a: Tensor<f64, _> = rt::zeros(([3, 2], &device));
        let mask = rt::asarray((vec![true, false, true], &device));
        a.array_index_assign(&mask, rt::full(([2, 2], 1.5f64, &device)));
        println!("{a}");
        // [[ 1.5 1.5]
        //  [ 0 0]
        //  [ 1.5 1.5]]
        assert_eq!(format!("{a}"), "[[ 1.5 1.5]\n [ 0 0]\n [ 1.5 1.5]]");
        assert_eq!(a.into_shape([-1]).to_vec(), vec![1.5, 1.5, 0.0, 0.0, 1.5, 1.5]);

        // duplicate index targets: the value written last wins
        let mut e: Tensor<i64, _> = rt::zeros(([4], &device));
        e.array_index_assign([3, 3, 3], rt::tensor_from_nested!([1i64, 2, 3], &device));
        assert_eq!(e.to_vec(), vec![0, 0, 0, 3]);
    }
}
