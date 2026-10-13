#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_diff {
    use super::*;
    static FUNC: &str = "doc_diff";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // first-order differences of a 1-D tensor
        let x = rt::tensor_from_nested!([1, 4, 6, 7, 12], &device);
        let result = rt::diff(&x, 0, 1, None, None);
        println!("{result}");
        // [ 3 2 1 5]
        assert_eq!(format!("{result}"), "[ 3 2 1 5]");
        let target = rt::tensor_from_nested!([3, 2, 1, 5], &device);
        assert!(rt::allclose(&result, &target, None));

        // along an axis, with an explicit prepend column
        let m = rt::arange((4, &device)).into_shape([2, 2]);
        let p = rt::full(([2, 1], 0, &device));
        let result = rt::diff(&m, 1, 1, Some(&p), None);
        println!("{result}");
        // [[ 0 1]
        //  [ 2 1]]
        assert_eq!(format!("{result}"), "[[ 0 1]\n [ 2 1]]");
        let target = rt::tensor_from_nested!([[0, 1], [2, 1]], &device);
        assert!(rt::allclose(&result, &target, None));

        // # Overload List: one call per distinct axis behavior
        let y = rt::tensor_from_nested!([1, 4, 9, 16], &device);
        // default axis (-1)
        println!("{}", rt::diff(&y, -1, 1, None, None));
        // [ 3 5 7]
        assert_eq!(format!("{}", rt::diff(&y, -1, 1, None, None)), "[ 3 5 7]");
        // explicit axis
        println!("{}", rt::diff(&y, 0, 1, None, None));
        // [ 3 5 7]
        assert_eq!(format!("{}", rt::diff(&y, 0, 1, None, None)), "[ 3 5 7]");
        // None == last axis
        println!("{}", rt::diff(&y, None, 1, None, None));
        // [ 3 5 7]
        assert_eq!(format!("{}", rt::diff(&y, None, 1, None, None)), "[ 3 5 7]");
        let app = rt::tensor_from_nested!([25], &device);
        // second order, with append
        println!("{}", rt::diff(&y, -1, 2, None, Some(&app)));
        // [ 2 2 2]
        assert_eq!(format!("{}", rt::diff(&y, -1, 2, None, Some(&app))), "[ 2 2 2]");
    }
}
