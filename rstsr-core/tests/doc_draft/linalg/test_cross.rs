use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_cross {
    use super::*;
    static FUNC: &str = "doc_cross";

    #[test]
    fn test_cross() {
        crate::specify_test!("test_cross");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // cross of two 3-vectors
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5, 6], &device);
        let c = rt::cross(&a, &b, -1);
        println!("{c}");
        // [ -3 6 -3]
        assert_eq!(format!("{c}"), "[ -3 6 -3]");
        assert_eq!(c.to_vec(), vec![-3, 6, -3]);

        // explicit axis, with the operands broadcast on the other axes
        let a2 = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let c2 = rt::cross(&a2, &b, 1);
        println!("{c2}");
        // [[ -3 6 -3]
        //  [ 0 0 0]]
        assert_eq!(format!("{c2}"), "[[ -3 6 -3]\n [ 0 0 0]]");
        assert_eq!(c2.shape(), &[2, 3]);
    }
}

// # Overload List snippet: one call per distinct axis behavior
#[cfg(test)]
mod doc_cross_overload {
    use super::*;
    static FUNC: &str = "doc_cross_overload";

    #[test]
    fn test_cross() {
        crate::specify_test!("test_cross");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5, 6], &device);

        // () or None: the last axis (Array-API default axis = -1)
        println!("{}", rt::cross(&a, &b, ()));
        // [ -3 6 -3]
        assert_eq!(format!("{}", rt::cross(&a, &b, ())), "[ -3 6 -3]");

        // integer: an explicit axis shared by both operands
        println!("{}", rt::cross(&a, &b, -1));
        // [ -3 6 -3]
        assert_eq!(format!("{}", rt::cross(&a, &b, -1)), "[ -3 6 -3]");

        let m = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        println!("{}", rt::cross(&m, &b, 1));
        // [[ -3 6 -3]
        //  [ 0 0 0]]
        assert_eq!(format!("{}", rt::cross(&m, &b, 1)), "[[ -3 6 -3]\n [ 0 0 0]]");
    }
}
