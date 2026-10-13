#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_argsort_custom {
    use super::*;
    static FUNC: &str = "doc_argsort_custom";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: indices for a custom comparator (sort by residue mod 2)
        let a = rt::tensor_from_nested!([3, 1, 2, 0], &device);
        let by_mod2 = |x: &i32, y: &i32| (x % 2).cmp(&(y % 2));
        println!("{}", rt::argsort_custom(&a, None, by_mod2));
        // [ 2 3 0 1]
        assert_eq!(format!("{}", rt::argsort_custom(&a, None, by_mod2)), "[ 2 3 0 1]");

        // # Overload List: one call per distinct behavior
        println!("{}", rt::argsort_custom(&a, 0, by_mod2)); // explicit axis
                                                            // [ 2 3 0 1]
        assert_eq!(format!("{}", rt::argsort_custom(&a, 0, by_mod2)), "[ 2 3 0 1]");
        println!("{}", rt::argsort_custom(&a, (0, true), by_mod2)); // axis + descending
                                                                    // [ 0 1 2 3]
        assert_eq!(format!("{}", rt::argsort_custom(&a, (0, true), by_mod2)), "[ 0 1 2 3]");
    }
}
