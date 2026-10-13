#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_searchsorted {
    use super::*;
    static FUNC: &str = "doc_searchsorted";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: default side
        let x1 = rt::tensor_from_nested!([11, 12, 14, 15, 16], &device);
        let x2 = rt::tensor_from_nested!([10, 13, 17], &device);
        println!("{}", rt::searchsorted(&x1, &x2, ()));
        // [ 0 2 5]
        assert_eq!(format!("{}", rt::searchsorted(&x1, &x2, ())), "[ 0 2 5]");

        // # Examples: insertion side
        let y1 = rt::tensor_from_nested!([10, 20, 30], &device);
        let v = rt::tensor_from_nested!([20], &device);
        println!("{}", rt::searchsorted(&y1, &v, "left"));
        // [ 1]
        assert_eq!(format!("{}", rt::searchsorted(&y1, &v, "left")), "[ 1]");
        println!("{}", rt::searchsorted(&y1, &v, "right"));
        // [ 2]
        assert_eq!(format!("{}", rt::searchsorted(&y1, &v, "right")), "[ 2]");

        // # Overload List: one call per distinct behavior
        println!("{}", rt::searchsorted(&x1, &x2, ())); // defaults
                                                        // [ 0 2 5]
        assert_eq!(format!("{}", rt::searchsorted(&x1, &x2, ())), "[ 0 2 5]");
        println!("{}", rt::searchsorted(&y1, &v, SearchSide::Right)); // enum side
                                                                      // [ 2]
        assert_eq!(format!("{}", rt::searchsorted(&y1, &v, SearchSide::Right)), "[ 2]");

        // sorter: permutation sorting x1 ascending ([2, 1, 3] -> [1, 2, 3])
        let u1 = rt::tensor_from_nested!([2, 1, 3], &device);
        let w = rt::tensor_from_nested!([3], &device);
        println!("{}", rt::searchsorted(&u1, &w, vec![1usize, 0, 2])); // sorter only
                                                                       // [ 2]
        assert_eq!(format!("{}", rt::searchsorted(&u1, &w, vec![1usize, 0, 2])), "[ 2]");
        println!("{}", rt::searchsorted(&u1, &w, (SearchSide::Left, vec![1usize, 0, 2]))); // side +
                                                                                           // sorter
                                                                                           // [ 2]
        assert_eq!(format!("{}", rt::searchsorted(&u1, &w, (SearchSide::Left, vec![1usize, 0, 2]))), "[ 2]");
    }
}
