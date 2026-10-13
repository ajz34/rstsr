#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_sort {
    use super::*;
    static FUNC: &str = "doc_sort";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: ascending sort of a 1-D tensor
        let a = rt::tensor_from_nested!([3, 1, 2], &device);
        println!("{}", rt::sort(&a, ()));
        // [ 1 2 3]
        assert_eq!(format!("{}", rt::sort(&a, ())), "[ 1 2 3]");

        // # Examples: axis matters on a 2-D tensor
        let b = rt::tensor_from_nested!([[9, 1, 5], [2, 8, 3]], &device);
        println!("{}", rt::sort(&b, 0));
        // [[ 2 1 3]
        //  [ 9 8 5]]
        assert_eq!(format!("{}", rt::sort(&b, 0)), "[[ 2 1 3]\n [ 9 8 5]]");
        println!("{}", rt::sort(&b, ()));
        // [[ 1 5 9]
        //  [ 2 3 8]]
        assert_eq!(format!("{}", rt::sort(&b, ())), "[[ 1 5 9]\n [ 2 3 8]]");

        // # Overload List: one call per distinct behavior
        let a = rt::tensor_from_nested!([3, 1, 2], &device);
        println!("{}", rt::sort(&a, ())); // defaults
                                          // [ 1 2 3]
        assert_eq!(format!("{}", rt::sort(&a, ())), "[ 1 2 3]");
        println!("{}", rt::sort(&a, 0)); // integer axis
                                         // [ 1 2 3]
        assert_eq!(format!("{}", rt::sort(&a, 0)), "[ 1 2 3]");
        println!("{}", rt::sort(&a, true)); // descending
                                            // [ 3 2 1]
        assert_eq!(format!("{}", rt::sort(&a, true)), "[ 3 2 1]");
        println!("{}", rt::sort(&a, (0, true))); // axis + descending
                                                 // [ 3 2 1]
        assert_eq!(format!("{}", rt::sort(&a, (0, true))), "[ 3 2 1]");
        println!("{}", rt::sort(&a, (0, true, true))); // axis + descending + stable
                                                       // [ 3 2 1]
        assert_eq!(format!("{}", rt::sort(&a, (0, true, true))), "[ 3 2 1]");
    }
}
