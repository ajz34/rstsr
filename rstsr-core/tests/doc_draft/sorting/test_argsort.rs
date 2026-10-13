#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_argsort {
    use super::*;
    static FUNC: &str = "doc_argsort";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: default argsort
        let a = rt::tensor_from_nested!([3, 1, 2], &device);
        println!("{}", rt::argsort(&a, ()));
        // [ 1 2 0]
        assert_eq!(format!("{}", rt::argsort(&a, ())), "[ 1 2 0]");

        // # Overload List: one call per distinct behavior
        println!("{}", rt::argsort(&a, ())); // defaults
                                             // [ 1 2 0]
        assert_eq!(format!("{}", rt::argsort(&a, ())), "[ 1 2 0]");
        println!("{}", rt::argsort(&a, 0)); // integer axis
                                            // [ 1 2 0]
        assert_eq!(format!("{}", rt::argsort(&a, 0)), "[ 1 2 0]");
        println!("{}", rt::argsort(&a, true)); // descending
                                               // [ 0 2 1]
        assert_eq!(format!("{}", rt::argsort(&a, true)), "[ 0 2 1]");
        println!("{}", rt::argsort(&a, (0, true))); // axis + descending
                                                    // [ 0 2 1]
        assert_eq!(format!("{}", rt::argsort(&a, (0, true))), "[ 0 2 1]");
        println!("{}", rt::argsort(&a, (0, false, true))); // axis + ascending + stable
                                                           // [ 1 2 0]
        assert_eq!(format!("{}", rt::argsort(&a, (0, false, true))), "[ 1 2 0]");
    }
}
