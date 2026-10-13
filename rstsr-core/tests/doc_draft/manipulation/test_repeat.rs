#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_repeat {
    use super::*;
    static FUNC: &str = "doc_repeat";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List snippet: one call per distinct behavior
        let a = rt::arange((3, &device));
        // scalar count: every element repeats twice (flattened form)
        println!("{}", rt::repeat(&a, 2));
        // [ 0 0 1 1 2 2]
        assert_eq!(format!("{}", rt::repeat(&a, 2)), "[ 0 0 1 1 2 2]");
        // list of per-element counts (flattened form)
        println!("{}", rt::repeat(&a, [3, 1, 2]));
        // [ 0 0 0 1 2 2]
        assert_eq!(format!("{}", rt::repeat(&a, [3, 1, 2])), "[ 0 0 0 1 2 2]");

        let b = rt::arange((6, &device)).into_shape([2, 3]);
        // tuple form with an integer axis
        println!("{}", rt::repeat(&b, ([2, 1], 0)));
        // [[ 0 1 2]
        //  [ 0 1 2]
        //  [ 3 4 5]]
        assert_eq!(format!("{}", rt::repeat(&b, ([2, 1], 0))), "[[ 0 1 2]\n [ 0 1 2]\n [ 3 4 5]]");
        // tuple form with axis = None: the flattened form
        println!("{}", rt::repeat(&b, (1, None)));
        // [ 0 1 2 3 4 5]
        assert_eq!(format!("{}", rt::repeat(&b, (1, None))), "[ 0 1 2 3 4 5]");

        // Overloads Table coverage: both listed forms
        // form 1: bare repeats (per-element counts)
        println!("{}", rt::repeat(&a, [1, 2, 1]));
        // [ 0 1 1 2]
        assert_eq!(format!("{}", rt::repeat(&a, [1, 2, 1])), "[ 0 1 1 2]");
        // form 2: (repeats, axis) with an integer axis
        let c = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
        println!("{}", rt::repeat(&c, (2, 1)));
        // [[ 0 0 1 1 2 2]
        //  [ 3 3 4 4 5 5]]
        assert_eq!(format!("{}", rt::repeat(&c, (2, 1))), "[[ 0 0 1 1 2 2]\n [ 3 3 4 4 5 5]]");
    }
}
