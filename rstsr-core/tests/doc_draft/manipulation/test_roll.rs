#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_roll {
    use super::*;
    static FUNC: &str = "doc_roll";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List snippet: one call per distinct behavior
        let a = rt::arange((6, &device));
        // bare integer shift -> flattened (axis = None)
        println!("{}", rt::roll(&a, 2));
        // [ 4 5 0 1 2 3]
        assert_eq!(format!("{}", rt::roll(&a, 2)), "[ 4 5 0 1 2 3]");
        // tuple form, axis = None -> flattened
        println!("{}", rt::roll(&a, (2, None)));
        // [ 4 5 0 1 2 3]
        assert_eq!(format!("{}", rt::roll(&a, (2, None))), "[ 4 5 0 1 2 3]");

        let b = rt::arange((9, &device)).into_shape([3, 3]);
        // integer axis
        println!("{}", rt::roll(&b, (1, 0)));
        // [[ 6 7 8]
        //  [ 0 1 2]
        //  [ 3 4 5]]
        assert_eq!(format!("{}", rt::roll(&b, (1, 0))), "[[ 6 7 8]\n [ 0 1 2]\n [ 3 4 5]]");
        // tuple shift with tuple axis
        println!("{}", rt::roll(&b, ((1, 1), (0, 1))));
        // [[ 8 6 7]
        //  [ 2 0 1]
        //  [ 5 3 4]]
        assert_eq!(format!("{}", rt::roll(&b, ((1, 1), (0, 1)))), "[[ 8 6 7]\n [ 2 0 1]\n [ 5 3 4]]");

        // Overloads Table coverage: every listed form
        // bare shift as a list -> flattened
        println!("{}", rt::roll(&a, [2]));
        // [ 4 5 0 1 2 3]
        assert_eq!(format!("{}", rt::roll(&a, [2])), "[ 4 5 0 1 2 3]");
        // integer shift broadcast to a tuple of axes
        println!("{}", rt::roll(&b, (1, [0, 1])));
        // [[ 8 6 7]
        //  [ 2 0 1]
        //  [ 5 3 4]]
        assert_eq!(format!("{}", rt::roll(&b, (1, [0, 1]))), "[[ 8 6 7]\n [ 2 0 1]\n [ 5 3 4]]");
    }
}
