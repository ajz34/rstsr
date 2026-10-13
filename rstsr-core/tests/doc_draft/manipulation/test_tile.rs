#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_tile {
    use super::*;
    static FUNC: &str = "doc_tile";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List snippet: one call per distinct behavior
        let a = rt::arange((3, &device));
        // integer repetition: tiles along a new trailing axis
        println!("{}", rt::tile(&a, 2));
        // [ 0 1 2 0 1 2]
        assert_eq!(format!("{}", rt::tile(&a, 2)), "[ 0 1 2 0 1 2]");
        // list of per-axis repetitions (rank promotion)
        println!("{}", rt::tile(&a, [2, 2]));
        // [[ 0 1 2 0 1 2]
        //  [ 0 1 2 0 1 2]]
        assert_eq!(format!("{}", rt::tile(&a, [2, 2])), "[[ 0 1 2 0 1 2]\n [ 0 1 2 0 1 2]]");
        // None: empty repetitions (all ones), returns a copy
        println!("{}", rt::tile(&a, None));
        // [ 0 1 2]
        assert_eq!(format!("{}", rt::tile(&a, None)), "[ 0 1 2]");

        // tiling a 2-D tensor along the second axis only
        let b = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        println!("{}", rt::tile(&b, [1, 2]));
        // [[ 1 2 1 2]
        //  [ 3 4 3 4]]
        assert_eq!(format!("{}", rt::tile(&b, [1, 2])), "[[ 1 2 1 2]\n [ 3 4 3 4]]");
    }
}
