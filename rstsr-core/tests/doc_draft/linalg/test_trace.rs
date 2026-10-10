use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_trace {
    use super::*;
    static FUNC: &str = "doc_trace";

    #[test]
    fn test_trace() {
        crate::specify_test!("test_trace");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // trace of a stack: each 2x2 matrix's diagonal is summed
        let a = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        let t = rt::trace(&a, ());
        println!("{t}");
        // [ 5 13]
        assert_eq!(format!("{t}"), "[ 5 13]");

        // a two-dimensional input gives a zero-dimensional result
        let m = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        assert_eq!(format!("{}", rt::trace(&m, ())), "5");
    }
}
