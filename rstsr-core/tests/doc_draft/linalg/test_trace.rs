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
        let t = rt::trace(&a, None);
        println!("{t}");
        // [ 5 13]
        assert_eq!(format!("{t}"), "[ 5 13]");

        // a two-dimensional input gives a zero-dimensional result
        let m = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
        assert_eq!(format!("{}", rt::trace(&m, None)), "5");
    }
}

// # Overload List snippet: one call per distinct offset behavior
#[cfg(test)]
mod doc_trace_overload {
    use super::*;
    static FUNC: &str = "doc_trace_overload";

    #[test]
    fn test_trace() {
        crate::specify_test!("test_trace");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);

        // () or None: the main diagonal over the device-default axes
        println!("{}", rt::trace(&a, None));
        // 15
        assert_eq!(format!("{}", rt::trace(&a, None)), "15");

        // integer: a shifted diagonal, same device-default axes
        println!("{}", rt::trace(&a, 1));
        // 8
        assert_eq!(format!("{}", rt::trace(&a, 1)), "8");

        // (offset, axis1, axis2): an explicit diagonal over named axes
        let s = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        println!("{}", rt::trace(&s, (0, -2, -1)));
        // [ 5 13]
        assert_eq!(format!("{}", rt::trace(&s, (0, -2, -1))), "[ 5 13]");
    }
}
