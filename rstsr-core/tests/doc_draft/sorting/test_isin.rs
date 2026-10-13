#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_isin {
    use super::*;
    static FUNC: &str = "doc_isin";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: element membership
        let a = rt::tensor_from_nested!([1, 2, 3, 4], &device);
        let b = rt::tensor_from_nested!([2, 4], &device);
        println!("{}", rt::isin(&a, &b, false));
        // [ false true false true]
        assert_eq!(format!("{}", rt::isin(&a, &b, false)), "[ false true false true]");
    }
}
