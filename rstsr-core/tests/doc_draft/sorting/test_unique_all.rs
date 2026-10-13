#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_unique_all {
    use super::*;
    static FUNC: &str = "doc_unique_all";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: all four fields in one pass
        let a = rt::tensor_from_nested!([3, 1, 3], &device);
        let res = rt::unique_all(&a);
        println!("{}", res.values);
        // [ 1 3]
        println!("{}", res.indices);
        // [ 1 0]
        println!("{}", res.inverse_indices);
        // [ 1 0 1]
        println!("{}", res.counts);
        // [ 1 2]
        assert_eq!(format!("{}", res.values), "[ 1 3]");
        assert_eq!(format!("{}", res.indices), "[ 1 0]");
        assert_eq!(format!("{}", res.inverse_indices), "[ 1 0 1]");
        assert_eq!(format!("{}", res.counts), "[ 1 2]");

        // # Examples: the result struct converts into a plain 4-tuple
        let (values, indices, inverse_indices, counts) = res.into();
        assert_eq!(format!("{}", values), "[ 1 3]");
        assert_eq!(format!("{}", indices), "[ 1 0]");
        assert_eq!(format!("{}", inverse_indices), "[ 1 0 1]");
        assert_eq!(format!("{}", counts), "[ 1 2]");
    }
}
