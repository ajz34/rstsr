#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_unique_counts {
    use super::*;
    static FUNC: &str = "doc_unique_counts";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: values and their multiplicities
        let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);
        let res = rt::unique_counts(&a);
        println!("{}", res.values);
        // [ 1 2 3]
        println!("{}", res.counts);
        // [ 2 1 2]
        assert_eq!(format!("{}", res.values), "[ 1 2 3]");
        assert_eq!(format!("{}", res.counts), "[ 2 1 2]");

        // # Examples: the result struct converts into a plain tuple
        let (values, counts) = res.into();
        println!("{}", values);
        // [ 1 2 3]
        println!("{}", counts);
        // [ 2 1 2]
        assert_eq!(format!("{}", values), "[ 1 2 3]");
        assert_eq!(format!("{}", counts), "[ 2 1 2]");
    }
}
