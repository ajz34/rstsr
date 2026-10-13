#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_unique_inverse {
    use super::*;
    static FUNC: &str = "doc_unique_inverse";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: inverse mapping back to the unique entries
        let a = rt::tensor_from_nested!([3, 1, 3], &device);
        let res = rt::unique_inverse(&a);
        println!("{}", res.values);
        // [ 1 3]
        println!("{}", res.inverse_indices);
        // [ 1 0 1]
        assert_eq!(format!("{}", res.values), "[ 1 3]");
        assert_eq!(format!("{}", res.inverse_indices), "[ 1 0 1]");

        // # Examples: the result struct converts into a plain tuple
        let (values, inverse_indices) = res.into();
        assert_eq!(format!("{}", values), "[ 1 3]");
        assert_eq!(format!("{}", inverse_indices), "[ 1 0 1]");
    }
}
