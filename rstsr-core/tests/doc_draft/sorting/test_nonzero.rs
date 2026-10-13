#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_nonzero {
    use super::*;
    static FUNC: &str = "doc_nonzero";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: row-major coordinate sequence
        let a = rt::tensor_from_nested!([[1, 0, 2], [0, 3, 0]], &device);
        let coords = rt::nonzero(&a);
        println!("{}", coords[0]);
        // [ 0 0 1]
        println!("{}", coords[1]);
        // [ 0 2 1]
        assert_eq!(format!("{}", coords[0]), "[ 0 0 1]");
        assert_eq!(format!("{}", coords[1]), "[ 0 2 1]");

        // # Elaborated examples: column-major visit order
        device.set_default_order(ColMajor);
        let a = rt::tensor_from_nested!([[1, 0, 2], [0, 3, 0]], &device);
        let coords = rt::nonzero(&a);
        println!("{}", coords[0]);
        // [ 0 1 0]
        println!("{}", coords[1]);
        // [ 0 1 2]
        assert_eq!(format!("{}", coords[0]), "[ 0 1 0]");
        assert_eq!(format!("{}", coords[1]), "[ 0 1 2]");
    }
}
