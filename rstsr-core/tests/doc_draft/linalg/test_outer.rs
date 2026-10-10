use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_outer {
    use super::*;
    static FUNC: &str = "doc_outer";

    #[test]
    fn test_outer() {
        crate::specify_test!("test_outer");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // outer product of two vectors
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5], &device);
        let c = rt::outer(&a, &b);
        println!("{c}");
        // [[ 4 5]
        //  [ 8 10]
        //  [ 12 15]]
        assert_eq!(format!("{c}"), "[[ 4 5]\n [ 8 10]\n [ 12 15]]");

        // the array-API definition: outer(x1, x2) == x1[:, None] * x2[None, :]
        let a2 = rt::tensor_from_nested!([[1], [2], [3]], &device);
        let b2 = rt::tensor_from_nested!([[4, 5]], &device);
        assert_eq!(format!("{}", rt::outer(&a, &b)), format!("{}", rt::mul(&a2, &b2)));
    }
}
