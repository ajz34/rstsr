use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_cross {
    use super::*;
    static FUNC: &str = "doc_cross";

    #[test]
    fn test_cross() {
        crate::specify_test!("test_cross");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // cross of two 3-vectors
        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        let b = rt::tensor_from_nested!([4, 5, 6], &device);
        let c = rt::cross(&a, &b, -1);
        println!("{c}");
        // [-3  6 -3]
        assert_eq!(c.to_vec(), vec![-3, 6, -3]);

        // explicit axis, with the operands broadcast on the other axes
        let a2 = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6]], &device);
        let c2 = rt::cross(&a2, &b, 1);
        assert_eq!(c2.shape(), &[2, 3]);
    }
}
