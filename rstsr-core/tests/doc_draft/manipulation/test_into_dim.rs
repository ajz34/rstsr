use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_into_dim {
    use super::*;
    static FUNC: &str = "doc_into_dim";

    #[test]
    fn doc_to_dim() {
        crate::specify_test!("doc_to_dim");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]); // shape: (2, 3), IxD
                                                             // `const_ndim()` returns `None` for a
                                                             // dynamic dimension
        assert_eq!(a.shape().const_ndim(), None);

        // `to_dim::<Ix2>()`: static dimension
        let b = a.to_dim::<Ix2>(); // shape: (2, 3), Ix2
        println!("{b}");
        // [[ 0 1 2]
        //  [ 3 4 5]]
        assert_eq!(b.shape().const_ndim(), Some(2));
        assert_eq!(format!("{b}"), "[[ 0 1 2]\n [ 3 4 5]]");

        // `.to_dyn()`: back to the dynamic dimension
        let c = b.to_dyn(); // shape: (2, 3), IxD
        assert_eq!(c.shape().const_ndim(), None);
    }

    #[test]
    #[should_panic]
    fn doc_to_dim_panic() {
        crate::specify_test!("doc_to_dim_panic");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device)).into_shape([2, 3]); // shape: (2, 3), IxD
        let b = a.to_dim::<Ix3>(); // shape: (2, 3), Ix3, panics
        println!("b: {:?}", b);
    }
}
