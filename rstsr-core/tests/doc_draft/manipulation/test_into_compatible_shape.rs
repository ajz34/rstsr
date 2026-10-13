use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_into_compatible_shape {
    use super::*;
    static FUNC: &str = "doc_into_compatible_shape";

    #[test]
    fn compatible_reshape() {
        crate::specify_test!("compatible_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // shape: (4, 6), stride: (6, 1), c-contiguous
        let a = rt::arange((24, &device)).into_shape([4, 6]);
        // Split a dimension: (4, 6) -> (2, 2, 6) - layout compatible
        let b = a.into_compatible_shape([2, 2, 6], RowMajor);
        assert_eq!(b.shape(), &[2, 2, 6]);
    }

    #[test]
    fn incompatible_reshape() {
        crate::specify_test!("incompatible_reshape");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
        let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
        // layout compatible
        assert!(a.to_compatible_shape_f([4, 6 * 9], RowMajor).is_ok());
        // layout incompatible
        assert!(a.to_compatible_shape_f([4 * 6, 9], RowMajor).is_err());
    }

    #[test]
    fn overload_list() {
        // # Overload List snippet: shape (integer vs list) and order (explicit vs None)
        crate::specify_test!("overload_list");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::arange((6, &device));
        println!("{:?}", a.to_compatible_shape([2, 3], RowMajor).shape()); // list shape, explicit
                                                                           // order
                                                                           // [2, 3]
        assert_eq!(a.to_compatible_shape([2, 3], RowMajor).shape(), &[2, 3]);
        assert_eq!(format!("{:?}", a.to_compatible_shape([2, 3], RowMajor).shape()), "[2, 3]");

        println!("{:?}", a.to_compatible_shape(6, None).shape()); // integer shape, order = None
                                                                  // [6]
        assert_eq!(a.to_compatible_shape(6, None).shape(), &[6]);
        assert_eq!(format!("{:?}", a.to_compatible_shape(6, None).shape()), "[6]");
    }
}
