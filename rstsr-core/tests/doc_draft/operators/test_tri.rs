//! doc_draft for the triangular pack/unpack methods (`pack_tri` /
//! `unpack_tri` and the `tril` / `triu` convenience forms).

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_tri {
    use super::*;
    static FUNC: &str = "doc_tri";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // pack the lower triangle of a 3x3 matrix into a length-6 vector; the
        // entries are visited in the device default order (RowMajor)
        let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        let packed = a.pack_tril();
        println!("{packed}");
        // [ 1 4 5 7 8 9]
        assert_eq!(format!("{packed}"), "[ 1 4 5 7 8 9]");
        assert_eq!(packed.to_vec(), vec![1.0, 4.0, 5.0, 7.0, 8.0, 9.0]);

        // unpack restores a symmetric 3x3 matrix
        let full = packed.unpack_tril(FlagSymm::Sy);
        println!("{full}");
        // [[ 1 4 7]
        //  [ 4 5 8]
        //  [ 7 8 9]]
        assert_eq!(format!("{full}"), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");

        // the upper-triangle packaging form
        let packed_u = a.pack_triu();
        println!("{packed_u}");
        // [ 1 2 3 5 6 9]
        assert_eq!(format!("{packed_u}"), "[ 1 2 3 5 6 9]");
        println!("{}", packed_u.unpack_triu(FlagSymm::Sy));
        // [[ 1 2 3]
        //  [ 2 5 6]
        //  [ 3 6 9]]
        assert_eq!(format!("{}", packed_u.unpack_triu(FlagSymm::Sy)), "[[ 1 2 3]\n [ 2 5 6]\n [ 3 6 9]]");

        // the general forms take the triangle flag explicitly
        assert_eq!(a.pack_tri(FlagUpLo::L).to_vec(), packed.to_vec());
        assert_eq!(a.pack_tri(FlagUpLo::U).to_vec(), packed_u.to_vec());
        assert_eq!(
            packed.unpack_tri(FlagUpLo::L, FlagSymm::Sy).into_shape([-1]).to_vec(),
            full.into_shape([-1]).to_vec()
        );

        // for ndim >= 3 the packed axis goes to the back under RowMajor ...
        let b = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
        println!("{:?}", b.pack_tril().shape());
        assert_eq!(b.pack_tril().shape(), &[2, 3]);

        // ... and to the front under ColMajor (the same logical tensor)
        device.set_default_order(ColMajor);
        let b = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
        println!("{:?}", b.pack_tril().shape());
        assert_eq!(b.pack_tril().shape(), &[3, 2]);
        device.set_default_order(RowMajor);

        // the `symm` flag derives the opposite triangle from the packed one:
        // `Ay` is anti-symmetric (the opposite entries are negated, the
        // diagonal is zero)
        let p = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);
        println!("{}", p.unpack_tri(FlagUpLo::L, FlagSymm::Ay));
        // [[ 0 -4 -7]
        //  [ 4 0 -8]
        //  [ 7 8 0]]
        assert_eq!(format!("{}", p.unpack_tri(FlagUpLo::L, FlagSymm::Ay)), "[[ 0 -4 -7]\n [ 4 0 -8]\n [ 7 8 0]]");
    }
}
