//! doc_draft for the triangular pack/unpack methods: the `pack_tri` and
//! `unpack_tri` docstring examples only (behavioral `uplo`/`symm` cases live in
//! `core_func::operators::test_tri`).

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

        // pack_tri: pack the lower triangle into a length-6 vector, entries
        // visited in the device default order
        let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
        println!("{}", a.pack_tri(FlagUpLo::L));
        // [ 1 4 5 7 8 9]
        assert_eq!(format!("{}", a.pack_tri(FlagUpLo::L)), "[ 1 4 5 7 8 9]");

        // pack_tri elaborated: for rank >= 3 the packed axis goes to the back
        // under RowMajor ...
        let b = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
        println!("{:?}", b.pack_tril().shape());
        // [2, 3]
        assert_eq!(b.pack_tril().shape(), &[2, 3]);

        // ... and to the front under ColMajor (the same logical tensor)
        device.set_default_order(ColMajor);
        let b = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
        println!("{:?}", b.pack_tril().shape());
        // [3, 2]
        assert_eq!(b.pack_tril().shape(), &[3, 2]);
        device.set_default_order(RowMajor);

        // unpack_tri (Sy): restore a symmetric matrix
        let packed = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);
        println!("{}", packed.unpack_tri(FlagUpLo::L, FlagSymm::Sy));
        // [[ 1 4 7]
        //  [ 4 5 8]
        //  [ 7 8 9]]
        assert_eq!(format!("{}", packed.unpack_tri(FlagUpLo::L, FlagSymm::Sy)), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");

        // unpack_tri (N): no symmetry assumed, the opposite triangle is zero-filled
        println!("{}", packed.unpack_tri(FlagUpLo::L, FlagSymm::N));
        // [[ 1 0 0]
        //  [ 4 5 0]
        //  [ 7 8 9]]
        assert_eq!(format!("{}", packed.unpack_tri(FlagUpLo::L, FlagSymm::N)), "[[ 1 0 0]\n [ 4 5 0]\n [ 7 8 9]]");
    }
}
