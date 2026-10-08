//! doc_draft for `array_index` (array indexing / fancy indexing).

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_array_index {
    use super::*;
    static FUNC: &str = "doc_array_index";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a host list is one index array on the first axis
        let a = rt::arange((12, &device)).into_shape([3, 4]);
        let result = rt::array_index(&a, [2, 0]);
        println!("{result}");
        // [[ 8 9 10 11]
        //  [ 0 1 2 3]]
        assert_eq!(format!("{result}"), "[[ 8 9 10 11]\n [ 0 1 2 3]]");

        // a tuple of index arrays zips them together
        let idx = rt::asarray((vec![0_isize, 2], &device));
        let result = rt::array_index(&a, (&idx, [1, 3]));
        println!("{result}");
        // [ 1 11]
        assert_eq!(format!("{result}"), "[ 1 11]");

        // basic indexers mix with index arrays
        let b = rt::arange((36, &device)).into_shape([4, 3, 3]);
        let result = rt::array_index(&b, (1..3, [0, 1, 2], [0, 2, 1]));
        println!("{result}");
        // [[ 9 14 16]
        //  [18 23 25]]
        assert_eq!(format!("{result}"), "[[ 9 14 16]\n [ 18 23 25]]");

        // placement of the broadcast dimensions
        let x = rt::arange((2 * 3 * 4, &device)).into_shape([2, 3, 4]);
        // the two index arrays are consecutive: the broadcast dimension stays
        // at position 1
        println!("{:?}", rt::array_index(&x, (.., [1, 0], 1)).shape());
        // [2, 2]
        // separated by a slice: it moves to the front
        println!("{:?}", rt::array_index(&x, ([1, 0], .., 1)).shape());
        // [2, 3]
        assert_eq!(rt::array_index(&x, (.., [1, 0], 1)).shape(), &[2, 2]);
        assert_eq!(rt::array_index(&x, ([1, 0], .., 1)).shape(), &[2, 3]);
    }
}
