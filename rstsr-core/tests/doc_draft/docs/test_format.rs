//! Doc-draft twin of the `format` module page (`src/format/mod.rs`).
//!
//! Every example displayed on that page must originate from this test; shown
//! output is pasted from the actual run and pinned by string assertions.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_format {
    use super::*;
    static FUNC: &str = "doc_format";

    #[test]
    fn display_truncation_exp() {
        crate::specify_test!("display_truncation_exp");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        println!("{a}");
        assert_eq!(format!("{a}"), "[[ 0 1 2]\n [ 3 4 5]]");

        let b = rt::asarray((vec![0.001, 1.0, 1000.0], &device));
        println!("{:e}", b);
        assert_eq!(format!("{:e}", b), "[ 1e-3 1e0 1e3]");

        let c = rt::arange((100, &device));
        println!("{c}");
        assert_eq!(format!("{c}"), "[ 0 1 2 ... 97 98 99]");

        // `Debug` renders through the same device implementation.
        let dbg = format!("{a:?}");
        assert!(dbg.contains("=== Debug Tensor Print ==="));
        assert!(dbg.contains("[[ 0 1 2]\n [ 3 4 5]]"));
    }
}
