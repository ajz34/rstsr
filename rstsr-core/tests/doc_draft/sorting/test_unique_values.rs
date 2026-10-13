#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_unique_values {
    use super::*;
    use num::Complex;
    static FUNC: &str = "doc_unique_values";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: ascending order for an orderable dtype
        let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);
        println!("{}", rt::unique_values(&a));
        // [ 1 2 3]
        assert_eq!(format!("{}", rt::unique_values(&a)), "[ 1 2 3]");

        // # Elaborated examples: first-occurrence order for a non-orderable dtype
        let c = |v: f64| Complex::new(v, 0.0);
        let a = rt::asarray((vec![c(0.), c(1.), c(0.), c(2.), c(0.), c(2.)], &device)).into_shape([2, 3]);
        println!("{}", rt::unique_values(&a));
        // [ 0+0i 1+0i 2+0i]
        assert_eq!(format!("{}", rt::unique_values(&a)), "[ 0+0i 1+0i 2+0i]");

        device.set_default_order(ColMajor);
        let a = rt::asarray((vec![c(0.), c(2.), c(1.), c(0.), c(0.), c(2.)], &device)).into_shape([2, 3]);
        println!("{}", rt::unique_values(&a));
        // [ 0+0i 2+0i 1+0i]
        assert_eq!(format!("{}", rt::unique_values(&a)), "[ 0+0i 2+0i 1+0i]");
    }
}
