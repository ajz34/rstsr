#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_sort_custom {
    use super::*;
    use core::cmp::Ordering;
    use num::Complex;
    static FUNC: &str = "doc_sort_custom";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Examples: complex values sorted by squared magnitude (escape hatch)
        let a = rt::asarray((vec![Complex::new(3.0, 0.0), Complex::new(1.0, 1.0), Complex::new(0.0, 2.0)], &device));
        let by_norm = |x: &Complex<f64>, y: &Complex<f64>| {
            let n1 = x.re * x.re + x.im * x.im;
            let n2 = y.re * y.re + y.im * y.im;
            n1.partial_cmp(&n2).unwrap_or(Ordering::Equal)
        };
        let out = rt::sort_custom(&a, None, by_norm);
        println!("{out}");
        // [ 1+1i 0+2i 3+0i]
        assert_eq!(format!("{out}"), "[ 1+1i 0+2i 3+0i]");

        // # Overload List: one call per distinct behavior
        let a = rt::tensor_from_nested!([5, 1, 4, 2, 3], &device);
        let by_mod3 = |x: &i32, y: &i32| (x % 3).cmp(&(y % 3));
        println!("{}", rt::sort_custom(&a, None, by_mod3)); // defaults
                                                            // [ 3 1 4 5 2]
        assert_eq!(format!("{}", rt::sort_custom(&a, None, by_mod3)), "[ 3 1 4 5 2]");

        let b = rt::tensor_from_nested!([[3, 1, 2], [6, 4, 5]], &device);
        let by_mod2 = |x: &i32, y: &i32| (x % 2).cmp(&(y % 2));
        println!("{}", rt::sort_custom(&b, 0, by_mod2)); // explicit axis
                                                         // [[ 6 4 2]
                                                         //  [ 3 1 5]]
        assert_eq!(format!("{}", rt::sort_custom(&b, 0, by_mod2)), "[[ 6 4 2]\n [ 3 1 5]]");
        println!("{}", rt::sort_custom(&b, (0, true), by_mod2)); // axis + descending
                                                                 // [[ 3 1 5]
                                                                 //  [ 6 4 2]]
        assert_eq!(format!("{}", rt::sort_custom(&b, (0, true), by_mod2)), "[[ 3 1 5]\n [ 6 4 2]]");
    }
}
