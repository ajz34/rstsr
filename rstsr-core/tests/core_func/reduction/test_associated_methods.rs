//! Associated-method forms of the custom-reduction and closeness families.
//!
//! `reduce_all` / `reduce_axes` / `reduce_with_args` / `allclose` / `allclose_all`
//! are also available as methods on `TensorAny`; the methods share the free
//! function's signature from the second parameter onward.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod associated_methods {
    use super::*;
    static FUNC: &str = "associated_methods";

    #[test]
    fn test_reduce_methods() {
        crate::specify_test!("test_reduce_methods");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1_i64, 2, 3], [4, 5, 6]], &device);
        let fold = (|| 0_i64, |acc: i64, x: i64| acc + x, |x: i64, y: i64| x + y, |acc: i64| acc);

        // method forms equal the free functions (and the built-in sum)
        assert_eq!(a.reduce_all(fold.0, fold.1, fold.2, fold.3), rt::sum(&a));
        assert_eq!(a.reduce_all(fold.0, fold.1, fold.2, fold.3), rt::reduce_all(&a, fold.0, fold.1, fold.2, fold.3));
        assert_equal(a.reduce_axes(1, fold.0, fold.1, fold.2, fold.3), rt::sum_axes(&a, 1), None);
        assert_equal(a.reduce_with_args(1, fold.0, fold.1, fold.2, fold.3), rt::sum_axes(&a, 1), None);
        assert!(a.reduce_all_f(fold.0, fold.1, fold.2, fold.3).is_ok());
        assert!(a.reduce_axes_f(1, fold.0, fold.1, fold.2, fold.3).is_ok());
        assert!(a.reduce_with_args_f(1, fold.0, fold.1, fold.2, fold.3).is_ok());
    }

    #[test]
    fn test_allclose_methods() {
        crate::specify_test!("test_allclose_methods");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1.0_f64, 2.0, 3.0], &device);
        let b = rt::tensor_from_nested!([1.0_f64, 2.0, 3.0 + 1e-12], &device);
        let c = rt::tensor_from_nested!([1.0_f64, 2.0, 4.0], &device);

        assert!(a.allclose(&b, None));
        assert_eq!(a.allclose(&b, None), rt::allclose(&a, &b, None));
        assert!(!a.allclose(&c, None));
        assert_eq!(a.allclose_all(&b, None), rt::allclose_all(&a, &b, None));
        assert!(a.allclose_f(&b, None).is_ok());
        assert!(a.allclose_all_f(&b, None).is_ok());
        // the second tensor may be a view (the methods take `impl TensorViewAPI`)
        let v = b.view();
        assert!(a.allclose(v, None));
    }
}
