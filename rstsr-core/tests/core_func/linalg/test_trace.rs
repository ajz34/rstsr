#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod numpy_trace {
    use super::*;
    static FUNC: &str = "numpy_trace";

    #[test]
    fn test_trace() {
        // numpy: v2.5.2 | _core/tests/test_numeric.py::TestNonarrayArgs::test_trace (L349)
        crate::specify_test!("test_trace");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // c = [[1, 2], [3, 4], [5, 6]]
        // assert_equal(np.trace(c), 5)
        // (a 3x2 input: the diagonal of the last two axes is [1, 4])
        let c = rt::tensor_from_nested!([[1, 2], [3, 4], [5, 6]], &device);
        let actual = rt::trace(&c, ());
        assert_eq!(actual.ndim(), 0);
        assert_eq!(actual.to_scalar(), 5);
    }
}

#[cfg(test)]
mod custom_trace {
    use super::*;
    static FUNC: &str = "custom_trace";

    #[test]
    fn test_trace_offsets() {
        crate::specify_test!("test_trace_offsets");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);
        // main diagonal, super-diagonal, sub-diagonal, and an out-of-range offset
        assert_eq!(rt::trace(&a, ()).to_scalar(), 15); // 1 + 5 + 9
        assert_eq!(rt::trace(&a, 1).to_scalar(), 8); // 2 + 6
        assert_eq!(rt::trace(&a, -1).to_scalar(), 12); // 4 + 8
        assert_eq!(rt::trace(&a, 5).to_scalar(), 0); // empty diagonal
    }

    #[test]
    fn test_trace_stack() {
        crate::specify_test!("test_trace_stack");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // a stacked input: each matrix's diagonal is summed over the last two axes
        let a = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        let actual = rt::trace(&a, ());
        let expected = rt::tensor_from_nested!([5, 13], &device);
        assert_equal(&actual, &expected, None);
        assert_eq!(actual.shape(), &[2]);
    }

    #[test]
    fn test_trace_requires_2d() {
        crate::specify_test!("test_trace_requires_2d");

        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let a = rt::tensor_from_nested!([1, 2, 3], &device);
        assert!(rt::trace_f(&a, ()).is_err());
    }
}
