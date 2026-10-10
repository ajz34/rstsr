//! Matrix multiplication with dtype promotion for the faer device.

use core::ops::{Add, Mul};
use core::slice::{from_raw_parts, from_raw_parts_mut};
use num::{Complex, Zero};

use super::super::matmul::{matmul_row_major_faer, same_type};
use crate::prelude_dev::*;

/// Write-only matmul with dtype promotion, row-major.
///
/// Same-type operands take the accelerated faer kernel (as in
/// [`matmul_uninit_row_major_faer`]); mixed dtypes cannot, and fall through to
/// the promoting naive kernel.
#[allow(clippy::too_many_arguments)]
fn matmul_uninit_ext_row_major_faer<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    alpha: TC,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
{
    // quick return for empty matrix
    if lc.size() == 0 {
        return Ok(());
    }

    // type check and dispatch
    macro_rules! impl_uninit_dispatch {
        ($ty: ty) => {
            if (same_type::<TA, $ty>() && same_type::<TB, $ty>() && same_type::<TC, $ty>()) {
                // SAFETY: `TypeId` equality above proves TA = TB = TC = $ty; the
                // reinterpreted slices have exactly the original lengths. As in
                // `matmul_uninit_row_major_faer`, only pointer passing happens before
                // the call (`beta = 0`), and the previous (uninitialized) contents of
                // `c` are never read.
                let a_slice = unsafe { from_raw_parts(a.as_ptr() as *const $ty, a.len()) };
                let b_slice = unsafe { from_raw_parts(b.as_ptr() as *const $ty, b.len()) };
                let c_slice = unsafe { from_raw_parts_mut(c.as_mut_ptr() as *mut $ty, c.len()) };
                let alpha = unsafe { *(&alpha as *const TC as *const $ty) };
                let beta = <$ty as Zero>::zero();
                return matmul_row_major_faer::<$ty, $ty, $ty, DA, DB, DC>(
                    c_slice, lc, a_slice, la, b_slice, lb, alpha, beta, pool,
                );
            }
        };
    }

    impl_uninit_dispatch!(f32);
    impl_uninit_dispatch!(f64);
    impl_uninit_dispatch!(Complex<f32>);
    impl_uninit_dispatch!(Complex<f64>);

    // mixed dtypes (or non-faer element types): promote element-wise
    matmul_ext_naive_uninit_cpu_rayon(c, lc, a, la, b, lb, alpha, pool)
}

#[allow(clippy::too_many_arguments)]
impl<TA, TB, TC, DA, DB, DC> DeviceExtMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceFaer
where
    TA: Clone + Send + Sync + 'static,
    TB: Clone + Send + Sync + 'static,
    TC: Clone + Send + Sync + 'static,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: DTypePromoteAPI<TB, Res = TC>,
    // the column-major branch swaps the operands (see the serial impl)
    TB: DTypePromoteAPI<TA, Res = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + PartialEq,
    Self: DeviceRawAPI<MaybeUninit<TC>, Raw = Vec<MaybeUninit<TC>>>,
{
    fn ext_matmul_uninit(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        alpha: TC,
    ) -> Result<()> {
        let default_order = self.default_order();
        let pool = self.get_current_pool();
        match default_order {
            RowMajor => matmul_uninit_ext_row_major_faer(c, lc, a, la, b, lb, alpha, pool),
            ColMajor => {
                let la = la.reverse_axes();
                let lb = lb.reverse_axes();
                let lc = lc.reverse_axes();
                matmul_uninit_ext_row_major_faer(c, &lc, b, &lb, a, &la, alpha, pool)
            },
        }
    }
}

#[cfg(test)]
mod test_ext {
    use super::*;

    #[test]
    fn test_ext_matmul_mixed_dtypes() {
        // u8 @ u16 -> u16. Mixed dtypes cannot use the faer kernel, so this
        // exercises the promoting naive fallback under both default orders.
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceFaer::default();
            device.set_default_order(order);
            let a = tensor_from_nested!([[1u8, 2], [3, 4]], &device);
            let b = tensor_from_nested!([[5u16, 6], [7, 8]], &device);
            let c = ext_matmul(&a, &b);
            assert_eq!(format!("{c}"), "[[ 19 22]\n [ 43 50]]");
        }
    }

    #[test]
    fn test_ext_matmul_same_dtype_matches_matmul() {
        // same-dtype operands take the accelerated kernel inside the ext op,
        // under both default orders (column-major exercises the operand swap)
        for order in [RowMajor, ColMajor] {
            let mut device = DeviceFaer::default();
            device.set_default_order(order);
            let a = tensor_from_nested!([[1.0f64, 2.0], [3.0, 4.0]], &device);
            let b = tensor_from_nested!([[5.0f64, 6.0], [7.0, 8.0]], &device);
            assert_eq!(format!("{}", ext_matmul(&a, &b)), format!("{}", matmul(&a, &b)));
        }
    }
}
