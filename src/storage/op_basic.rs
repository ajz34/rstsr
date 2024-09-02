use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};
use num::Zero;

use crate::prelude_dev::*;

pub trait OpAssignAPI<T, DC, DA>
where
    DC: DimAPI,
    DA: DimAPI,
    Self: DeviceAPI<T>,
{
    fn assign_arbitary_layout(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()>;
}

/* #region basic binary operations involving ternary tensors */

macro_rules! impl_op_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_ternary:ident) => {
        pub trait $DeviceOpAPI<T, TB, D>
        where
            T: $Op<TB, Output = T> + Clone,
            D: DimAPI,
            Self: DeviceAPI<T> + DeviceAPI<TB>,
        {
            fn $op_ternary(
                &self,
                c: &mut Storage<T, Self>,
                lc: &Layout<D>,
                a: &Storage<T, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;
        }
    };
}

impl_op_api!(DeviceAddAPI, Add, add_ternary);
impl_op_api!(DeviceSubAPI, Sub, sub_ternary);
impl_op_api!(DeviceMulAPI, Mul, mul_ternary);
impl_op_api!(DeviceDivAPI, Div, div_ternary);
impl_op_api!(DeviceRemAPI, Rem, rem_ternary);
impl_op_api!(DeviceBitOrAPI, BitOr, bitor_ternary);
impl_op_api!(DeviceBitAndAPI, BitAnd, bitand_ternary);
impl_op_api!(DeviceBitXorAPI, BitXor, bitxor_ternary);
impl_op_api!(DeviceShlAPI, Shl, shl_ternary);
impl_op_api!(DeviceShrAPI, Shr, shr_ternary);

/* #endregion */

pub trait OpSumAPI<T, D>
where
    T: Zero + Add<Output = T> + Clone,
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
