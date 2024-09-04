use crate::prelude_dev::*;
use num::Zero;

/* #region ternary-op */

macro_rules! trait_op_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_ternary:ident) => {
        pub trait $DeviceOpAPI<T, TB, D>
        where
            T: core::ops::$Op<TB, Output = T>,
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

trait_op_api!(DeviceAddAPI, Add, add_ternary);
trait_op_api!(DeviceSubAPI, Sub, sub_ternary);
trait_op_api!(DeviceMulAPI, Mul, mul_ternary);
trait_op_api!(DeviceDivAPI, Div, div_ternary);
trait_op_api!(DeviceRemAPI, Rem, rem_ternary);
trait_op_api!(DeviceBitOrAPI, BitOr, bitor_ternary);
trait_op_api!(DeviceBitAndAPI, BitAnd, bitand_ternary);
trait_op_api!(DeviceBitXorAPI, BitXor, bitxor_ternary);
trait_op_api!(DeviceShlAPI, Shl, shl_ternary);
trait_op_api!(DeviceShrAPI, Shr, shr_ternary);

/* #endregion */

/* #region binary-op */

macro_rules! trait_op_assign_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_binary:ident) => {
        pub trait $DeviceOpAPI<T, TB, D>
        where
            T: core::ops::$Op<TB>,
            D: DimAPI,
            Self: DeviceAPI<T> + DeviceAPI<TB>,
        {
            fn $op_binary(
                &self,
                c: &mut Storage<T, Self>,
                lc: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;
        }
    };
}

trait_op_assign_api!(DeviceAddAssignAPI, AddAssign, add_assign_binary);
trait_op_assign_api!(DeviceSubAssignAPI, SubAssign, sub_assign_binary);
trait_op_assign_api!(DeviceMulAssignAPI, MulAssign, mul_assign_binary);
trait_op_assign_api!(DeviceDivAssignAPI, DivAssign, div_assign_binary);
trait_op_assign_api!(DeviceRemAssignAPI, RemAssign, rem_assign_binary);
trait_op_assign_api!(DeviceBitOrAssignAPI, BitOrAssign, bitor_assign_binary);
trait_op_assign_api!(DeviceBitAndAssignAPI, BitAndAssign, bitand_assign_binary);
trait_op_assign_api!(DeviceBitXorAssignAPI, BitXorAssign, bitxor_assign_binary);
trait_op_assign_api!(DeviceShlAssignAPI, ShlAssign, shl_assign_binary);
trait_op_assign_api!(DeviceShrAssignAPI, ShrAssign, shr_assign_binary);

/* #endregion */

pub trait OpSumAPI<T, D>
where
    T: Zero + core::ops::Add<Output = T>,
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
