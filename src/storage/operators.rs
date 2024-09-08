use crate::prelude_dev::*;
use num::Zero;

/* #region op_mutc_refa_refb_func */

#[allow(non_camel_case_types)]
#[allow(clippy::too_many_arguments)]
pub trait DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TC, &TA, &TB),
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
{
    fn op_mutc_refa_refb_func(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<D>,
        a: &Storage<TA, Self>,
        la: &Layout<D>,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
        f: F,
    ) -> Result<()>;
}

macro_rules! impl_op_mutc_refa_refb_func {
    ($DeviceOpAPI:ident, $Op:ident, $op_mutc_refa_refb_func:ident) => {
        pub trait $DeviceOpAPI<TA, TB, TC, D>
        where
            TA: core::ops::$Op<TB, Output = TC>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
        {
            fn $op_mutc_refa_refb_func(
                &self,
                c: &mut Storage<TC, Self>,
                lc: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;
        }
    };
}

#[rustfmt::skip]
mod impl_op_mutc_refa_refb_func {
    use super::*;
    impl_op_mutc_refa_refb_func!(DeviceAddAPI   , Add   , op_mutc_refa_refb_add   );
    impl_op_mutc_refa_refb_func!(DeviceSubAPI   , Sub   , op_mutc_refa_refb_sub   );
    impl_op_mutc_refa_refb_func!(DeviceMulAPI   , Mul   , op_mutc_refa_refb_mul   );
    impl_op_mutc_refa_refb_func!(DeviceDivAPI   , Div   , op_mutc_refa_refb_div   );
    impl_op_mutc_refa_refb_func!(DeviceRemAPI   , Rem   , op_mutc_refa_refb_rem   );
    impl_op_mutc_refa_refb_func!(DeviceBitOrAPI , BitOr , op_mutc_refa_refb_bitor );
    impl_op_mutc_refa_refb_func!(DeviceBitAndAPI, BitAnd, op_mutc_refa_refb_bitand);
    impl_op_mutc_refa_refb_func!(DeviceBitXorAPI, BitXor, op_mutc_refa_refb_bitxor);
    impl_op_mutc_refa_refb_func!(DeviceShlAPI   , Shl   , op_mutc_refa_refb_shl   );
    impl_op_mutc_refa_refb_func!(DeviceShrAPI   , Shr   , op_mutc_refa_refb_shr   );
}
pub use impl_op_mutc_refa_refb_func::*;

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
