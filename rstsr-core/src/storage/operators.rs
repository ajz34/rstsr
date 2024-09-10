use crate::prelude_dev::*;
use num::Zero;

/* #region op_func */

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

#[allow(non_camel_case_types)]
#[allow(clippy::too_many_arguments)]
pub trait DeviceOp_MutA_RefB_API<TA, TB, D, F>
where
    D: DimAPI,
    F: FnMut(&mut TA, &TB),
    Self: DeviceAPI<TA> + DeviceAPI<TB>,
{
    fn op_muta_refb_func(
        &self,
        a: &mut Storage<TA, Self>,
        la: &Layout<D>,
        b: &Storage<TB, Self>,
        lb: &Layout<D>,
        f: F,
    ) -> Result<()>;
}

#[allow(non_camel_case_types)]
#[allow(clippy::too_many_arguments)]
pub trait DeviceOp_MutA_API<T, D, F>
where
    D: DimAPI,
    F: FnMut(&mut T),
    Self: DeviceAPI<T>,
{
    fn op_muta_func(&self, a: &mut Storage<T, Self>, la: &Layout<D>, f: F) -> Result<()>;
}

/* #endregion */

/* #region op_mutc_refa_refb_operator */

macro_rules! impl_op_mutc_refa_refb_operator {
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
mod impl_op_mutc_refa_refb_operator {
    use super::*;
    impl_op_mutc_refa_refb_operator!(DeviceAddAPI   , Add   , op_mutc_refa_refb_add   );
    impl_op_mutc_refa_refb_operator!(DeviceSubAPI   , Sub   , op_mutc_refa_refb_sub   );
    impl_op_mutc_refa_refb_operator!(DeviceMulAPI   , Mul   , op_mutc_refa_refb_mul   );
    impl_op_mutc_refa_refb_operator!(DeviceDivAPI   , Div   , op_mutc_refa_refb_div   );
    impl_op_mutc_refa_refb_operator!(DeviceRemAPI   , Rem   , op_mutc_refa_refb_rem   );
    impl_op_mutc_refa_refb_operator!(DeviceBitOrAPI , BitOr , op_mutc_refa_refb_bitor );
    impl_op_mutc_refa_refb_operator!(DeviceBitAndAPI, BitAnd, op_mutc_refa_refb_bitand);
    impl_op_mutc_refa_refb_operator!(DeviceBitXorAPI, BitXor, op_mutc_refa_refb_bitxor);
    impl_op_mutc_refa_refb_operator!(DeviceShlAPI   , Shl   , op_mutc_refa_refb_shl   );
    impl_op_mutc_refa_refb_operator!(DeviceShrAPI   , Shr   , op_mutc_refa_refb_shr   );
}
pub use impl_op_mutc_refa_refb_operator::*;

/* #endregion */

/* #region op_muta_refb_operator */

macro_rules! trait_op_assign_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_muta_refb_func:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TA: core::ops::$Op<TB>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn $op_muta_refb_func(
                &self,
                a: &mut Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_func {
    use super::*;
    trait_op_assign_api!(DeviceAddAssignAPI   , AddAssign   , op_muta_refb_add_assign   );
    trait_op_assign_api!(DeviceSubAssignAPI   , SubAssign   , op_muta_refb_sub_assign   );
    trait_op_assign_api!(DeviceMulAssignAPI   , MulAssign   , op_muta_refb_mul_assign   );
    trait_op_assign_api!(DeviceDivAssignAPI   , DivAssign   , op_muta_refb_div_assign   );
    trait_op_assign_api!(DeviceRemAssignAPI   , RemAssign   , op_muta_refb_rem_assign   );
    trait_op_assign_api!(DeviceBitOrAssignAPI , BitOrAssign , op_muta_refb_bitor_assign );
    trait_op_assign_api!(DeviceBitAndAssignAPI, BitAndAssign, op_muta_refb_bitand_assign);
    trait_op_assign_api!(DeviceBitXorAssignAPI, BitXorAssign, op_muta_refb_bitxor_assign);
    trait_op_assign_api!(DeviceShlAssignAPI   , ShlAssign   , op_muta_refb_shl_assign   );
    trait_op_assign_api!(DeviceShrAssignAPI   , ShrAssign   , op_muta_refb_shr_assign   );
}
pub use impl_op_muta_refb_func::*;

macro_rules! trait_op_unary_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_muta_refb_func:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TB: core::ops::$Op<Output = TA>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn $op_muta_refb_func(
                &self,
                a: &mut Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;
        }
    };
}

trait_op_unary_api!(DeviceNegAPI, Neg, op_muta_refb_neg);
trait_op_unary_api!(DeviceNotAPI, Not, op_muta_refb_not);

/* #endregion */

pub trait OpSumAPI<T, D>
where
    T: Zero + core::ops::Add<Output = T>,
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
