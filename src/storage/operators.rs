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

/* #region op_muta_refb_func */

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

/* #endregion */

pub trait OpSumAPI<T, D>
where
    T: Zero + core::ops::Add<Output = T>,
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T>;
}
