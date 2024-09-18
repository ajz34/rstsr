use crate::prelude_dev::*;

macro_rules! trait_op_assign_api {
    ($DeviceOpAPI:ident, $Op:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TA: $Op<TB>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn op_muta_refb(
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
    use core::ops::*;
    trait_op_assign_api!(DeviceAddAssignAPI   , AddAssign   );
    trait_op_assign_api!(DeviceSubAssignAPI   , SubAssign   );
    trait_op_assign_api!(DeviceMulAssignAPI   , MulAssign   );
    trait_op_assign_api!(DeviceDivAssignAPI   , DivAssign   );
    trait_op_assign_api!(DeviceRemAssignAPI   , RemAssign   );
    trait_op_assign_api!(DeviceBitOrAssignAPI , BitOrAssign );
    trait_op_assign_api!(DeviceBitAndAssignAPI, BitAndAssign);
    trait_op_assign_api!(DeviceBitXorAssignAPI, BitXorAssign);
    trait_op_assign_api!(DeviceShlAssignAPI   , ShlAssign   );
    trait_op_assign_api!(DeviceShrAssignAPI   , ShrAssign   );
}
pub use impl_op_muta_refb_func::*;

macro_rules! trait_op_unary_api {
    ($DeviceOpAPI:ident, $Op:ident, $op_muta_refb_func:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TB: $Op<Output = TA>,
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

mod impl_op_unary_api {
    use super::*;
    use core::ops::*;
    trait_op_unary_api!(DeviceNegAPI, Neg, op_muta_refb_neg);
    trait_op_unary_api!(DeviceNotAPI, Not, op_muta_refb_not);
}
pub use impl_op_unary_api::*;
