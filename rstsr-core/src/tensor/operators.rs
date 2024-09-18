use crate::prelude_dev::*;

/* #region op_func */

pub fn op_mutc_refa_refb_func<RA, RB, RC, DA, DB, DC, TA, TB, TC, B, F>(
    c: &mut TensorBase<RC, DC>,
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    f: F,
) -> Result<()>
where
    // lifetime and data constraints
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    RC: DataMutAPI<Data = Storage<TC, B>>,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    // broadcast constraints
    DC: DimMaxAPI<DA> + DimMaxAPI<DB>,
    <DC as DimMaxAPI<DA>>::Max: DimConvertAPI<DC>,
    <DC as DimMaxAPI<DB>>::Max: DimConvertAPI<DC>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, DC, F>,
    F: FnMut(&mut TC, &TA, &TB),
{
    rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
    let lc = c.layout();
    let la = a.layout();
    let lb = b.layout();
    // all layouts should be broadcastable to lc
    // we can first generate broadcasted shape, then check this
    let (lc_b, la_b) = broadcast_layout_to_first(lc, la)?;
    rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
    let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb)?;
    rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
    // op provided by device
    let device = c.device().clone();
    let storage_c = c.data_mut().storage_mut();
    let storage_a = a.data().storage();
    let storage_b = b.data().storage();
    device.op_mutc_refa_refb_func(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b, f)
}

pub fn op_refa_refb_func<RA, RB, DA, DB, TA, TB, TC, B, F>(
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    f: F,
) -> Result<Tensor<TC, <DA as DimMaxAPI<DB>>::Max, B>>
where
    // lifetime and data constraints
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB>,
    // broadcast constraints
    DA: DimMaxAPI<DB>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, <DA as DimMaxAPI<DB>>::Max, F>,
    B: DeviceCreationAnyAPI<TC>,
    F: FnMut(&mut TC, &TA, &TB),
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let la = a.layout();
    let lb = b.layout();
    let (la_b, lb_b) = broadcast_layout(la, lb)?;
    // generate output layout
    let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::K)?;
    let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::K)?;
    let lc = if lc_from_a == lc_from_b {
        lc_from_a
    } else {
        match TensorOrder::default() {
            TensorOrder::C => la_b.shape().c(),
            TensorOrder::F => la_b.shape().f(),
        }
    };
    // generate empty c
    let device = a.device();
    let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
    // add provided by device
    let storage_a = a.data().storage();
    let storage_b = b.data().storage();
    device.op_mutc_refa_refb_func(&mut storage_c, &lc, storage_a, &la_b, storage_b, &lb_b, f)?;
    // return tensor
    Tensor::new(DataOwned::from(storage_c), lc)
}

pub fn op_muta_refb_func<RA, RB, DA, DB, TA, TB, B, F>(
    a: &mut TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    f: F,
) -> Result<()>
where
    // lifetime and data constraints
    RA: DataMutAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB>,
    // broadcast constraints
    DA: DimMaxAPI<DB>,
    <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
    // operation constraints
    B: DeviceOp_MutA_RefB_API<TA, TB, DA, F>,
    F: FnMut(&mut TA, &TB),
{
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    let la = a.layout();
    let lb = b.layout();
    // all layouts should be broadcastable to lc
    // we can first generate broadcasted shape, then check this
    let (la_b, lb_b) = broadcast_layout_to_first(la, lb)?;
    rstsr_assert_eq!(la_b, *la, InvalidLayout)?;
    // op provided by device
    let device = a.device().clone();
    let storage_a = a.data_mut().storage_mut();
    let storage_b = b.data().storage();
    device.op_muta_refb_func(storage_a, &la_b, storage_b, &lb_b, f)
}

pub fn op_muta_func<R, T, D, B, F>(a: &mut TensorBase<R, D>, f: F) -> Result<()>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
    B: DeviceOp_MutA_API<T, D, F>,
    F: FnMut(&mut T),
{
    let la = a.layout().clone();
    let device = a.device().clone();
    let storage_a = a.data_mut().storage_mut();
    device.op_muta_func(storage_a, &la, f)
}

/* #endregion */

/* #region op_mutc_refa_refb_operation */

macro_rules! impl_op_mutc_refa_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident, $op:ident, $op_mutc_refa_refb_func:ident, $op_refa_refb_func:ident) => {
        pub fn $op_mutc_refa_refb_func<RA, RB, RC, DA, DB, DC, TA, TB, TC, B>(
            c: &mut TensorBase<RC, DC>,
            a: &TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<()>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            RC: DataMutAPI<Data = Storage<TC, B>>,
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
            // broadcast constraints
            DC: DimMaxAPI<DA> + DimMaxAPI<DB>,
            <DC as DimMaxAPI<DA>>::Max: DimConvertAPI<DC>,
            <DC as DimMaxAPI<DB>>::Max: DimConvertAPI<DC>,
            // operation constraints
            TA: $Op<TB, Output = TC>,
            B: $DeviceOpAPI<TA, TB, TC, DC>,
        {
            rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
            rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
            let lc = c.layout();
            let la = a.layout();
            let lb = b.layout();
            // all layouts should be broadcastable to lc
            // we can first generate broadcasted shape, then check this
            let (lc_b, la_b) = broadcast_layout_to_first(lc, la)?;
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb)?;
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            // op provided by device
            let device = c.device().clone();
            let storage_c = c.data_mut().storage_mut();
            let storage_a = a.data().storage();
            let storage_b = b.data().storage();
            device.op_mutc_refa_refb(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b)
        }

        pub fn $op_refa_refb_func<RA, RB, DA, DB, TA, TB, B>(
            a: &TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<Tensor<<TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as $Op<TB>>::Output>,
        {
            rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
            let la = a.layout();
            let lb = b.layout();
            let (la_b, lb_b) = broadcast_layout(la, lb)?;
            // generate output layout
            let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::K)?;
            let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::K)?;
            let lc = if lc_from_a == lc_from_b {
                lc_from_a
            } else {
                match TensorOrder::default() {
                    TensorOrder::C => la_b.shape().c(),
                    TensorOrder::F => la_b.shape().f(),
                }
            };
            // generate empty c
            let device = a.device();
            let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
            // op provided by device
            let storage_a = a.data().storage();
            let storage_b = b.data().storage();
            device.op_mutc_refa_refb(&mut storage_c, &lc, storage_a, &la_b, storage_b, &lb_b)?;
            // return tensor
            Tensor::new(DataOwned::from(storage_c), lc)
        }

        impl<RA, RB, DA, DB, TA, TB, B> $Op<&TensorBase<RB, DB>> for &TensorBase<RA, DA>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as $Op<TB>>::Output>,
        {
            type Output = Tensor<<TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>;

            fn $op(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
                $op_refa_refb_func(self, rhs).unwrap()
            }
        }

        impl<'a, RB, DA, DB, TA, TB, B> $Op<&TensorBase<RB, DB>> for TensorView<'a, TA, DA, B>
        where
            // lifetime and data constraints
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as $Op<TB>>::Output>,
        {
            type Output = Tensor<<TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>;

            fn $op(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
                $op_refa_refb_func(&self.view(), rhs).unwrap()
            }
        }

        impl<'b, RA, DA, DB, TA, TB, B> $Op<TensorView<'b, TB, DB, B>> for &TensorBase<RA, DA>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as $Op<TB>>::Output>,
        {
            type Output = Tensor<<TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>;

            fn $op(self, rhs: TensorView<'b, TB, DB, B>) -> Self::Output {
                $op_refa_refb_func(self, &rhs.view()).unwrap()
            }
        }

        impl<'a, 'b, DA, DB, TA, TB, B> $Op<TensorView<'b, TB, DB, B>> for TensorView<'a, TA, DA, B>
        where
            // lifetime and data constraints
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as $Op<TB>>::Output>,
        {
            type Output = Tensor<<TA as $Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>;

            fn $op(self, rhs: TensorView<'b, TB, DB, B>) -> Self::Output {
                $op_refa_refb_func(&self.view(), &rhs.view()).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_mutc_refa_refb_operator {
    use super::*;
    use core::ops::*;
    impl_op_mutc_refa_refb_operator!(DeviceAddAPI   , Add   , add   , op_mut_refa_refb_add   , op_refa_refb_add   );
    impl_op_mutc_refa_refb_operator!(DeviceSubAPI   , Sub   , sub   , op_mut_refa_refb_sub   , op_refa_refb_sub   );
    impl_op_mutc_refa_refb_operator!(DeviceMulAPI   , Mul   , mul   , op_mut_refa_refb_mul   , op_refa_refb_mul   );
    impl_op_mutc_refa_refb_operator!(DeviceDivAPI   , Div   , div   , op_mut_refa_refb_div   , op_refa_refb_div   );
//  impl_op_mutc_refa_refb_operator!(DeviceRemAPI   , Rem   , rem   , op_mut_refa_refb_rem   , op_refa_refb_rem   );
    impl_op_mutc_refa_refb_operator!(DeviceBitOrAPI , BitOr , bitor , op_mut_refa_refb_bitor , op_refa_refb_bitor );
    impl_op_mutc_refa_refb_operator!(DeviceBitAndAPI, BitAnd, bitand, op_mut_refa_refb_bitand, op_refa_refb_bitand);
    impl_op_mutc_refa_refb_operator!(DeviceBitXorAPI, BitXor, bitxor, op_mut_refa_refb_bitxor, op_refa_refb_bitxor);
    impl_op_mutc_refa_refb_operator!(DeviceShlAPI   , Shl   , shl   , op_mut_refa_refb_shl   , op_refa_refb_shl   );
    impl_op_mutc_refa_refb_operator!(DeviceShrAPI   , Shr   , shr   , op_mut_refa_refb_shr   , op_refa_refb_shr   );
}
pub use impl_op_mutc_refa_refb_operator::*;

/* #endregion */

/* #region op_muta_refb_operation */

macro_rules! impl_op_muta_refb_operator {
    ($DeviceOpAPI:ident, $Op:ident, $op:ident, $op_muta_refb_func:ident) => {
        pub fn $op_muta_refb_func<RA, RB, DA, DB, TA, TB, B>(
            a: &mut TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<()>
        where
            // lifetime and data constraints
            RA: DataMutAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, DA>,
        {
            rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
            let la = a.layout();
            let lb = b.layout();
            // all layouts should be broadcastable to lc
            // we can first generate broadcasted shape, then check this
            let (la_b, lb_b) = broadcast_layout_to_first(la, lb)?;
            rstsr_assert_eq!(la_b, *la, InvalidLayout)?;
            // op provided by device
            let device = a.device().clone();
            let storage_a = a.data_mut().storage_mut();
            let storage_b = b.data().storage();
            device.op_muta_refb(storage_a, &la_b, storage_b, &lb_b)
        }

        impl<RA, RB, DA, DB, TA, TB, B> $Op<&TensorBase<RB, DB>> for TensorBase<RA, DA>
        where
            // lifetime and data constraints
            RA: DataMutAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, DA>,
        {
            fn $op(&mut self, rhs: &TensorBase<RB, DB>) {
                $op_muta_refb_func(self, rhs).unwrap()
            }
        }

        impl<RA, RB, DA, DB, TA, TB, B> $Op<TensorBase<RB, DB>> for TensorBase<RA, DA>
        where
            // lifetime and data constraints
            RA: DataMutAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            // operation constraints
            TA: $Op<TB>,
            B: $DeviceOpAPI<TA, TB, DA>,
        {
            fn $op(&mut self, rhs: TensorBase<RB, DB>) {
                $op_muta_refb_func(self, &rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_operator {
    use super::*;
    use core::ops::*;
    impl_op_muta_refb_operator!(DeviceAddAssignAPI   , AddAssign   , add_assign   , op_muta_refb_add_assign   );
    impl_op_muta_refb_operator!(DeviceSubAssignAPI   , SubAssign   , sub_assign   , op_muta_refb_sub_assign   );
    impl_op_muta_refb_operator!(DeviceMulAssignAPI   , MulAssign   , mul_assign   , op_muta_refb_mul_assign   );
    impl_op_muta_refb_operator!(DeviceDivAssignAPI   , DivAssign   , div_assign   , op_muta_refb_div_assign   );
    impl_op_muta_refb_operator!(DeviceRemAssignAPI   , RemAssign   , rem_assign   , op_muta_refb_rem_assign   );
    impl_op_muta_refb_operator!(DeviceBitOrAssignAPI , BitOrAssign , bitor_assign , op_muta_refb_bitor_assign );
    impl_op_muta_refb_operator!(DeviceBitAndAssignAPI, BitAndAssign, bitand_assign, op_muta_refb_bitand_assign);
    impl_op_muta_refb_operator!(DeviceBitXorAssignAPI, BitXorAssign, bitxor_assign, op_muta_refb_bitxor_assign);
    impl_op_muta_refb_operator!(DeviceShlAssignAPI   , ShlAssign   , shl_assign   , op_muta_refb_shl_assign   );
    impl_op_muta_refb_operator!(DeviceShrAssignAPI   , ShrAssign   , shr_assign   , op_muta_refb_shr_assign   );
}
pub use impl_op_muta_refb_operator::*;

macro_rules! impl_op_muta_refb_unary {
    ($DeviceOpAPI:ident, $Op:ident, $op:ident, $op_muta_refb_func:ident) => {
        pub fn $op_muta_refb_func<RA, RB, DA, DB, TA, TB, B>(
            a: &mut TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<()>
        where
            // lifetime and data constraints
            RA: DataMutAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            // operation constraints
            TB: $Op<Output = TA>,
            B: $DeviceOpAPI<TA, TB, DA>,
        {
            rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
            let la = a.layout();
            let lb = b.layout();
            // all layouts should be broadcastable to lc
            // we can first generate broadcasted shape, then check this
            let (la_b, lb_b) = broadcast_layout_to_first(la, lb)?;
            rstsr_assert_eq!(la_b, *la, InvalidLayout)?;
            // op provided by device
            let device = a.device().clone();
            let storage_a = a.data_mut().storage_mut();
            let storage_b = b.data().storage();
            device.$op_muta_refb_func(storage_a, &la_b, storage_b, &lb_b)
        }

        impl<R, D, TA, TB, B> $Op for &TensorBase<R, D>
        where
            // lifetime and data constraints
            R: DataAPI<Data = Storage<TB, B>>,
            D: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // operation constraints
            TB: $Op<Output = TA>,
            B: $DeviceOpAPI<TA, TB, D>,
            B: DeviceCreationAnyAPI<TA>,
        {
            type Output = Tensor<TA, D, B>;
            fn $op(self) -> Self::Output {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K).unwrap();
                let mut storage_a =
                    unsafe { device.empty_impl(la.bounds_index().unwrap().1).unwrap() };
                // compute and return
                device.$op_muta_refb_func(&mut storage_a, &la, storage_b, lb).unwrap();
                return unsafe { Tensor::new_unchecked(DataOwned::from(storage_a), la) };
            }
        }

        impl<'b, D, TA, TB, B> $Op for TensorView<'b, TB, D, B>
        where
            // lifetime and data
            // constraints
            D: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // operation constraints
            TB: $Op<Output = TA>,
            B: $DeviceOpAPI<TA, TB, D>,
            B: DeviceCreationAnyAPI<TA>,
        {
            type Output = Tensor<TA, D, B>;
            fn $op(self) -> Self::Output {
                let lb = self.layout();
                let storage_b = self.data().storage();
                // generate empty output tensor
                let device = self.device();
                let la = layout_for_array_copy(lb, TensorIterOrder::K).unwrap();
                let mut storage_a =
                    unsafe { device.empty_impl(la.bounds_index().unwrap().1).unwrap() };
                // compute and return
                device.$op_muta_refb_func(&mut storage_a, &la, storage_b, lb).unwrap();
                return unsafe { Tensor::new_unchecked(DataOwned::from(storage_a), la) };
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_unary {
    use super::*;
    use core::ops::*;
    impl_op_muta_refb_unary!(DeviceNegAPI, Neg, neg, op_muta_refb_neg);
    impl_op_muta_refb_unary!(DeviceNotAPI, Not, not, op_muta_refb_not);
}
pub use impl_op_muta_refb_unary::*;

/* #endregion */

/* #region op_owna_refb_operation */

macro_rules! op_owna_refb_operator {
    (
        $op: ident,
        $DeviceOpAPI: ident,
        $Op: ident,
        $op_refa_refb_func: ident,
        $closure_muta_refb: expr,
        $closure_refa_mutb: expr
    ) => {
        impl<RB, DA, DB, TA, TB, B> $Op<&TensorBase<RB, DB>> for Tensor<TA, DA, B>
        where
            // lifetime and
            // data constraints
            RB: DataAPI<Data = Storage<TB, B>>,
            TA: Clone,
            TB: Clone,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            DA: DimConvertAPI<<DA as DimMaxAPI<DB>>::Max>,
            // operation constraints
            TA: $Op<TB, Output = TA>,
            B: DeviceCreationAnyAPI<TA>,
            B: $DeviceOpAPI<TA, TB, TA, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceOp_MutA_RefB_API<TA, TB, DA, fn(&mut TA, &TB)>,
        {
            type Output = Tensor<TA, <DA as DimMaxAPI<DB>>::Max, B>;
            fn $op(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
                if self.layout().is_broadcasted()
                    || broadcast_layout_to_first(self.layout(), rhs.layout()).is_err()
                {
                    // output shape of c is not the same to input owned a
                    $op_refa_refb_func(&self, rhs).unwrap()
                } else {
                    // reuse a as c
                    let mut s = self;
                    op_muta_refb_func(&mut s, rhs, $closure_muta_refb).unwrap();
                    s.into_dim::<<DA as DimMaxAPI<DB>>::Max>().unwrap() // this unwrap
                                                                        // should be safe
                }
            }
        }
        impl<'b, DA, DB, TA, TB, B> $Op<TensorView<'b, TB, DB, B>> for Tensor<TA, DA, B>
        where
            // lifetime and data constraints
            TA: Clone,
            TB: Clone,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            DA: DimConvertAPI<<DA as DimMaxAPI<DB>>::Max>,
            // operation constraints
            TA: $Op<TB, Output = TA>,
            B: DeviceCreationAnyAPI<TA>,
            B: $DeviceOpAPI<TA, TB, TA, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceOp_MutA_RefB_API<TA, TB, DA, fn(&mut TA, &TB)>,
        {
            type Output = Tensor<TA, <DA as DimMaxAPI<DB>>::Max, B>;
            fn $op(self, rhs: TensorView<'b, TB, DB, B>) -> Self::Output {
                let rhs = &rhs.view();
                if self.layout().is_broadcasted()
                    || broadcast_layout_to_first(self.layout(), rhs.layout()).is_err()
                {
                    // output shape of c is not the same to input owned a
                    $op_refa_refb_func(&self, rhs).unwrap()
                } else {
                    // reuse a as c
                    let mut s = self;
                    op_muta_refb_func(&mut s, rhs, $closure_muta_refb).unwrap();
                    s.into_dim::<<DA as DimMaxAPI<DB>>::Max>().unwrap() // this unwrap
                                                                        // should be safe
                }
            }
        }

        impl<RA, DA, DB, TA, TB, B> $Op<Tensor<TB, DB, B>> for &TensorBase<RA, DA>
        where
            // lifetime and
            // data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            TA: Clone,
            TB: Clone,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DB: DimMaxAPI<DA>,
            DA: DimMaxAPI<DB>,
            <DB as DimMaxAPI<DA>>::Max: DimConvertAPI<DB>,
            DB: DimConvertAPI<<DA as DimMaxAPI<DB>>::Max>,
            // operation constraints
            TA: $Op<TB, Output = TB>,
            B: DeviceCreationAnyAPI<TB>,
            B: $DeviceOpAPI<TA, TB, TB, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceOp_MutA_RefB_API<TB, TA, DB, fn(&mut TB, &TA)>,
        {
            type Output = Tensor<TB, <DA as DimMaxAPI<DB>>::Max, B>;
            fn $op(self, rhs: Tensor<TB, DB, B>) -> Self::Output {
                if self.layout().is_broadcasted()
                    || broadcast_layout_to_first(rhs.layout(), self.layout()).is_err()
                {
                    // output shape of c is not the same to input owned a
                    $op_refa_refb_func(self, &rhs).unwrap()
                } else {
                    // reuse b as c
                    let mut rhs = rhs;
                    op_muta_refb_func(&mut rhs, self, $closure_refa_mutb).unwrap();
                    rhs.into_dim::<<DA as DimMaxAPI<DB>>::Max>().unwrap()
                }
            }
        }

        impl<'a, DA, DB, TA, TB, B> $Op<Tensor<TB, DB, B>> for TensorView<'a, TA, DA, B>
        where
            // lifetime and data constraints
            TA: Clone,
            TB: Clone,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DB: DimMaxAPI<DA>,
            DA: DimMaxAPI<DB>,
            <DB as DimMaxAPI<DA>>::Max: DimConvertAPI<DB>,
            DB: DimConvertAPI<<DA as DimMaxAPI<DB>>::Max>,
            // operation constraints
            TA: $Op<TB, Output = TB>,
            B: DeviceCreationAnyAPI<TB>,
            B: $DeviceOpAPI<TA, TB, TB, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceOp_MutA_RefB_API<TB, TA, DB, fn(&mut TB, &TA)>,
        {
            type Output = Tensor<TB, <DA as DimMaxAPI<DB>>::Max, B>;
            fn $op(self, rhs: Tensor<TB, DB, B>) -> Self::Output {
                let lhs = &self.view();
                if lhs.layout().is_broadcasted()
                    || broadcast_layout_to_first(rhs.layout(), lhs.layout()).is_err()
                {
                    // output shape of c is not the same to input owned a
                    $op_refa_refb_func(lhs, &rhs).unwrap()
                } else {
                    // reuse b as c
                    let mut rhs = rhs;
                    op_muta_refb_func(&mut rhs, lhs, $closure_refa_mutb).unwrap();
                    rhs.into_dim::<<DA as DimMaxAPI<DB>>::Max>().unwrap()
                }
            }
        }

        impl<DA, DB, T, B> $Op<Tensor<T, DB, B>> for Tensor<T, DA, B>
        where
            // lifetime and
            // data constraints
            T: Clone,
            DA: DimAPI,
            DB: DimAPI,
            B: DeviceAPI<T>,
            // broadcast constraints
            DB: DimMaxAPI<DA>,
            DA: DimMaxAPI<DB>,
            <DB as DimMaxAPI<DA>>::Max: DimConvertAPI<DB>,
            <DA as DimMaxAPI<DB>>::Max: DimConvertAPI<DA>,
            DB: DimConvertAPI<<DA as DimMaxAPI<DB>>::Max>,
            DA: DimConvertAPI<<DA as DimMaxAPI<DB>>::Max>,
            // operation constraints
            T: $Op<T, Output = T>,
            B: DeviceCreationAnyAPI<T>,
            B: $DeviceOpAPI<T, T, T, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceOp_MutA_RefB_API<T, T, DA, fn(&mut T, &T)>,
            B: DeviceOp_MutA_RefB_API<T, T, DB, fn(&mut T, &T)>,
        {
            type Output = Tensor<T, <DA as DimMaxAPI<DB>>::Max, B>;
            fn $op(self, rhs: Tensor<T, DB, B>) -> Self::Output {
                if !self.layout().is_broadcasted()
                    && broadcast_layout_to_first(self.layout(), rhs.layout()).is_ok()
                {
                    // reuse a as c
                    let mut s = self;
                    op_muta_refb_func(&mut s, &rhs, $closure_muta_refb).unwrap();
                    s.into_dim::<<DA as DimMaxAPI<DB>>::Max>().unwrap()
                } else if !rhs.layout().is_broadcasted()
                    && broadcast_layout_to_first(rhs.layout(), self.layout()).is_ok()
                {
                    // reuse b as c
                    let mut rhs = rhs;
                    op_muta_refb_func(&mut rhs, &self, $closure_refa_mutb).unwrap();
                    rhs.into_dim::<<DA as DimMaxAPI<DB>>::Max>().unwrap()
                } else {
                    // output shape of c is not the same to input owned a
                    $op_refa_refb_func(&self, &rhs).unwrap()
                }
            }
        }
    };
}

#[rustfmt::skip]
pub mod op_owna_refb_operator {
    use super::*;
    use core::ops::*;
    op_owna_refb_operator!(add   , DeviceAddAPI   , Add   , op_refa_refb_add   , |a, b| *a = a.clone() +  b.clone(), |b, a| *b = a.clone() +  b.clone());
    op_owna_refb_operator!(sub   , DeviceSubAPI   , Sub   , op_refa_refb_sub   , |a, b| *a = a.clone() -  b.clone(), |b, a| *b = a.clone() -  b.clone());
    op_owna_refb_operator!(mul   , DeviceMulAPI   , Mul   , op_refa_refb_mul   , |a, b| *a = a.clone() *  b.clone(), |b, a| *b = a.clone() *  b.clone());
    op_owna_refb_operator!(div   , DeviceDivAPI   , Div   , op_refa_refb_div   , |a, b| *a = a.clone() /  b.clone(), |b, a| *b = a.clone() /  b.clone());
//  op_owna_refb_operator!(rem   , DeviceRemAPI   , Rem   , op_refa_refb_rem   , |a, b| *a = a.clone() %  b.clone(), |b, a| *b = a.clone() %  b.clone());
    op_owna_refb_operator!(bitor , DeviceBitOrAPI , BitOr , op_refa_refb_bitor , |a, b| *a = a.clone() |  b.clone(), |b, a| *b = a.clone() |  b.clone());
    op_owna_refb_operator!(bitand, DeviceBitAndAPI, BitAnd, op_refa_refb_bitand, |a, b| *a = a.clone() &  b.clone(), |b, a| *b = a.clone() &  b.clone());
    op_owna_refb_operator!(bitxor, DeviceBitXorAPI, BitXor, op_refa_refb_bitxor, |a, b| *a = a.clone() ^  b.clone(), |b, a| *b = a.clone() ^  b.clone());
    op_owna_refb_operator!(shl   , DeviceShlAPI   , Shl   , op_refa_refb_shl   , |a, b| *a = a.clone() << b.clone(), |b, a| *b = a.clone() << b.clone());
    op_owna_refb_operator!(shr   , DeviceShrAPI   , Shr   , op_refa_refb_shr   , |a, b| *a = a.clone() >> b.clone(), |b, a| *b = a.clone() >> b.clone());
}

/* #endregion */

/* #region op_owna_operation */

macro_rules! impl_op_muta_unary {
    ($op:ident, $Op:ident, $op_muta_refb_closure:expr) => {
        impl<T, D, B> $Op for Tensor<T, D, B>
        where
            // lifetime and data constraints
            T: Clone,
            D: DimAPI,
            B: DeviceAPI<T>,
            // op provided by device
            T: $Op<Output = T>,
            B: DeviceOp_MutA_API<T, D, fn(&mut T)>,
        {
            type Output = Tensor<T, D, B>;
            fn $op(self) -> Self::Output {
                let mut s = self;
                op_muta_func(&mut s, $op_muta_refb_closure).unwrap();
                s
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_unary {
    use super::*;
    use core::ops::*;
    impl_op_muta_unary!(neg, Neg, |a| *a = -a.clone());
    impl_op_muta_unary!(not, Not, |a| *a = !a.clone());
}

/* #endregion */

#[cfg(test)]
mod test {
    use crate::prelude_dev::*;

    #[test]
    fn test_add() {
        // contiguous
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let c = &a + &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let a = Tensor::linspace_cpu(1.0, 6.0, 6).into_shape_assume_contig::<Ix2>([2, 3]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 6.0, 3);
        let c = &a + &b;
        let c_ref = vec![3., 6., 9., 6., 9., 12.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [1, 2, 3] + [5, 1, 2, 1]
        // a = np.linspace(1, 6, 6).reshape(1, 2, 3)
        // b = np.linspace(1, 10, 10).reshape(5, 1, 2, 1)
        let a = Tensor::linspace_cpu(1.0, 6.0, 6);
        let a = a.into_shape_assume_contig::<Ix3>([1, 2, 3]).unwrap();
        let b = Tensor::linspace_cpu(1.0, 10.0, 10);
        let b = b.into_shape_assume_contig::<Ix4>([5, 1, 2, 1]).unwrap();
        let c = &a + &b;
        let c_ref = vec![
            2., 3., 4., 6., 7., 8., 4., 5., 6., 8., 9., 10., 6., 7., 8., 10., 11., 12., 8., 9.,
            10., 12., 13., 14., 10., 11., 12., 14., 15., 16.,
        ];
        let c_ref = c_ref.into();
        assert!(allclose_f64(&c, &c_ref));

        // transposed
        let a = Tensor::linspace_cpu(1.0, 9.0, 9);
        let a = a.into_shape_assume_contig::<Ix2>([3, 3]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 18.0, 9);
        let b = b.into_shape_assume_contig::<Ix2>([3, 3]).unwrap().into_reverse_axes();
        let c = &a + &b;
        let c_ref = vec![3., 10., 17., 8., 15., 22., 13., 20., 27.].into();
        assert!(allclose_f64(&c, &c_ref));

        // negative strides
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let a = a.flip(0);
        let c = &a + &b;
        let c_ref = vec![7., 8., 9., 10., 11.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let b = b.flip(0);
        let c = &a + &b;
        let c_ref = vec![11., 10., 9., 8., 7.].into();
        assert!(allclose_f64(&c, &c_ref));

        // view
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let c = a.view() + &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));

        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let c = a + b.view();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_sub() {
        // contiguous
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let c = &a - &b;
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_mul() {
        // contiguous
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let c = &a * &b;
        let c_ref = vec![2., 8., 18., 32., 50.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    #[allow(clippy::deref_addrof)]
    fn test_add_assign() {
        // contiguous
        let mut c = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        c += &b;
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        c += b;
        let c_ref = vec![5., 10., 15., 20., 25.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let c = Tensor::linspace_cpu(1.0, 6.0, 6);
        let mut c = c.into_shape_assume_contig::<Ix2>([2, 3]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 6.0, 3);
        // let mut c_mut = c.view_mut();
        // c_mut += &b;
        *&mut c.view_mut() += &b;
        let c_ref = vec![3., 6., 9., 6., 9., 12.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    #[allow(clippy::deref_addrof)]
    fn test_sub_assign() {
        // contiguous
        let mut c = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        c -= &b;
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));

        // broadcast
        // [2, 3] + [3]
        let c = Tensor::linspace_cpu(1.0, 6.0, 6);
        let mut c = c.into_shape_assume_contig::<Ix2>([2, 3]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 6.0, 3);
        // let mut c_mut = c.view_mut();
        // c_mut += &b;
        *&mut c.view_mut() -= &b;
        let c_ref = vec![-1., -2., -3., 2., 1., 0.].into();
        assert!(allclose_f64(&c, &c_ref));
    }

    #[test]
    fn test_add_consume() {
        // a + &b, same shape
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + &b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a + &b, broadcastable
        let a = Tensor::linspace_cpu(1.0, 10.0, 10).into_shape_assume_contig([2, 5]).unwrap();
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + &b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
        // a + &b, non-broadcastable
        let a = Tensor::linspace_cpu(2.0, 10.0, 5);
        let b = Tensor::linspace_cpu(1.0, 10.0, 10).into_shape_assume_contig([2, 5]).unwrap();
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + &b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15., 8., 11., 14., 17., 20.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_ne!(a_ptr, c_ptr);
        // &a + b
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let b_ptr = b.data().storage().rawvec().as_ptr();
        let c = &a + b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(b_ptr, c_ptr);
        // a + b, same shape
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a + b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![3., 6., 9., 12., 15.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
    }

    #[test]
    fn test_sub_consume() {
        // &a - b
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let b_ptr = b.data().storage().rawvec().as_ptr();
        let c = &a - b;
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(b_ptr, c_ptr);
        // a - &b
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = Tensor::linspace_cpu(2.0, 10.0, 5);
        let a_ptr = a.data().storage().rawvec().as_ptr();
        let c = a - b.view();
        let c_ptr = c.data().storage().rawvec().as_ptr();
        let c_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&c, &c_ref));
        assert_eq!(a_ptr, c_ptr);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::linspace_cpu(1.0, 5.0, 5);
        let b = -&a;
        let b_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&b, &b_ref));
        let b = -a;
        let b_ref = vec![-1., -2., -3., -4., -5.].into();
        assert!(allclose_f64(&b, &b_ref));
    }
}
