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
    DC: DimMaxAPI<DA, Max = DC> + DimMaxAPI<DB, Max = DC>,
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

pub fn op_refa_refb_func<RA, RB, DA, DB, DC, TA, TB, TC, B, F>(
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
    DC: DimAPI,
    B: DeviceAPI<TA> + DeviceAPI<TB>,
    // broadcast constraints
    DA: DimMaxAPI<DB, Max = DC>,
    // operation constraints
    B: DeviceOp_MutC_RefA_RefB_API<TA, TB, TC, DC, F>,
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
    DA: DimMaxAPI<DB, Max = DA>,
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
    use super::*;

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
