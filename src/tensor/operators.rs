use crate::prelude_dev::*;

/* #region op_mutc_refa_refb_func */

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
    let storage_c = c.data_mut().as_storage_mut();
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

/* #endregion */

/* #region op_mutc_refa_refb_operation */

macro_rules! impl_op_mutc_refa_refb_func {
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
            TA: core::ops::$Op<TB, Output = TC>,
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
            let storage_c = c.data_mut().as_storage_mut();
            let storage_a = a.data().storage();
            let storage_b = b.data().storage();
            device.$op_mutc_refa_refb_func(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b)
        }

        pub fn $op_refa_refb_func<RA, RB, DA, DB, TA, TB, B>(
            a: &TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<Tensor<<TA as core::ops::$Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            TA: Clone,
            TB: Clone,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: core::ops::$Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as core::ops::$Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as core::ops::$Op<TB>>::Output>,
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
            device.$op_mutc_refa_refb_func(
                &mut storage_c,
                &lc,
                storage_a,
                &la_b,
                storage_b,
                &lb_b,
            )?;
            // return tensor
            Tensor::new(DataOwned::from(storage_c), lc)
        }

        impl<RA, RB, DA, DB, TA, TB, B> core::ops::$Op<&TensorBase<RB, DB>> for &TensorBase<RA, DA>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<TA, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            TA: Clone,
            TB: Clone,
            B: DeviceAPI<TA> + DeviceAPI<TB>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            TA: core::ops::$Op<TB>,
            B: $DeviceOpAPI<TA, TB, <TA as core::ops::$Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max>,
            B: DeviceCreationAnyAPI<<TA as core::ops::$Op<TB>>::Output>,
        {
            type Output = Tensor<<TA as core::ops::$Op<TB>>::Output, <DA as DimMaxAPI<DB>>::Max, B>;

            fn $op(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
                $op_refa_refb_func(self, rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_mutc_refa_refb_func {
    use super::*;
    impl_op_mutc_refa_refb_func!(DeviceAddAPI   , Add   , add   , op_mutc_refa_refb_add   , op_refa_refb_add   );
    impl_op_mutc_refa_refb_func!(DeviceSubAPI   , Sub   , sub   , op_mutc_refa_refb_sub   , op_refa_refb_sub   );
    impl_op_mutc_refa_refb_func!(DeviceMulAPI   , Mul   , mul   , op_mutc_refa_refb_mul   , op_refa_refb_mul   );
    impl_op_mutc_refa_refb_func!(DeviceDivAPI   , Div   , div   , op_mutc_refa_refb_div   , op_refa_refb_div   );
    impl_op_mutc_refa_refb_func!(DeviceRemAPI   , Rem   , rem   , op_mutc_refa_refb_rem   , op_refa_refb_rem   );
    impl_op_mutc_refa_refb_func!(DeviceBitOrAPI , BitOr , bitor , op_mutc_refa_refb_bitor , op_refa_refb_bitor );
    impl_op_mutc_refa_refb_func!(DeviceBitAndAPI, BitAnd, bitand, op_mutc_refa_refb_bitand, op_refa_refb_bitand);
    impl_op_mutc_refa_refb_func!(DeviceBitXorAPI, BitXor, bitxor, op_mutc_refa_refb_bitxor, op_refa_refb_bitxor);
    impl_op_mutc_refa_refb_func!(DeviceShlAPI   , Shl   , shl   , op_mutc_refa_refb_shl   , op_refa_refb_shl   );
    impl_op_mutc_refa_refb_func!(DeviceShrAPI   , Shr   , shr   , op_mutc_refa_refb_shr   , op_refa_refb_shr   );
}
pub use impl_op_mutc_refa_refb_func::*;

/* #endregion */

/* #region op_muta_refb_func */

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
    let storage_a = a.data_mut().as_storage_mut();
    let storage_b = b.data().storage();
    device.op_muta_refb_func(storage_a, &la_b, storage_b, &lb_b, f)
}

macro_rules! impl_op_muta_refb_func {
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
            TA: core::ops::$Op<TB>,
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
            let storage_a = a.data_mut().as_storage_mut();
            let storage_b = b.data().storage();
            device.$op_muta_refb_func(storage_a, &la_b, storage_b, &lb_b)
        }

        impl<RC, RB, DC, DB, T, TB, B> core::ops::$Op<&TensorBase<RB, DB>> for TensorBase<RC, DC>
        where
            // lifetime and
            // data constraints
            RC: DataMutAPI<Data = Storage<T, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DC: DimAPI,
            DB: DimAPI,
            T: Clone,
            TB: Clone,
            B: DeviceAPI<T> + DeviceAPI<TB>,
            // broadcast constraints
            DC: DimMaxAPI<DB>,
            <DC as DimMaxAPI<DB>>::Max: DimConvertAPI<DC>,
            // operation constraints
            T: core::ops::$Op<TB>,
            B: $DeviceOpAPI<T, TB, DC>,
        {
            fn $op(&mut self, rhs: &TensorBase<RB, DB>) {
                $op_muta_refb_func(self, rhs).unwrap()
            }
        }

        impl<RC, RB, DC, DB, T, TB, B> core::ops::$Op<TensorBase<RB, DB>> for TensorBase<RC, DC>
        where
            // lifetime and
            // data constraints
            RC: DataMutAPI<Data = Storage<T, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DC: DimAPI,
            DB: DimAPI,
            T: Clone,
            TB: Clone,
            B: DeviceAPI<T> + DeviceAPI<TB>,
            // broadcast constraints
            DC: DimMaxAPI<DB>,
            <DC as DimMaxAPI<DB>>::Max: DimConvertAPI<DC>,
            // operation constraints
            T: core::ops::$Op<TB>,
            B: $DeviceOpAPI<T, TB, DC>,
        {
            fn $op(&mut self, rhs: TensorBase<RB, DB>) {
                $op_muta_refb_func(self, &rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_func {
    use super::*;
    impl_op_muta_refb_func!(DeviceAddAssignAPI   , AddAssign   , add_assign, op_muta_refb_add_assign);
    impl_op_muta_refb_func!(DeviceSubAssignAPI   , SubAssign   , sub_assign, op_muta_refb_sub_assign);
    impl_op_muta_refb_func!(DeviceMulAssignAPI   , MulAssign   , mul_assign, op_muta_refb_mul_assign);
    impl_op_muta_refb_func!(DeviceDivAssignAPI   , DivAssign   , div_assign, op_muta_refb_div_assign);
    impl_op_muta_refb_func!(DeviceRemAssignAPI   , RemAssign   , rem_assign, op_muta_refb_rem_assign);
    impl_op_muta_refb_func!(DeviceBitOrAssignAPI , BitOrAssign , bitor_assign, op_muta_refb_bitor_assign);
    impl_op_muta_refb_func!(DeviceBitAndAssignAPI, BitAndAssign, bitand_assign, op_muta_refb_bitand_assign);
    impl_op_muta_refb_func!(DeviceBitXorAssignAPI, BitXorAssign, bitxor_assign, op_muta_refb_bitxor_assign);
    impl_op_muta_refb_func!(DeviceShlAssignAPI   , ShlAssign   , shl_assign, op_muta_refb_shl_assign);
    impl_op_muta_refb_func!(DeviceShrAssignAPI   , ShrAssign   , shr_assign, op_muta_refb_shr_assign);
}
pub use impl_op_muta_refb_func::*;

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
}
