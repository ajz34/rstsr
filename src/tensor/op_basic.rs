use crate::prelude_dev::*;

/* #region ternary-op */

macro_rules! impl_tensor_op_ref {
    ($DeviceOpAPI:ident, $Op:ident, $op_ternary:ident, $op:ident, $tensor_op_ternary:ident, $tensor_op_binary:ident) => {
        pub fn $tensor_op_ternary<RC, RA, RB, DC, DA, DB, T, TB, B>(
            c: &mut TensorBase<RC, DC>,
            a: &TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<()>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<T, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            RC: DataMutAPI<Data = Storage<T, B>>,
            DA: DimAPI,
            DB: DimAPI,
            DC: DimAPI,
            T: Clone,
            TB: Clone,
            B: DeviceAPI<T> + DeviceAPI<TB>,
            // broadcast constraints
            DC: DimMaxAPI<DA> + DimMaxAPI<DB>,
            <DC as DimMaxAPI<DA>>::Max: DimConvertAPI<DC>,
            <DC as DimMaxAPI<DB>>::Max: DimConvertAPI<DC>,
            // operation constraints
            T: core::ops::$Op<TB, Output = T>,
            B: $DeviceOpAPI<T, TB, DC>,
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
            device.$op_ternary(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b)
        }

        pub fn $tensor_op_binary<RA, RB, DA, DB, T, TB, B>(
            a: &TensorBase<RA, DA>,
            b: &TensorBase<RB, DB>,
        ) -> Result<Tensor<T, <DA as DimMaxAPI<DB>>::Max, B>>
        where
            // lifetime and data constraints
            RA: DataAPI<Data = Storage<T, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            T: Clone,
            TB: Clone,
            B: DeviceAPI<T> + DeviceAPI<TB> + DeviceCreationAnyAPI<T>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            T: core::ops::$Op<TB, Output = T>,
            B: $DeviceOpAPI<T, TB, <DA as DimMaxAPI<DB>>::Max>,
        {
            rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
            let la = a.layout();
            let lb = b.layout();
            let (la_b, lb_b) = broadcast_layout(la, lb)?;
            // generate output layout
            let lc = if la_b.c_contig() && lb_b.c_contig() {
                la_b.shape().c()
            } else if la_b.f_contig() && lb_b.f_contig() {
                la_b.shape().f()
            } else {
                match TensorOrder::default() {
                    TensorOrder::C => la_b.shape().c(),
                    TensorOrder::F => la_b.shape().f(),
                }
            };
            // generate empty c
            let device = a.device().clone();
            let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
            // add provided by device
            let storage_a = a.data().storage();
            let storage_b = b.data().storage();
            device.$op_ternary(&mut storage_c, &lc, storage_a, &la_b, storage_b, &lb_b)?;
            // return tensor
            Tensor::new(DataOwned { storage: storage_c }, lc)
        }

        impl<RA, RB, DA, DB, T, TB, B> core::ops::$Op<&TensorBase<RB, DB>> for &TensorBase<RA, DA>
        where
            // lifetime and data
            // constraints
            RA: DataAPI<Data = Storage<T, B>>,
            RB: DataAPI<Data = Storage<TB, B>>,
            DA: DimAPI,
            DB: DimAPI,
            T: Clone,
            TB: Clone,
            B: DeviceAPI<T> + DeviceAPI<TB> + DeviceCreationAnyAPI<T>,
            // broadcast constraints
            DA: DimMaxAPI<DB>,
            // operation constraints
            T: core::ops::$Op<TB, Output = T>,
            B: $DeviceOpAPI<T, TB, <DA as DimMaxAPI<DB>>::Max>,
        {
            type Output = Tensor<T, <DA as DimMaxAPI<DB>>::Max, B>;

            fn $op(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
                $tensor_op_binary(self, rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_tensor_op_ref {
    use super::*;
    impl_tensor_op_ref!(DeviceAddAPI, Add, add_ternary, add, tensor_add_ternary, tensor_add_binary);
    impl_tensor_op_ref!(DeviceSubAPI, Sub, sub_ternary, sub, tensor_sub_ternary, tensor_sub_binary);
    impl_tensor_op_ref!(DeviceMulAPI, Mul, mul_ternary, mul, tensor_mul_tenary, tensor_mul_binary);
    impl_tensor_op_ref!(DeviceDivAPI, Div, div_ternary, div, tensor_div_tenary, tensor_div_binary);
    impl_tensor_op_ref!(DeviceRemAPI, Rem, rem_ternary, rem, tensor_rem_tenary, tensor_rem_binary);
    impl_tensor_op_ref!(DeviceBitOrAPI, BitOr, bitor_ternary, bitor, tensor_bitor_tenary, tensor_bitor_binary);
    impl_tensor_op_ref!(DeviceBitAndAPI, BitAnd, bitand_ternary, bitand, tensor_bitand_tenary, tensor_bitand_binary);
    impl_tensor_op_ref!(DeviceBitXorAPI, BitXor, bitxor_ternary, bitxor, tensor_bitxor_tenary, tensor_bitxor_binary);
    impl_tensor_op_ref!(DeviceShlAPI, Shl, shl_ternary, shl, tensor_shl_tenary, tensor_shl_binary);
    impl_tensor_op_ref!(DeviceShrAPI, Shr, shr_ternary, shr, tensor_shr_tenary, tensor_shr_binary);
}
pub use impl_tensor_op_ref::*;

/* #endregion */

/* #region binary-op */

macro_rules! impl_tensor_op_assign_ref {
    ($DeviceOpAPI:ident, $Op:ident, $op_binary:ident, $op:ident, $tensor_op_binary:ident) => {
        pub fn $tensor_op_binary<RC, RB, DC, DB, T, TB, B>(
            c: &mut TensorBase<RC, DC>,
            b: &TensorBase<RB, DB>,
        ) -> Result<()>
        where
            // lifetime and data constraints
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
            rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
            let lc = c.layout();
            let lb = b.layout();
            let (lc_b, lb_b) = broadcast_layout(lc, lb)?;
            let lc_b = lc_b.into_dim::<DC>()?;
            let lb_b = lb_b.into_dim::<DC>()?;
            // all layouts should be broadcastable to lc
            rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
            // op provided by device
            let device = c.device().clone();
            let storage_c = c.data_mut().as_storage_mut();
            let storage_b = b.data().storage();
            device.$op_binary(storage_c, &lc_b, storage_b, &lb_b)
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
                $tensor_op_binary(self, rhs).unwrap()
            }
        }
    };
}

#[rustfmt::skip]
mod impl_tensor_op_assign_ref {
    use super::*;
    impl_tensor_op_assign_ref!(DeviceAddAssignAPI, AddAssign, add_assign_binary, add_assign, tensor_add_assign_binary);
    impl_tensor_op_assign_ref!(DeviceSubAssignAPI, SubAssign, sub_assign_binary, sub_assign, tensor_sub_assign_binary);
    impl_tensor_op_assign_ref!(DeviceMulAssignAPI, MulAssign, mul_assign_binary, mul_assign, tensor_mul_assign_binary);
    impl_tensor_op_assign_ref!(DeviceDivAssignAPI, DivAssign, div_assign_binary, div_assign, tensor_div_assign_binary);
    impl_tensor_op_assign_ref!(DeviceRemAssignAPI, RemAssign, rem_assign_binary, rem_assign, tensor_rem_assign_binary);
    impl_tensor_op_assign_ref!(DeviceBitOrAssignAPI, BitOrAssign, bitor_assign_binary, bitor_assign, tensor_bitor_assign_binary);
    impl_tensor_op_assign_ref!(DeviceBitAndAssignAPI, BitAndAssign, bitand_assign_binary, bitand_assign, tensor_bitand_assign_binary);
    impl_tensor_op_assign_ref!(DeviceBitXorAssignAPI, BitXorAssign, bitxor_assign_binary, bitxor_assign, tensor_bitxor_assign_binary);
    impl_tensor_op_assign_ref!(DeviceShlAssignAPI, ShlAssign, shl_assign_binary, shl_assign, tensor_shl_assign_binary);
    impl_tensor_op_assign_ref!(DeviceShrAssignAPI, ShrAssign, shr_assign_binary, shr_assign, tensor_shr_assign_binary);
}
pub use impl_tensor_op_assign_ref::*;

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
