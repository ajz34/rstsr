//! Element-wise binary "common" functions: two-argument math
//! ([`atan2`](atan2()), [`copysign`](copysign()), [`hypot`](hypot()),
//! [`log_add_exp`](log_add_exp()), [`nextafter`](nextafter()),
//! [`floor_divide`](floor_divide()), [`pow`](pow()),
//! [`maximum`](maximum())/[`minimum`](minimum())) and comparisons
//! ([`equal`](equal()), [`not_equal`](not_equal()), [`greater`](greater()),
//! [`greater_equal`](greater_equal()), [`less`](less()),
//! [`less_equal`](less_equal()) and their alias forms [`eq`](eq()),
//! [`ne`](ne()), [`gt`](gt()), [`ge`](ge()), [`lt`](lt()), [`le`](le())).
//!
//! Comparison functions return boolean tensors; comparison *operators*
//! (`==`, `<`, ...) between tensors are not overloaded, so use these
//! functions.
//!
//! [`pow`](pow()) promotes mixed operand dtypes to their common type (which
//! is also the result dtype) and accepts integer, floating-point and complex
//! operands. An integer base with a negative integer exponent is rejected
//! (the array API leaves it unspecified; NumPy raises there too), and a
//! complex base uses the principal branch `exp(exponent * ln(base))`.
//!
//! The `ext_*` promoted binary family — [`ext_add`](ext_add()),
//! [`ext_sub`](ext_sub()), [`ext_mul`](ext_mul()), [`ext_div`](ext_div()),
//! [`ext_bitand`](ext_bitand()), [`ext_bitor`](ext_bitor()),
//! [`ext_bitxor`](ext_bitxor()), [`ext_shl`](ext_shl()), [`ext_shr`](ext_shr())
//! — also promotes mixed operand dtypes, serving the Array API binary
//! functions. In Rust code, prefer the native operators (`+`, `-`, `*`, `/`,
//! `&`, `|`, `^`, `<<`, `>>`) and the same-dtype functions (`rt::add`,
//! `rt::bitand`, ...); the `ext_*` family exists for Array API fulfillment and
//! is not the idiomatic rstsr surface.
//!
//! # Examples
//!
//! ```rust
//! # use rstsr::prelude::*;
//! # let mut device = DeviceCpu::default();
//! # device.set_default_order(RowMajor);
//! let a = rt::tensor_from_nested!([1, 5, 3], &device);
//! let b = rt::tensor_from_nested!([4, 2, 6], &device);
//! println!("{}", rt::maximum(&a, &b));
//! // [ 4 5 6]
//! println!("{}", rt::minimum(&a, &b));
//! // [ 1 2 3]
//! println!("{}", rt::gt(&a, &b));
//! // [ false true false]
//! println!("{}", rt::eq(&a, &b));
//! // [ false false false]
//! # assert_eq!(format!("{}", rt::maximum(&a, &b)), "[ 4 5 6]");
//! # assert_eq!(format!("{}", rt::gt(&a, &b)), "[ false true false]");
//! ```
//!
//! [`pow`](pow()) raises each element to a power:
//!
//! ```rust
//! # use rstsr::prelude::*;
//! # let mut device = DeviceCpu::default();
//! # device.set_default_order(RowMajor);
//! let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
//! println!("{}", rt::pow(&a, 2));
//! // [[ 1 4]
//! //  [ 9 16]]
//! # assert_eq!(format!("{}", rt::pow(&a, 2)), "[[ 1 4]\n [ 9 16]]");
//! ```

use crate::prelude_dev::*;

/* Structure of implementation

Exception functions:
- floor_divide: integer and float are different, so we need to implement two functions
- pow: different input types occurs

*/

/* #region tensor traits */

#[duplicate_item(
    op              op_f              TensorOpAPI             ;
   [atan2        ] [atan2_f        ] [TensorATan2API       ];
   [copysign     ] [copysign_f     ] [TensorCopySignAPI    ];
   [equal        ] [equal_f        ] [TensorEqualAPI       ];
   [floor_divide ] [floor_divide_f ] [TensorFloorDivideAPI ];
   [greater      ] [greater_f      ] [TensorGreaterAPI     ];
   [greater_equal] [greater_equal_f] [TensorGreaterEqualAPI];
   [hypot        ] [hypot_f        ] [TensorHypotAPI       ];
   [less         ] [less_f         ] [TensorLessAPI        ];
   [less_equal   ] [less_equal_f   ] [TensorLessEqualAPI   ];
   [log_add_exp  ] [log_add_exp_f  ] [TensorLogAddExpAPI   ];
   [maximum      ] [maximum_f      ] [TensorMaximumAPI     ];
   [minimum      ] [minimum_f      ] [TensorMinimumAPI     ];
   [not_equal    ] [not_equal_f    ] [TensorNotEqualAPI    ];
   [pow          ] [pow_f          ] [TensorPowAPI         ];
   [nextafter    ] [nextafter_f    ] [TensorNextAfterAPI   ];
   [ext_add      ] [ext_add_f      ] [TensorExtAddAPI      ];
   [ext_sub      ] [ext_sub_f      ] [TensorExtSubAPI      ];
   [ext_mul      ] [ext_mul_f      ] [TensorExtMulAPI      ];
   [ext_div      ] [ext_div_f      ] [TensorExtDivAPI      ];
   [ext_bitand   ] [ext_bitand_f   ] [TensorExtBitAndAPI   ];
   [ext_bitor    ] [ext_bitor_f    ] [TensorExtBitOrAPI    ];
   [ext_bitxor   ] [ext_bitxor_f   ] [TensorExtBitXorAPI   ];
   [ext_shl      ] [ext_shl_f      ] [TensorExtShlAPI      ];
   [ext_shr      ] [ext_shr_f      ] [TensorExtShrAPI      ];
)]
pub trait TensorOpAPI<TRB> {
    type Output;
    fn op_f(self, b: TRB) -> Result<Self::Output>;
    fn op(self, b: TRB) -> Self::Output
    where
        Self: Sized,
    {
        self.op_f(b).rstsr_unwrap()
    }
}

#[duplicate_item(
    op_f              TensorOpAPI               OpAPI             ;
   [atan2_f        ] [TensorATan2API       ] [OpATan2API       ];
   [copysign_f     ] [TensorCopySignAPI    ] [OpCopySignAPI    ];
   [equal_f        ] [TensorEqualAPI       ] [OpEqualAPI       ];
   [floor_divide_f ] [TensorFloorDivideAPI ] [OpFloorDivideAPI ];
   [greater_f      ] [TensorGreaterAPI     ] [OpGreaterAPI     ];
   [greater_equal_f] [TensorGreaterEqualAPI] [OpGreaterEqualAPI];
   [hypot_f        ] [TensorHypotAPI       ] [OpHypotAPI       ];
   [less_f         ] [TensorLessAPI        ] [OpLessAPI        ];
   [less_equal_f   ] [TensorLessEqualAPI   ] [OpLessEqualAPI   ];
   [log_add_exp_f  ] [TensorLogAddExpAPI   ] [OpLogAddExpAPI   ];
   [maximum_f      ] [TensorMaximumAPI     ] [OpMaximumAPI     ];
   [minimum_f      ] [TensorMinimumAPI     ] [OpMinimumAPI     ];
   [not_equal_f    ] [TensorNotEqualAPI    ] [OpNotEqualAPI    ];
   [pow_f          ] [TensorPowAPI         ] [OpPowAPI         ];
   [nextafter_f    ] [TensorNextAfterAPI   ] [OpNextAfterAPI   ];
   [ext_add_f      ] [TensorExtAddAPI      ] [OpExtAddAPI      ];
   [ext_sub_f      ] [TensorExtSubAPI      ] [OpExtSubAPI      ];
   [ext_mul_f      ] [TensorExtMulAPI      ] [OpExtMulAPI      ];
   [ext_div_f      ] [TensorExtDivAPI      ] [OpExtDivAPI      ];
   [ext_bitand_f   ] [TensorExtBitAndAPI   ] [OpExtBitAndAPI   ];
   [ext_bitor_f    ] [TensorExtBitOrAPI    ] [OpExtBitOrAPI    ];
   [ext_bitxor_f   ] [TensorExtBitXorAPI   ] [OpExtBitXorAPI   ];
   [ext_shl_f      ] [TensorExtShlAPI      ] [OpExtShlAPI      ];
   [ext_shr_f      ] [TensorExtShrAPI      ] [OpExtShrAPI      ];
)]
mod impl_trait_binary {
    use super::*;

    impl<RA, TA, DA, RB, TB, DB, B> TensorOpAPI<&TensorAny<RB, TB, B, DB>> for &TensorAny<RA, TA, B, DA>
    where
        RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        DA: DimAPI + DimMaxAPI<DB>,
        DB: DimAPI,
        DA::Max: DimAPI,
        B: OpAPI<TA, TB, DA::Max>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
    {
        type Output = Tensor<B::TOut, B, DA::Max>;

        fn op_f(self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
            // check device
            rstsr_assert!(self.device().same_device(b.device()), DeviceMismatch)?;

            // check and broadcast layout
            let la = self.layout();
            let lb = b.layout();
            let default_order = self.device().default_order();
            let (la_b, lb_b) = broadcast_layout(la, lb, default_order)?;
            let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default())?;
            let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default())?;
            let lc = if lc_from_a == lc_from_b {
                lc_from_a
            } else {
                match self.device().default_order() {
                    RowMajor => la_b.shape().c(),
                    ColMajor => la_b.shape().f(),
                }
            };

            // perform operation and return
            let device = self.device();
            let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
            device.op_mutc_refa_refb(storage_c.raw_mut(), &lc, self.raw(), &la_b, b.raw(), &lb_b)?;
            // SAFETY: the op above wrote every element of the fresh `storage_c`.
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            Tensor::new_f(storage_c, lc)
        }
    }

    #[duplicate_item(
        ImplType                                                             TrA                         TrB                       ;
       [TA, DA, TB, DB, B, R: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>] [&TensorAny<R, TA, B, DA> ] [TensorView<'_, TB, B, DB>];
       [TA, DA, TB, DB, B, R: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>] [TensorView<'_, TA, B, DA>] [&TensorAny<R, TB, B, DB> ];
       [TA, DA, TB, DB, B                                                 ] [TensorView<'_, TA, B, DA>] [TensorView<'_, TB, B, DB>];
    )]
    impl<ImplType> TensorOpAPI<TrB> for TrA
    where
        DA: DimAPI + DimMaxAPI<DB>,
        DB: DimAPI,
        DA::Max: DimAPI,
        B: OpAPI<TA, TB, DA::Max>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
    {
        type Output = Tensor<B::TOut, B, DA::Max>;

        fn op_f(self, b: TrB) -> Result<Self::Output> {
            TensorOpAPI::op_f(&self.view(), &b.view())
        }
    }

    impl<RA, TA, DA, TB, B> TensorOpAPI<TB> for &TensorAny<RA, TA, B, DA>
    where
        RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        DA: DimAPI,
        B: OpAPI<TA, TB, DA>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TB: num::Num,
    {
        type Output = Tensor<B::TOut, B, DA>;

        fn op_f(self, b: TB) -> Result<Self::Output> {
            // check and broadcast layout
            let la = self.layout();
            let lc = layout_for_array_copy(la, TensorIterOrder::default())?;

            // perform operation and return
            let device = self.device();
            let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
            device.op_mutc_refa_numb(storage_c.raw_mut(), &lc, self.raw(), la, b)?;
            // SAFETY: the op above wrote every element of the fresh `storage_c`.
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            Tensor::new_f(storage_c, lc)
        }
    }

    impl<TA, DA, TB, B> TensorOpAPI<TB> for TensorView<'_, TA, B, DA>
    where
        DA: DimAPI,
        B: OpAPI<TA, TB, DA>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TB: num::Num,
    {
        type Output = Tensor<B::TOut, B, DA>;

        fn op_f(self, b: TB) -> Result<Self::Output> {
            (&self).op_f(b)
        }
    }

    impl<RB, TA, DB, TB, B> TensorOpAPI<&TensorAny<RB, TB, B, DB>> for TA
    where
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        DB: DimAPI,
        B: OpAPI<TA, TB, DB>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TA: num::Num,
    {
        type Output = Tensor<B::TOut, B, DB>;

        fn op_f(self, b: &TensorAny<RB, TB, B, DB>) -> Result<Self::Output> {
            // check and broadcast layout
            let lb = b.layout();
            let lc = layout_for_array_copy(lb, TensorIterOrder::default())?;

            // perform operation and return
            let device = b.device();
            let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
            device.op_mutc_numa_refb(storage_c.raw_mut(), &lc, self, b.raw(), lb)?;
            // SAFETY: the op above wrote every element of the fresh `storage_c`.
            let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
            Tensor::new_f(storage_c, lc)
        }
    }

    impl<TA, DB, TB, B> TensorOpAPI<TensorView<'_, TB, B, DB>> for TA
    where
        DB: DimAPI,
        B: OpAPI<TA, TB, DB>,
        B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<B::TOut> + DeviceCreationAnyAPI<B::TOut>,
        TA: num::Num,
    {
        type Output = Tensor<B::TOut, B, DB>;

        fn op_f(self, b: TensorView<'_, TB, B, DB>) -> Result<Self::Output> {
            TensorOpAPI::op_f(self, &b.view())
        }
    }
}

/* #endregion */

/* #region function impl */

macro_rules! func_binary {
    ($op: ident, $op_f: ident, $TensorOpAPI: ident, $OpAPI: ident, $($op2: ident, $op2_f: ident),*) => {
        pub fn $op_f<TRA, TRB>(a: TRA, b: TRB) -> Result<TRA::Output>
        where
            TRA: $TensorOpAPI<TRB>,
        {
            a.$op_f(b)
        }

        pub fn $op<TRA, TRB>(a: TRA, b: TRB) -> TRA::Output
        where
            TRA: $TensorOpAPI<TRB>,
        {
            a.$op(b)
        }

        $(
            pub fn $op2_f<TRA, TRB>(a: TRA, b: TRB) -> Result<TRA::Output>
            where
                TRA: $TensorOpAPI<TRB>,
            {
                a.$op_f(b)
            }

            pub fn $op2<TRA, TRB>(a: TRA, b: TRB) -> TRA::Output
            where
                TRA: $TensorOpAPI<TRB>,
            {
                a.$op(b)
            }
        )*
    };
}

#[rustfmt::skip]
mod func_binary {
    use super::*;
    func_binary!(atan2         , atan2_f           , TensorATan2API            , DeviceATan2API            ,);
    func_binary!(copysign      , copysign_f        , TensorCopySignAPI         , DeviceCopySignAPI         ,);
    func_binary!(floor_divide  , floor_divide_f    , TensorFloorDivideAPI      , DeviceFloorDivideAPI      ,);
    func_binary!(hypot         , hypot_f           , TensorHypotAPI            , DeviceHypotAPI            ,);
    func_binary!(log_add_exp   , log_add_exp_f     , TensorLogAddExpAPI        , DeviceLogAddExpAPI        ,);
    func_binary!(pow           , pow_f             , TensorPowAPI              , DevicePowAPI              ,);
    func_binary!(maximum       , maximum_f         , TensorMaximumAPI          , DeviceMaximumAPI          ,);
    func_binary!(minimum       , minimum_f         , TensorMinimumAPI          , DeviceMinimumAPI          ,);
    func_binary!(equal         , equal_f           , TensorEqualAPI            , DeviceEqualAPI            , eq, eq_f, equal_than      , equal_than_f      );
    func_binary!(less          , less_f            , TensorLessAPI             , DeviceLessAPI             , lt, lt_f, less_than       , less_than_f       );
    func_binary!(greater       , greater_f         , TensorGreaterAPI          , DeviceGreaterAPI          , gt, gt_f, greater_than    , greater_than_f    );
    func_binary!(less_equal    , less_equal_f      , TensorLessEqualAPI        , DeviceLessEqualAPI        , le, le_f, less_equal_to   , less_equal_to_f   );
    func_binary!(greater_equal , greater_equal_f   , TensorGreaterEqualAPI     , DeviceGreaterEqualAPI     , ge, ge_f, greater_equal_to, greater_equal_to_f);
    func_binary!(not_equal     , not_equal_f       , TensorNotEqualAPI         , DeviceNotEqualAPI         , ne, ne_f, not_equal_to    , not_equal_to_f    );
    func_binary!(nextafter     , nextafter_f       , TensorNextAfterAPI        , DeviceNextAfterAPI        ,);
    func_binary!(ext_add       , ext_add_f         , TensorExtAddAPI           , OpExtAddAPI               ,);
    func_binary!(ext_sub       , ext_sub_f         , TensorExtSubAPI           , OpExtSubAPI               ,);
    func_binary!(ext_mul       , ext_mul_f         , TensorExtMulAPI           , OpExtMulAPI               ,);
    func_binary!(ext_div       , ext_div_f         , TensorExtDivAPI           , OpExtDivAPI               ,);
    func_binary!(ext_bitand    , ext_bitand_f      , TensorExtBitAndAPI        , OpExtBitAndAPI            ,);
    func_binary!(ext_bitor     , ext_bitor_f       , TensorExtBitOrAPI         , OpExtBitOrAPI             ,);
    func_binary!(ext_bitxor    , ext_bitxor_f      , TensorExtBitXorAPI        , OpExtBitXorAPI            ,);
    func_binary!(ext_shl       , ext_shl_f         , TensorExtShlAPI           , OpExtShlAPI               ,);
    func_binary!(ext_shr       , ext_shr_f         , TensorExtShrAPI           , OpExtShrAPI               ,);
}

pub use func_binary::*;

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pow() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = arange(6u32).into_shape([2, 3]);
            let b = arange(3u32);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1, 1, 4, 1, 4, 25]);

            let a = arange(6.0).into_shape([2, 3]);

            let b = arange(3.0);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);

            let b = arange(3);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);
        }
        #[cfg(feature = "col_major")]
        {
            let a = arange(6u32).into_shape([3, 2]);
            let b = arange(3u32);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1, 1, 4, 1, 4, 25]);

            let a = arange(6.0).into_shape([3, 2]);

            let b = arange(3.0);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);

            let b = arange(3);
            let c = pow(&a, &b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![1.0, 1.0, 4.0, 1.0, 4.0, 25.0]);
        }
    }

    #[test]
    fn test_floor_divide() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = arange(6u32).into_shape([2, 3]); // u32
            let b = asarray(vec![1_i32, 2, 2]); // i32
            let c = a.floor_divide(&b); // i64
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0_i64, 0, 1, 3, 2, 2]);

            let a = arange(6.0).into_shape([2, 3]);

            let b = asarray(vec![1.0, 2.0, 2.0]);
            let c = a.floor_divide(&b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0.0, 0.0, 1.0, 3.0, 2.0, 2.0]);

            let b = asarray(vec![0.0, 2.0, 2.0]);
            let c = a.floor_divide_f(&b);
            println!("{c:?}");
        }
        #[cfg(feature = "col_major")]
        {
            // [3, 2] + [3]
            let a = arange(6u32).into_shape([3, 2]); // u32
            let b = asarray(vec![1_i32, 2, 2]); // i32
            let c = a.floor_divide(&b); // i64
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0_i64, 0, 1, 3, 2, 2]);

            let a = arange(6.0).into_shape([3, 2]);

            let b = asarray(vec![1.0, 2.0, 2.0]);
            let c = a.floor_divide(&b);
            println!("{c:?}");
            assert_eq!(c.reshape([6]).to_vec(), vec![0.0, 0.0, 1.0, 3.0, 2.0, 2.0]);

            let b = asarray(vec![0.0, 2.0, 2.0]);
            let c = a.floor_divide_f(&b);
            println!("{c:?}");
        }
    }

    #[test]
    fn test_ge_gt() {
        let a = asarray(vec![1., 2., 3., 4., 5., 6.]);
        let b = asarray(vec![1., 3., 2., 5., 5., 2.]);

        let c = gt(a.view(), &b);
        assert_eq!(c.raw(), &[false, false, true, false, false, true]);
        let c = ge(a.view(), &b);
        assert_eq!(c.raw(), &[true, false, true, false, true, true]);

        let c_sum = c.sum();
        assert_eq!(c_sum, 4);
    }

    #[test]
    fn test_refa_numb() {
        let a = asarray(vec![1., 3., 2., 5., 5., 2.]);
        let b = a.greater_equal(3.0);
        assert_eq!(b.raw(), &[false, true, false, true, true, false]);
        let b = a.pow(2);
        assert_eq!(b.raw(), &[1.0, 9.0, 4.0, 25.0, 25.0, 4.0]);
        let b = 2.0.pow(a.view());
        assert_eq!(b.raw(), &[2.0, 8.0, 4.0, 32.0, 32.0, 4.0]);
    }

    #[test]
    fn test_ext_arith() {
        // same type: promotion is the identity, so `ext_*` matches `+`
        let a = asarray(vec![1.0_f64, 2.0, 3.0]);
        let b = asarray(vec![10.0_f64, 20.0, 30.0]);
        assert_eq!(ext_add(&a, &b).raw(), &[11.0, 22.0, 33.0]);

        // mixed int x int: promote to the wider type
        let a = asarray(vec![1_i8, 2, 3]);
        let b = asarray(vec![10_i16, 20, 30]);
        assert_eq!(ext_add(&a, &b).raw(), &[11_i16, 22, 33]);

        // mixed signed x unsigned: promote to int16
        let a = asarray(vec![-1_i8, 2, 3]);
        let b = asarray(vec![1_u8, 2, 3]);
        assert_eq!(ext_add(&a, &b).raw(), &[0_i16, 4, 6]);

        // mixed float widths: promote to float64
        let a = asarray(vec![1.5_f32, 2.5]);
        let b = asarray(vec![1.0_f64, 1.0]);
        assert_eq!(ext_sub(&a, &b).raw(), &[0.5_f64, 1.5]);

        // method form + mixed multiply
        let a = asarray(vec![2_i8, 3, 4]);
        let b = asarray(vec![10_i32, 10, 10]);
        assert_eq!(a.ext_mul(&b).raw(), &[20_i32, 30, 40]);

        // float divide keeps the promoted float dtype
        let a = asarray(vec![1.0_f64, 2.0, 3.0]);
        let b = asarray(vec![2.0_f32, 4.0, 1.0]);
        assert_eq!(ext_div(&a, &b).raw(), &[0.5, 0.5, 3.0]);
    }

    #[test]
    fn test_ext_bitwise() {
        // mixed int widths: promote then bit-op
        let a = asarray(vec![0b1100_i8, 0b1010]);
        let b = asarray(vec![0b1010_i16, 0b0110]);
        assert_eq!(ext_bitand(&a, &b).raw(), &[0b1000_i16, 0b0010]);
        assert_eq!(ext_bitor(&a, &b).raw(), &[0b1110_i16, 0b1110]);
        assert_eq!(ext_bitxor(&a, &b).raw(), &[0b0110_i16, 0b1100]);

        // shifts promote the amount type as well
        let a = asarray(vec![1_u8, 2, 4]);
        let b = asarray(vec![1_i16, 1, 2]);
        assert_eq!(ext_shl(&a, &b).raw(), &[2_i16, 4, 16]);
        let c = asarray(vec![8_i16, 16, 32]);
        let d = asarray(vec![1_u8, 2, 3]);
        assert_eq!(ext_shr(&c, &d).raw(), &[4_i16, 4, 4]);

        // shift amount >= bit width: left yields 0, right saturates to the sign
        // (not Rust's masked `<<`/`>>`)
        let e = asarray(vec![1_i8, 1, -1]);
        let f = asarray(vec![8_i8, 9, 8]);
        assert_eq!(ext_shl(&e, &f).raw(), &[0_i8, 0, 0]);
        assert_eq!(ext_shr(&e, &f).raw(), &[0_i8, 0, -1]);

        // bool stays bool
        let a = asarray(vec![true, false, true]);
        let b = asarray(vec![false, false, true]);
        assert_eq!(ext_bitand(&a, &b).raw(), &[false, false, true]);
        assert_eq!(ext_bitor(&a, &b).raw(), &[true, false, true]);
    }
}
