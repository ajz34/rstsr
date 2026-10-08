use crate::prelude_dev::*;
use core::sync::atomic::{AtomicBool, Ordering};
use num::complex::ComplexFloat;
use num::Float;
use rstsr_dtype_traits::{DTypeIntoFloatAPI, DTypePromoteAPI, ExtFloat, ExtNum, ExtReal};

// output with special promotion
#[duplicate_item(
     OpAPI               TraitT           func_inner;
    [OpATan2API       ] [Float         ] [Float::atan2(a, b)            ];
    [OpCopySignAPI    ] [Float         ] [Float::copysign(a, b)         ];
    [OpHypotAPI       ] [Float         ] [Float::hypot(a, b)            ];
    [OpNextAfterAPI   ] [ExtFloat      ] [ExtFloat::ext_nextafter(a, b) ];
    [OpLogAddExpAPI   ] [ComplexFloat  ] [(a.exp() + b.exp()).ln()      ];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB, Res: DTypeIntoFloatAPI<FloatType: TraitT + Send + Sync>>,
    TB: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = <TA::Res as DTypeIntoFloatAPI>::FloatType;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let (a, b) = (a.into_float(), b.into_float());
            c.write(func_inner);
        };
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let (a, b) = (a.into_float(), b.into_float());
            c.write(func_inner);
        };
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let (a, b) = (a.into_float(), b.into_float());
            c.write(func_inner);
        };
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}

// general promotion
#[duplicate_item(
     OpAPI               TO        TraitT           func_inner;
    [OpMaximumAPI     ] [TA::Res] [ExtReal       ] [ExtReal::ext_max(a, b)         ];
    [OpMinimumAPI     ] [TA::Res] [ExtReal       ] [ExtReal::ext_min(a, b)         ];
    [OpFloorDivideAPI ] [TA::Res] [ExtReal       ] [ExtReal::ext_floor_divide(a, b)];
    [OpEqualAPI       ] [bool   ] [PartialEq     ] [a == b                         ];
    [OpNotEqualAPI    ] [bool   ] [PartialEq     ] [a != b                         ];
    [OpGreaterAPI     ] [bool   ] [PartialOrd    ] [a > b                          ];
    [OpGreaterEqualAPI] [bool   ] [PartialOrd    ] [a >= b                         ];
    [OpLessAPI        ] [bool   ] [PartialOrd    ] [a < b                          ];
    [OpLessEqualAPI   ] [bool   ] [PartialOrd    ] [a <= b                         ];
    [OpExtAddAPI      ] [TA::Res] [Clone + Add<Output = TA::Res>] [a + b];
    [OpExtSubAPI      ] [TA::Res] [Clone + Sub<Output = TA::Res>] [a - b];
    [OpExtMulAPI      ] [TA::Res] [Clone + Mul<Output = TA::Res>] [a * b];
    [OpExtDivAPI      ] [TA::Res] [Clone + Div<Output = TA::Res>] [a / b];
    [OpExtBitAndAPI   ] [TA::Res] [Clone + BitAnd<Output = TA::Res>] [a & b];
    [OpExtBitOrAPI    ] [TA::Res] [Clone + BitOr<Output = TA::Res>]  [a | b];
    [OpExtBitXorAPI   ] [TA::Res] [Clone + BitXor<Output = TA::Res>] [a ^ b];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB, Res: TraitT + Send + Sync>,
    TB: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = TO;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            c.write(func_inner);
        };
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            c.write(func_inner);
        };
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |c: &mut MaybeUninit<Self::TOut>, a: &TA, b: &TB| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            c.write(func_inner);
        };
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}

// pow: the output is the promoted type, and the kernel is `ExtNum::ext_pow`
impl<TA, TB, D> OpPowAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB>,
    TB: Clone + Send + Sync,
    TA::Res: ExtNum + Send + Sync,
    D: DimAPI,
{
    type TOut = TA::Res;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let bad_exp = AtomicBool::new(false);
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            match a.ext_pow(b) {
                Some(v) => {
                    c.write(v);
                },
                None => {
                    bad_exp.store(true, Ordering::Relaxed);
                },
            }
        })?;
        if bad_exp.load(Ordering::Relaxed) {
            return Err(rstsr_error!(InvalidValue, "integer power requires a non-negative exponent"));
        }
        Ok(())
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let bad_exp = AtomicBool::new(false);
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut |c, a, b| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            match a.ext_pow(b) {
                Some(v) => {
                    c.write(v);
                },
                None => {
                    bad_exp.store(true, Ordering::Relaxed);
                },
            }
        })?;
        if bad_exp.load(Ordering::Relaxed) {
            return Err(rstsr_error!(InvalidValue, "integer power requires a non-negative exponent"));
        }
        Ok(())
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        lc: &Layout<D>,
        a: TA,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()> {
        let bad_exp = AtomicBool::new(false);
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut |c, a, b| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            match a.ext_pow(b) {
                Some(v) => {
                    c.write(v);
                },
                None => {
                    bad_exp.store(true, Ordering::Relaxed);
                },
            }
        })?;
        if bad_exp.load(Ordering::Relaxed) {
            return Err(rstsr_error!(InvalidValue, "integer power requires a non-negative exponent"));
        }
        Ok(())
    }
}

// `ext_shl` / `ext_shr`: the array-API shift rule differs from Rust's `<<`/`>>`
// when the shift amount reaches the bit width (left shift yields 0; right shift
// saturates to the sign), so these two are implemented separately from the
// promotion table above. `PrimInt` also confines the bound to integer types.
#[duplicate_item(
     OpAPI           shift   ;
    [OpExtShlAPI] [{ let sh = num::ToPrimitive::to_usize(&b).unwrap_or(usize::MAX); if sh >= bits { <Self::TOut as num::Zero>::zero() } else { a << sh } }];
    [OpExtShrAPI] [{ let sh = num::ToPrimitive::to_usize(&b).unwrap_or(usize::MAX).min(bits - 1); a >> sh }];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceRayonAutoImpl
where
    TA: Clone + Send + Sync + DTypePromoteAPI<TB>,
    <TA as DTypePromoteAPI<TB>>::Res: Clone + Send + Sync + num::PrimInt,
    TB: Clone + Send + Sync,
    D: DimAPI,
{
    type TOut = <TA as DTypePromoteAPI<TB>>::Res;

    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let bits = core::mem::size_of::<Self::TOut>() * 8;
            c.write(shift);
        })
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut |c, a, b| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let bits = core::mem::size_of::<Self::TOut>() * 8;
            c.write(shift);
        })
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<Self::TOut>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut |c, a, b| {
            let (a, b) = TA::promote_pair(a.clone(), b.clone());
            let bits = core::mem::size_of::<Self::TOut>() * 8;
            c.write(shift);
        })
    }
}
