use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use rstsr_dtype_traits::{DTypeIntoFloatAPI, ExtComplexFloat, ExtNum, ExtReal};

/* #region same type */

#[duplicate_item(
     OpAPI             NumTrait       func_inner;
    [OpAcosAPI      ] [ExtComplexFloat] [b.ext_acos()  ];
    [OpAcoshAPI     ] [ExtComplexFloat] [b.ext_acosh() ];
    [OpAsinAPI      ] [ExtComplexFloat] [b.ext_asin()  ];
    [OpAsinhAPI     ] [ExtComplexFloat] [b.ext_asinh() ];
    [OpAtanAPI      ] [ComplexFloat] [b.atan()  ];
    [OpAtanhAPI     ] [ExtComplexFloat] [b.ext_atanh() ];
    [OpConjAPI      ] [ComplexFloat] [b.conj()  ];
    [OpCosAPI       ] [ComplexFloat] [b.cos()   ];
    [OpCoshAPI      ] [ExtComplexFloat] [b.ext_cosh()  ];
    [OpExpAPI       ] [ComplexFloat] [b.exp()   ];
    [OpExpm1API     ] [ExtComplexFloat] [b.ext_exp_m1()];
    [OpInvAPI       ] [ComplexFloat] [b.recip() ];
    [OpLogAPI       ] [ComplexFloat] [b.ln()    ];
    [OpLog1pAPI     ] [ExtComplexFloat] [b.ext_log_1p() ];
    [OpLog2API      ] [ComplexFloat] [b.log2()  ];
    [OpLog10API     ] [ComplexFloat] [b.log10() ];
    [OpReciprocalAPI] [ComplexFloat] [b.recip() ];
    [OpSinAPI       ] [ComplexFloat] [b.sin()   ];
    [OpSinhAPI      ] [ExtComplexFloat] [b.ext_sinh()  ];
    [OpSqrtAPI      ] [ExtComplexFloat] [b.ext_sqrt()  ];
    [OpTanAPI       ] [ExtComplexFloat] [b.ext_tan()   ];
    [OpTanhAPI      ] [ExtComplexFloat] [b.ext_tanh()  ];
)]
impl<T, D> OpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + DTypeIntoFloatAPI<FloatType: NumTrait + Send + Sync>,
    D: DimAPI,
{
    type TOut = T::FloatType;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<Self::TOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>, b: &T| {
            let b = b.clone().into_float();
            a.write(func_inner);
        };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<Self::TOut>>, la: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>| {
            // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
            // read then overwritten via `write`.
            let b = unsafe { a.assume_init_read() };
            a.write(func_inner);
        };
        self.op_muta_func(a, la, &mut func)
    }
}

// dtype-preserving rounding (integers are already integral)
#[duplicate_item(
     OpAPI           func_inner;
    [OpCeilAPI   ] [ExtReal::ext_ceil(b.clone())  ];
    [OpFloorAPI  ] [ExtReal::ext_floor(b.clone()) ];
    [OpTruncAPI  ] [ExtReal::ext_trunc(b.clone()) ];
)]
impl<T, D> OpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ExtReal,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<Self::TOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>, b: &T| {
            a.write(func_inner);
        };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<Self::TOut>>, la: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>| {
            // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
            // read then overwritten via `write`.
            let b = unsafe { a.assume_init_read() };
            a.write(func_inner);
        };
        self.op_muta_func(a, la, &mut func)
    }
}

// `round` also covers complex: real and imaginary parts rounded independently
impl<T, D> OpRoundAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ExtNum,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<Self::TOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>, b: &T| {
            a.write(b.clone().ext_round());
        };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<Self::TOut>>, la: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<Self::TOut>| {
            // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
            // read then overwritten via `write`.
            let b = unsafe { a.assume_init_read() };
            a.write(b.clone().ext_round());
        };
        self.op_muta_func(a, la, &mut func)
    }
}

// NumPy-style unary minus: covers unsigned dtypes (two's complement wrap)
impl<T, D> OpExtNegAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + ExtNum,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<Self::TOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_neg());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<Self::TOut>>, la: &Layout<D>) -> Result<()> {
        self.op_muta_func(a, la, &mut |a| unsafe {
            // SAFETY: in-place op — reads an initialized element, then overwrites it via `write`.
            a.write(a.assume_init_read().ext_neg());
        })
    }
}

impl<T, D> OpSquareAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + Send + Sync + Mul<Output = T>,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<T>, b: &T| {
            a.write(b.clone() * b.clone());
        };
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let mut func = |a: &mut MaybeUninit<T>| {
            // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
            // read then overwritten via `write`.
            let b = unsafe { a.assume_init_read() };
            a.write(b.clone() * b);
        };
        self.op_muta_func(a, la, &mut func)
    }
}

/* #endregion */

/* #region boolean output */

#[duplicate_item(
     OpAPI           NumTrait       func                         ;
    [OpIsFiniteAPI] [ComplexFloat] [|a, b| { a.write(b.is_finite()  ); } ];
    [OpIsInfAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_infinite()); } ];
    [OpIsNanAPI   ] [ComplexFloat] [|a, b| { a.write(b.is_nan()     ); } ];
)]
impl<T, D> OpAPI<T, D> for DeviceRayonAutoImpl
where
    T: Clone + NumTrait + Send + Sync,
    D: DimAPI,
{
    type TOut = bool;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<bool>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut func)
    }

    fn op_muta(&self, _a: &mut Vec<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<T>();
        unreachable!("{:?} is not supported in this function.", type_b);
    }
}

/* #endregion */

/* #region signbit */

impl<T, D> OpSignBitAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtReal + Send + Sync,
    D: DimAPI,
{
    type TOut = bool;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<bool>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_signbit());
        })
    }

    fn op_muta(&self, _a: &mut Vec<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
        let type_b = core::any::type_name::<T>();
        unreachable!("{:?} is not supported in this function.", type_b);
    }
}

/* #endregion */

/* #region complex specific implementation */

impl<T, D> OpAbsAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    T::AbsOut: Send + Sync,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_abs());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if T::ABS_UNCHANGED {
            return Ok(());
        } else if T::ABS_SAME_TYPE {
            // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
            // read then overwritten via `write`.
            return self.op_muta_func(a, la, &mut |a| unsafe {
                // SAFETY: in-place op — reads an initialized element, then overwrites it via
                // `write`.
                a.write(a.assume_init_read().ext_abs());
            });
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> OpImagAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    T::AbsOut: Send + Sync,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_imag());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if T::ABS_SAME_TYPE {
            // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
            // read then overwritten via `write`.
            return self.op_muta_func(a, la, &mut |a| unsafe {
                // SAFETY: in-place op — reads an initialized element, then overwrites it via
                // `write`.
                a.write(a.assume_init_read().ext_imag());
            });
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> OpRealAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    T::AbsOut: Send + Sync,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut Vec<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &Vec<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_real());
        })
    }

    fn op_muta(&self, _a: &mut Vec<MaybeUninit<T::AbsOut>>, _la: &Layout<D>) -> Result<()> {
        if T::ABS_SAME_TYPE {
            return Ok(());
        } else {
            let type_b = core::any::type_name::<T>();
            unreachable!("{:?} is not supported in this function.", type_b);
        }
    }
}

impl<T, D> OpSignAPI<T, D> for DeviceRayonAutoImpl
where
    T: ExtNum + Send + Sync,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>, b: &Vec<T>, lb: &Layout<D>) -> Result<()> {
        self.op_muta_refb_func(a, la, b, lb, &mut |a, b| {
            a.write(b.clone().ext_sign());
        })
    }

    fn op_muta(&self, a: &mut Vec<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        // SAFETY: in-place op — `a` is an initialized element of the caller's buffer;
        // read then overwritten via `write`.
        self.op_muta_func(a, la, &mut |a| unsafe {
            // SAFETY: in-place op — reads an initialized element, then overwrites it via
            // `write`.
            a.write(a.assume_init_read().ext_sign());
        })
    }
}

/* #endregion */
