use core::any::TypeId;

use crate::prelude_dev::*;
use rstsr_dtype_traits::ExtNum;

#[duplicate_item(
     OpAPI         Op       func                                  ;
    [OpAddAPI   ] [Add   ] [|c, a, b| { c.write(a.clone() +  b.clone()); }];
    [OpSubAPI   ] [Sub   ] [|c, a, b| { c.write(a.clone() -  b.clone()); }];
    [OpMulAPI   ] [Mul   ] [|c, a, b| { c.write(a.clone() *  b.clone()); }];
    [OpDivAPI   ] [Div   ] [|c, a, b| { c.write(a.clone() /  b.clone()); }];
    [OpBitOrAPI ] [BitOr ] [|c, a, b| { c.write(a.clone() |  b.clone()); }];
    [OpBitAndAPI] [BitAnd] [|c, a, b| { c.write(a.clone() &  b.clone()); }];
    [OpBitXorAPI] [BitXor] [|c, a, b| { c.write(a.clone() ^  b.clone()); }];
    [OpShlAPI   ] [Shl   ] [|c, a, b| { c.write(a.clone() << b.clone()); }];
    [OpShrAPI   ] [Shr   ] [|c, a, b| { c.write(a.clone() >> b.clone()); }];
)]
impl<TA, TB, TC, D> OpAPI<TA, TB, TC, D> for DeviceCpuSerial
where
    TA: Clone + Op<TB, Output = TC>,
    TB: Clone,
    TC: Clone,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut func)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut func)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut func)
    }
}

/// `rem` fast-path dtypes for the serial device: the array-API floored
/// remainder (`ExtReal::ext_rem`) is applied to these via a `TypeId` guard;
/// every other `Rem` dtype (integers, ...) keeps Rust's C-style `%`. Mirrors
/// the `half`-inclusive list in `set.rs`.
macro_rules! for_each_rem_float {
    ($mac:ident) => {
        $mac!(f32);
        $mac!(f64);
        $mac!(half::f16);
        $mac!(half::bf16);
    };
}

// `rem` is specialized instead of using the generic table above: keep the
// `Rem` bound (so common `Rem` dtypes still take Rust's `%`), but route the
// float dtypes to `ext_rem` (floored / sign-of-divisor, plus the float special
// cases). The `rem`→`Rem` operator binding is deliberately commented out at the
// tensor layer (`op_binary_arithmetic.rs`), so this only backs the `rt::rem`
// family.
impl<TA, TB, TC, D> OpRemAPI<TA, TB, TC, D> for DeviceCpuSerial
where
    TA: Clone + Rem<TB, Output = TC> + 'static,
    TB: Clone + 'static,
    TC: Clone + 'static,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        macro_rules! rem_refb {
            ($ty:ty) => {{
                if TypeId::of::<TA>() == TypeId::of::<$ty>()
                    && TypeId::of::<TB>() == TypeId::of::<$ty>()
                    && TypeId::of::<TC>() == TypeId::of::<$ty>()
                {
                    // SAFETY: `TypeId` equality proves `TA == TB == TC == $ty`; the
                    // re-typed raw Vecs address the same dtype-independent layout.
                    let c = unsafe { &mut *(c as *mut Vec<MaybeUninit<TC>> as *mut Vec<MaybeUninit<$ty>>) };
                    let a = unsafe { &*(a as *const Vec<TA> as *const Vec<$ty>) };
                    let b = unsafe { &*(b as *const Vec<TB> as *const Vec<$ty>) };
                    return self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
                        c.write(a.clone().ext_rem(b.clone()));
                    });
                }
            }};
        }
        for_each_rem_float!(rem_refb);
        self.op_mutc_refa_refb_func(c, lc, a, la, b, lb, &mut |c, a, b| {
            c.write(a.clone() % b.clone());
        })
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        macro_rules! rem_numb {
            ($ty:ty) => {{
                if TypeId::of::<TA>() == TypeId::of::<$ty>()
                    && TypeId::of::<TB>() == TypeId::of::<$ty>()
                    && TypeId::of::<TC>() == TypeId::of::<$ty>()
                {
                    // SAFETY: `TypeId` equality proves `TA == TB == TC == $ty`; `b`
                    // is a scalar of that same dtype (re-read as `$ty`).
                    let c = unsafe { &mut *(c as *mut Vec<MaybeUninit<TC>> as *mut Vec<MaybeUninit<$ty>>) };
                    let a = unsafe { &*(a as *const Vec<TA> as *const Vec<$ty>) };
                    let b = unsafe { *(&b as *const TB as *const $ty) };
                    return self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut |c, a, b| {
                        c.write(a.clone().ext_rem(b.clone()));
                    });
                }
            }};
        }
        for_each_rem_float!(rem_numb);
        self.op_mutc_refa_numb_func(c, lc, a, la, b, &mut |c, a, b| {
            c.write(a.clone() % b.clone());
        })
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &Vec<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        macro_rules! rem_numa {
            ($ty:ty) => {{
                if TypeId::of::<TA>() == TypeId::of::<$ty>()
                    && TypeId::of::<TB>() == TypeId::of::<$ty>()
                    && TypeId::of::<TC>() == TypeId::of::<$ty>()
                {
                    // SAFETY: `TypeId` equality proves `TA == TB == TC == $ty`; `a`
                    // is a scalar of that same dtype (re-read as `$ty`).
                    let c = unsafe { &mut *(c as *mut Vec<MaybeUninit<TC>> as *mut Vec<MaybeUninit<$ty>>) };
                    let a = unsafe { *(&a as *const TA as *const $ty) };
                    let b = unsafe { &*(b as *const Vec<TB> as *const Vec<$ty>) };
                    return self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut |c, a, b| {
                        c.write(a.clone().ext_rem(b.clone()));
                    });
                }
            }};
        }
        for_each_rem_float!(rem_numa);
        self.op_mutc_numa_refb_func(c, lc, a, b, lb, &mut |c, a, b| {
            c.write(a.clone() % b.clone());
        })
    }
}
