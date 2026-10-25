use crate::prelude_dev::*;
use core::ops::*;
use num::{Float, FromPrimitive, ToPrimitive};

/* #region arange */

pub fn arange_by_partial_ord_cpu_serial<T>(start: T, end: T, step: T) -> Vec<T>
where
    T: PartialOrd + Clone + Add<Output = T>,
{
    // This is serial implementation and low performance. The step must point
    // from `start` towards `end`; sign-mismatched ranges are empty (numpy
    // semantics) and the direction-aware condition keeps the loop bounded.
    let up = start < end;
    let mut result = Vec::new();
    let mut current = start;
    loop {
        let next = current.clone() + step.clone();
        let inside = match up {
            true => current < end && current < next,
            false => current > end && current > next,
        };
        if !inside {
            break;
        }
        result.push(current);
        current = next;
    }
    result
}

pub fn arange_by_primitive_isize_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive,
{
    // This will use rust's internal iterator, should be much faster.
    // However, it only works for types that from/to primitives implemented.
    // Also, it only works for integer types because of the precision issue.

    // We just try to convert to isize. We believe it's really rare case that usize is used in
    // arange; and anyway, fallback is always available with efficiency loss.
    let (start_, end_, step_) = (start.to_isize()?, end.to_isize()?, step.to_isize()?);
    let n = Float::ceil((end_ - start_) as f64 / step_ as f64).to_isize()?;

    // The unwrap here is probably safe, since start/end/nstep are all checked.
    // `Some(map(option.unwrap).collect())` is much faster than `map(option).collect()` in this
    // case, so decided unwrap here.
    let mut result: Vec<T> = (0..n).map(|i| T::from_isize(start_ + i * step_).unwrap()).collect();

    // the interval may be open on the right, so we need to pop the last element if it's out of
    // range.
    let last_val = result.last().cloned();
    if (step_ > 0 && last_val.as_ref().is_some_and(|x| *x >= end))
        || (step_ < 0 && last_val.as_ref().is_some_and(|x| *x <= end))
    {
        result.pop();
    }
    Some(result)
}

pub fn arange_by_primitive_f64_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive,
{
    // NumPy `_calc_length`: ceil((end - start) / step) with no trailing pop
    // (the endpoint stays when the arithmetic lands exactly on it)
    let (start_, end_, step_) = (start.to_f64()?, end.to_f64()?, step.to_f64()?);
    let span = (end_ - start_) / step_;
    if span <= 0.0 {
        return Some(Vec::new());
    }
    let n = Float::ceil(span).to_usize()?;
    // NumPy fills from the second element: `delta = (start + step) - start`;
    // slots 0/1 are `start` and `start + step` verbatim
    let next = start_ + step_;
    let delta = next - start_;
    // branch-free fill (vectorizable); the formula is not exact at the first
    // two slots, so they are pinned afterwards
    let mut result: Vec<T> = (0..n).map(|i| T::from_f64(start_ + i as f64 * delta).unwrap()).collect();
    result[0] = T::from_f64(start_).unwrap();
    if n > 1 {
        result[1] = T::from_f64(next).unwrap();
    }
    Some(result)
}

/// `f32` variant: the length and the elements are computed in `f32` (NumPy
/// computes `_calc_length` in the scalar dtype; the `f64` path would round
/// `1.0 / 0.01f32` to 101 elements instead of 100).
pub fn arange_by_primitive_f32_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive,
{
    let (start_, end_, step_) = (start.to_f32()?, end.to_f32()?, step.to_f32()?);
    let span = (end_ - start_) / step_;
    if span <= 0.0 {
        return Some(Vec::new());
    }
    let n = span.ceil().to_usize()?;
    let next = start_ + step_;
    let delta = next - start_;
    let mut result: Vec<T> = (0..n).map(|i| T::from_f32(start_ + i as f32 * delta).unwrap()).collect();
    result[0] = T::from_f32(start_).unwrap();
    if n > 1 {
        result[1] = T::from_f32(next).unwrap();
    }
    Some(result)
}

/// Half-width variant: mirrors NumPy `HALF_fill` (the quotient is rounded in
/// the scalar dtype, the elements are evaluated in `f32` with the
/// second-element delta).
pub fn arange_by_primitive_half_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + Sub<Output = T> + Div<Output = T> + ToPrimitive + FromPrimitive,
{
    let span = ((end - start.clone()) / step.clone()).to_f32()?;
    if span <= 0.0 {
        return Some(Vec::new());
    }
    let n = span.ceil().to_usize()?;
    let start_ = start.to_f32()?;
    let next = start.clone() + step;
    let delta = next.to_f32()? - start_;
    let mut result: Vec<T> = (0..n).map(|i| T::from_f32(start_ + i as f32 * delta).unwrap()).collect();
    result[0] = start;
    if n > 1 {
        result[1] = next;
    }
    Some(result)
}

/// Create a 1-D array of evenly spaced values within a given interval, using CPU serial
/// implementation.
///
/// Some numerical types (e.g., isize, usize) will be accelerated using internal iterators, while
/// others will use a generic implementation.
pub fn arange_cpu_serial<T>(start: T, end: T, step: T) -> Vec<T>
where
    T: PartialOrd + Clone + Add<Output = T> + 'static,
{
    use core::any::TypeId;
    use core::mem::*;

    #[derive(Clone, Copy)]
    enum ArangeMode {
        Int,
        F64,
        F32,
        Half,
    }

    // 1. transmute type without affecting input/output types
    #[inline]
    // SAFETY: `TypeId` equality (checked at the call site) proves `T == U`, so the
    // `transmute_copy` of the scalars and the final `transmute::<Vec<U>, Vec<T>>`
    // are no-op reinterpretations of identically-typed data.
    unsafe fn transmute_accelerated<T, U>(start: &T, end: &T, step: &T, mode: ArangeMode) -> Option<Vec<T>>
    where
        U: PartialOrd + Clone + Add<Output = U> + Sub<Output = U> + Div<Output = U> + ToPrimitive + FromPrimitive,
    {
        let (start, end, step) =
            (transmute_copy::<T, U>(start), transmute_copy::<T, U>(end), transmute_copy::<T, U>(step));
        let v = match mode {
            ArangeMode::Int => arange_by_primitive_isize_cpu_serial(start, end, step),
            ArangeMode::F64 => arange_by_primitive_f64_cpu_serial(start, end, step),
            ArangeMode::F32 => arange_by_primitive_f32_cpu_serial(start, end, step),
            ArangeMode::Half => arange_by_primitive_half_cpu_serial(start, end, step),
        };
        v.map(|result| transmute::<Vec<U>, Vec<T>>(result))
    }

    // 2. convenient macro in match definition
    macro_rules! expand_accelerated { ($mode: expr, $($ty: ty),*) => {
        match TypeId::of::<T>() {
            $(id if id == TypeId::of::<$ty>() => unsafe { transmute_accelerated::<T, $ty>(&start, &end, &step, $mode) },)*
            _ => None
        }
    }}

    // 3. execute accelerated implementation if possible (float widths keep
    // their own arithmetic width, matching NumPy's length computation)
    expand_accelerated!(ArangeMode::F64, f64)
        .or_else(|| expand_accelerated!(ArangeMode::F32, f32))
        .or_else(|| expand_accelerated!(ArangeMode::Half, half::f16, half::bf16))
        .unwrap_or_else(|| {
            expand_accelerated!(ArangeMode::Int, usize, isize, i8, i16, i32, i64, u8, u16, u32, u64)
                .unwrap_or_else(|| arange_by_partial_ord_cpu_serial(start, end, step))
        })
}

/* #endregion */
