use crate::prelude_dev::*;
use core::ops::*;
use num::{complex::ComplexFloat, FromPrimitive, ToPrimitive};

/* #region arange */

/// NOTE: this implementation does not involve a thread pool.
pub fn arange_by_primitive_isize_cpu_rayon<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive + Send + Sync,
{
    // This will use rust's internal iterator, should be much faster.
    // However, it only works for types that from/to primitives implemented.
    // Also, it only works for integer types because of the precision issue.

    // We just try to convert to isize. We believe it's really rare case that usize is used in
    // arange; and anyway, fallback is always available with efficiency loss.
    let (start_, end_, step_) = (start.to_isize()?, end.to_isize()?, step.to_isize()?);
    let n = ((end_ - start_) as f64 / step_ as f64).ceil().to_isize()?;

    // The unwrap here is probably safe, since start/end/nstep are all checked.
    // `Some(map(option.unwrap).collect())` is much faster than `map(option).collect()` in this
    // case, so decided unwrap here.
    let mut result: Vec<T> = (0..n).into_par_iter().map(|i| T::from_isize(start_ + i * step_).unwrap()).collect();

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

/// NOTE: this implementation does not involve a thread pool.
pub fn arange_by_primitive_f64_cpu_rayon<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive + Send + Sync,
{
    // NumPy `_calc_length`: no trailing pop (the endpoint stays when the
    // arithmetic lands exactly on it)
    let (start_, end_, step_) = (start.to_f64()?, end.to_f64()?, step.to_f64()?);
    let span = (end_ - start_) / step_;
    if span <= 0.0 {
        return Some(Vec::new());
    }
    let n = span.ceil().to_usize()?;
    // NumPy fills from the second element: `delta = (start + step) - start`;
    // slots 0/1 are `start` and `start + step` verbatim
    let next = start_ + step_;
    let delta = next - start_;
    let mut result: Vec<T> = (0..n).into_par_iter().map(|i| T::from_f64(start_ + i as f64 * delta).unwrap()).collect();
    result[0] = T::from_f64(start_).unwrap();
    if n > 1 {
        result[1] = T::from_f64(next).unwrap();
    }
    Some(result)
}

/// `f32` variant: length and elements computed in `f32` (NumPy's
/// `_calc_length` runs in the scalar dtype).
pub fn arange_by_primitive_f32_cpu_rayon<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive + Send + Sync,
{
    let (start_, end_, step_) = (start.to_f32()?, end.to_f32()?, step.to_f32()?);
    let span = (end_ - start_) / step_;
    if span <= 0.0 {
        return Some(Vec::new());
    }
    let n = span.ceil().to_usize()?;
    let next = start_ + step_;
    let delta = next - start_;
    let mut result: Vec<T> = (0..n).into_par_iter().map(|i| T::from_f32(start_ + i as f32 * delta).unwrap()).collect();
    result[0] = T::from_f32(start_).unwrap();
    if n > 1 {
        result[1] = T::from_f32(next).unwrap();
    }
    Some(result)
}

/// Half-width variant: mirrors NumPy `HALF_fill` (the quotient is rounded in
/// the scalar dtype, the elements are evaluated in `f32` with the
/// second-element delta).
pub fn arange_by_primitive_half_cpu_rayon<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync,
{
    let span = ((end - start.clone()) / step.clone()).to_f32()?;
    if span <= 0.0 {
        return Some(Vec::new());
    }
    let n = span.ceil().to_usize()?;
    let start_ = start.to_f32()?;
    let next = start.clone() + step;
    let delta = next.to_f32()? - start_;
    let mut result: Vec<T> = (0..n).into_par_iter().map(|i| T::from_f32(start_ + i as f32 * delta).unwrap()).collect();
    result[0] = start;
    if n > 1 {
        result[1] = next;
    }
    Some(result)
}

/// Create a 1-D array of evenly spaced values within a given interval, using CPU rayon
/// implementation.
///
/// Some numerical types (e.g., isize, usize) will be accelerated using internal iterators, while
/// others will use a generic implementation.
pub fn arange_cpu_rayon<T>(start: T, end: T, step: T, pool: Option<&ThreadPool>) -> Vec<T>
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
    unsafe fn transmute_accelerated<T, U>(
        start: &T,
        end: &T,
        step: &T,
        mode: ArangeMode,
        pool: Option<&ThreadPool>,
    ) -> Option<Vec<T>>
    where
        U: PartialOrd
            + Clone
            + Add<Output = U>
            + Sub<Output = U>
            + Div<Output = U>
            + ToPrimitive
            + FromPrimitive
            + Send
            + Sync,
    {
        let (start, end, step) =
            (transmute_copy::<T, U>(start), transmute_copy::<T, U>(end), transmute_copy::<T, U>(step));
        let task = || match mode {
            ArangeMode::Int => arange_by_primitive_isize_cpu_rayon(start.clone(), end.clone(), step.clone()),
            ArangeMode::F64 => arange_by_primitive_f64_cpu_rayon(start.clone(), end.clone(), step.clone()),
            ArangeMode::F32 => arange_by_primitive_f32_cpu_rayon(start.clone(), end.clone(), step.clone()),
            ArangeMode::Half => arange_by_primitive_half_cpu_rayon(start.clone(), end.clone(), step.clone()),
        };
        pool.map_or_else(task, |pool| pool.install(task)).map(|result| transmute::<Vec<U>, Vec<T>>(result))
    }

    // 2. convenient macro in match definition
    macro_rules! expand_accelerated { ($mode: expr, $($ty: ty),*) => {
        match TypeId::of::<T>() {
            $(id if id == TypeId::of::<$ty>() => unsafe { transmute_accelerated::<T, $ty>(&start, &end, &step, $mode, pool) },)*
            _ => None
        }
    }}

    // 3. execute accelerated implementation if possible
    expand_accelerated!(ArangeMode::F64, f64)
        .or_else(|| expand_accelerated!(ArangeMode::F32, f32))
        .or_else(|| expand_accelerated!(ArangeMode::Half, half::f16, half::bf16))
        .unwrap_or_else(|| {
            expand_accelerated!(ArangeMode::Int, usize, isize, i8, i16, i32, i64, u8, u16, u32, u64)
                .unwrap_or_else(|| arange_by_partial_ord_cpu_serial(start, end, step))
        })
}

/* #endregion */

/* #region linspace */

pub fn linspace_cpu_rayon<T>(start: T, end: T, n: usize, endpoint: bool, pool: Option<&ThreadPool>) -> Option<Vec<T>>
where
    T: ComplexFloat + Clone + Send + Sync,
{
    // handle special cases
    if n == 0 {
        return Some(vec![]);
    } else if n == 1 {
        return Some(vec![start]);
    }

    // step should be usually safe to unwrap, since usize should be convertible to float for most
    // cases, though I'm not sure if FP8 or even smaller types are supported.
    let step = match endpoint {
        true => (end - start) / T::from(n - 1)?,
        false => (end - start) / T::from(n)?,
    };

    // unwrap should be safe here: if `n` can be converted, this can surely be converted by `i`.
    let task = || -> Vec<T> {
        let mut out: Vec<T> = (0..n).into_par_iter().map(|i| start + T::from(i).unwrap() * step).collect();
        if endpoint {
            // endpoint included exactly, not as `start + (n - 1) * step`
            // (which rounds to a neighbor); as in NumPy for float64
            out[n - 1] = end;
        }
        out
    };
    Some(pool.map_or_else(task, |pool| pool.install(task)))
}

/* #endregion */
