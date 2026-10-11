use crate::prelude_dev::*;

pub trait DeviceVecdotAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    fn vecdot(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()>;
}

pub trait DeviceOuterAPI<TA, TB, TC>
where
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    /// Outer product of two **one-dimensional** arrays, writing
    /// `c[i, j] = a[i] * b[j]` into the uninitialized `(N, M)` output. No
    /// conjugation. Implementations must initialize every element of `lc`.
    fn outer(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<Ix2>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<Ix1>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<Ix1>,
    ) -> Result<()>;
}

pub trait DeviceCrossAPI<TA, TB, TC, D>
where
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    /// Cross product of 3-element vectors along the (non-negative) axis `axis`,
    /// writing into the uninitialized output `c`. All three layouts share one
    /// shape whose `axis` is 3; component `k` is
    /// `a[.., k+1, ..] * b[.., k+2, ..] - a[.., k+2, ..] * b[.., k+1, ..]`
    /// (cyclically). No conjugation. Implementations must initialize every
    /// element of `lc`.
    fn cross(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<D>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        axis: usize,
    ) -> Result<()>;
}

pub trait DeviceTensordotAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    /// Generalized tensor contraction: contract `axes_a` of `a` with `axes_b`
    /// of `b`, writing the non-contracted axes (a then b, original order) into
    /// the `MaybeUninit` output `c`. No conjugation. `axes_a`/`axes_b` are
    /// already normalized, non-negative and pairwise aligned.
    fn tensordot(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw,
        lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()>;
}
