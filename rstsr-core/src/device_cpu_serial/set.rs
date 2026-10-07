//! Set-operation device impls for [`DeviceCpuSerial`]: general `PartialEq`
//! paths, with the sorted fast path dispatched per scalar dtype (TypeId +
//! re-typed raw buffers; the matmul-dispatch technique).

use core::any::TypeId;

use num::Complex;

use crate::prelude_dev::*;

/// Expand `$mac!` once per fast-path scalar dtype (the sorted-path
/// candidates). This rstsr-core copy additionally lists the `half` dtypes;
/// the symlinked `auto_impl::set` copy cannot name them (the device crates do
/// not depend on the `half` crate), so the two lists are kept in sync by
/// omission there.
macro_rules! for_each_fast_dtype {
    ($mac:ident) => {
        $mac!(f32);
        $mac!(f64);
        $mac!(bool);
        $mac!(i8);
        $mac!(i16);
        $mac!(i32);
        $mac!(i64);
        $mac!(isize);
        $mac!(u8);
        $mac!(u16);
        $mac!(u32);
        $mac!(u64);
        $mac!(usize);
        $mac!(i128);
        $mac!(u128);
        $mac!(half::f16);
        $mac!(half::bf16);
    };
}

impl<T, D> OpUniqueAPI<T, D> for DeviceCpuSerial
where
    T: Clone + PartialEq + 'static,
    D: DimAPI,
{
    fn unique_values(&self, a: &Vec<T>, la: &Layout<D>, values: &mut Vec<MaybeUninit<T>>) -> Result<usize> {
        // sorted fast path per TypeId-verified scalar dtype; anything else
        // (e.g. complex — its sorted order would deviate from the naive
        // first-occurrence contract for NaN entries) takes the general
        // first-occurrence path
        macro_rules! sorted_values {
            ($ty:ty) => {{
                if TypeId::of::<T>() == TypeId::of::<$ty>() {
                    // SAFETY: `TypeId` equality proves `T == $ty`; the re-typed raw
                    // Vec references address the same (dtype-independent) Vec
                    // layout, and all element access happens at `$ty`.
                    let a = unsafe { &*(a as *const Vec<T> as *const Vec<$ty>) };
                    let values = unsafe { &mut *(values as *mut Vec<MaybeUninit<T>> as *mut Vec<MaybeUninit<$ty>>) };
                    let access = LineAccess::new(a, la)?;
                    let u = unique_values_sorted_cpu_serial(values, &access)?;
                    values.truncate(u);
                    return Ok(u);
                }
            }};
        }
        for_each_fast_dtype!(sorted_values);
        let access = LineAccess::new(a, la)?;
        let u = unique_values_naive_cpu_serial(values, &access)?;
        values.truncate(u);
        Ok(u)
    }

    fn unique_all(
        &self,
        a: &Vec<T>,
        la: &Layout<D>,
        values: &mut Vec<MaybeUninit<T>>,
        indices: &mut Vec<MaybeUninit<usize>>,
        inverse: &mut Vec<MaybeUninit<usize>>,
        counts: &mut Vec<MaybeUninit<usize>>,
    ) -> Result<usize> {
        macro_rules! sorted_all {
            ($ty:ty) => {{
                if TypeId::of::<T>() == TypeId::of::<$ty>() {
                    // SAFETY: as in `unique_values` above.
                    let a = unsafe { &*(a as *const Vec<T> as *const Vec<$ty>) };
                    let values = unsafe { &mut *(values as *mut Vec<MaybeUninit<T>> as *mut Vec<MaybeUninit<$ty>>) };
                    let access = LineAccess::new(a, la)?;
                    let flat_c = |i: usize| -> usize { i };
                    let u = unique_all_sorted_cpu_serial(values, indices, inverse, counts, &access, &flat_c)?;
                    // contract: only the first `u` entries are initialized; truncate
                    // the raw Vecs so `assume_init_impl` at the tensor level is exact
                    values.truncate(u);
                    indices.truncate(u);
                    counts.truncate(u);
                    inverse.truncate(la.size());
                    return Ok(u);
                }
            }};
        }
        for_each_fast_dtype!(sorted_all);
        let access = LineAccess::new(a, la)?;
        let flat_c = |i: usize| -> usize { i };
        let u = unique_all_naive_cpu_serial(values, indices, inverse, counts, &access, &flat_c)?;
        // contract: values/indices/counts hold exactly `u` entries; inverse
        // holds `n`; truncate the raw Vecs so `assume_init_impl` is exact
        // (layout size — a broadcast view's storage may be shorter)
        values.truncate(u);
        indices.truncate(u);
        counts.truncate(u);
        inverse.truncate(la.size());
        Ok(u)
    }
}

/// Fill the `isin` output `c` (row-major visit order): the sorted
/// binary-search path for the TypeId-listed orderable dtypes (complex
/// included), the general linear-scan path otherwise.
fn isin_fill_dispatch<T, D1>(
    c: &mut [MaybeUninit<bool>],
    x1: &Vec<T>,
    l1: &Layout<D1>,
    x2: &Vec<T>,
    l2: &Layout<IxD>,
) -> Result<()>
where
    T: Clone + PartialEq + 'static,
    D1: DimAPI,
{
    macro_rules! sorted_isin {
        ($ty:ty) => {{
            if TypeId::of::<T>() == TypeId::of::<$ty>() {
                // SAFETY: `TypeId` equality proves `T == $ty`; the re-typed raw
                // Vec references address the same (dtype-independent) Vec layout.
                let x1 = unsafe { &*(x1 as *const Vec<T> as *const Vec<$ty>) };
                let x2 = unsafe { &*(x2 as *const Vec<T> as *const Vec<$ty>) };
                return isin_sorted_cpu_serial(c, x1, l1, x2, l2);
            }
        }};
    }
    for_each_fast_dtype!(sorted_isin);
    sorted_isin!(Complex<f32>);
    sorted_isin!(Complex<f64>);
    isin_naive_cpu_serial(c, x1, l1, x2, l2)
}

impl<T, D1> OpIsinAPI<T, D1> for DeviceCpuSerial
where
    T: Clone + PartialEq + 'static,
    D1: DimAPI,
{
    fn isin(
        &self,
        x1: &Vec<T>,
        l1: &Layout<D1>,
        x2: &Vec<T>,
        l2: &Layout<IxD>,
        invert: bool,
    ) -> Result<(Storage<DataOwned<Vec<bool>>, bool, Self>, Layout<IxD>)> {
        // the kernel writes in row-major visit order; the output must be
        // C-contig regardless of the device default order (values contract)
        let shape: Vec<usize> = l1.shape().as_ref().to_vec();
        let layout_c = shape.new_c_contig(None);
        let (_, idx_max) = layout_c.bounds_index()?;
        let mut storage = self.uninit_impl(idx_max)?;
        isin_fill_dispatch(storage.raw_mut(), x1, l1, x2, l2)?;
        if invert {
            for slot in storage.raw_mut().iter_mut() {
                // SAFETY: every slot was written by `isin_fill_dispatch`.
                let v = unsafe { slot.assume_init_mut() };
                *v = !*v;
            }
        }
        // SAFETY: `isin_fill_dispatch` (+ the invert flip above) wrote every
        // element of the fresh storage exactly once.
        let storage = unsafe { <Self as DeviceCreationAnyAPI<bool>>::assume_init_impl(storage)? };
        Ok((storage, layout_c))
    }
}
