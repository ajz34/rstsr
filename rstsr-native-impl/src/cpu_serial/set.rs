//! Set-operation kernels (serial): naive unique (PartialEq, general bound)
//! and unique_all sweeps; sorted / linear-scan isin.

use crate::prelude_dev::*;
use core::cmp::Ordering;

use rstsr_dtype_traits::ExtSortCmp;

/// Element accessor over a layout: offset + stride walk in the given visit
/// order through an arbitrary input layout.
pub struct LineAccess<'a, T> {
    data: &'a [T],
    offsets: Vec<usize>,
}

impl<'a, T> LineAccess<'a, T> {
    /// Collect the offsets of every element of `la` into `data` in `order`.
    pub fn new<D: DimAPI>(data: &'a [T], la: &Layout<D>, order: FlagOrder) -> Result<Self> {
        let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&la.to_dim()?, order)?;
        let offsets = iter.map(|(_, off)| off).collect();
        Ok(Self { data, offsets })
    }

    #[inline]
    pub fn get(&self, i: usize) -> &T {
        // SAFETY-CONTRACT: offsets are validated layout addresses.
        &self.data[self.offsets[i]]
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Naive unique: first-occurrence order in the accessor's visit sequence.
///
/// Writes up to `n` entries into `values`; returns the unique count `u`.
/// Values are compared with `PartialEq` (covers complex and any dtype);
/// NaNs are distinct entries (PartialEq never merges NaNs); signed zeros
/// merge (`==` holds) with the first-seen encoding kept.
pub fn unique_values_naive_cpu_serial<T>(values: &mut [MaybeUninit<T>], access: &LineAccess<'_, T>) -> Result<usize>
where
    T: Clone + PartialEq,
{
    let n = access.len();
    let mut seen: Vec<&T> = Vec::with_capacity(n);
    let mut count = 0_usize;
    for i in 0..n {
        let v = access.get(i);
        if seen.contains(&v) {
            continue;
        }
        seen.push(v);
        values[count].write(v.clone());
        count += 1;
    }
    Ok(count)
}

/// Naive unique_all sweep over the accessor's visit sequence.
///
/// Writes: unique values into `values` (first-occurrence order), their
/// first-occurrence flat index into `indices` (length `u`; visit position
/// under the caller's order), the unique-entry index for every input element
/// into `inverse` (length `n`), and the multiplicity of each unique value
/// into `counts` (length `u`). Returns `u`.
pub fn unique_all_naive_cpu_serial<T>(
    values: &mut [MaybeUninit<T>],
    indices: &mut [MaybeUninit<usize>],
    inverse: &mut [MaybeUninit<usize>],
    counts: &mut [MaybeUninit<usize>],
    access: &LineAccess<'_, T>,
    flat_c: &dyn Fn(usize) -> usize,
) -> Result<usize>
where
    T: Clone + PartialEq,
{
    let n = access.len();
    let mut seen: Vec<&T> = Vec::with_capacity(n);
    let mut count = 0_usize;
    for (i, slot_out) in inverse.iter_mut().take(n).enumerate() {
        let v = access.get(i);
        let mut fresh = false;
        let slot = match seen.iter().position(|s| *s == v) {
            Some(slot) => slot,
            None => {
                seen.push(v);
                values[count].write(v.clone());
                indices[count].write(flat_c(i));
                counts[count].write(1);
                count += 1;
                fresh = true;
                count - 1
            },
        };
        slot_out.write(slot);
        if !fresh {
            // increment the running multiplicity of `slot`
            let c = unsafe { counts[slot].assume_init_mut() };
            *c += 1;
        }
    }
    Ok(count)
}

/// Isin, sorted path: sort and dedupe a copy of `x2`'s values (read in
/// `order`) by the total order of [`ExtSortCmp`], then binary-search each
/// `x1` element. Writes `bool`s into `c` in the `order` visit sequence.
///
/// Membership is value equality (`==`, array-api `isin`'s contract via
/// `equal`): a NaN key is never a member — NumPy parity — so NaN keys short-
/// circuit to `false` without searching.
pub fn isin_sorted_cpu_serial<T, D1>(
    c: &mut [MaybeUninit<bool>],
    x1: &[T],
    l1: &Layout<D1>,
    x2: &[T],
    l2: &Layout<IxD>,
    order: FlagOrder,
) -> Result<()>
where
    T: Clone + PartialEq + ExtSortCmp,
    D1: DimAPI,
{
    // efficiency exception (registered): sort a deduped copy of x2's values
    let access2 = LineAccess::new(x2, l2, order)?;
    let n2 = access2.len();
    let mut x2_sorted: Vec<T> = Vec::with_capacity(n2);
    for i in 0..n2 {
        x2_sorted.push(access2.get(i).clone());
    }
    x2_sorted.sort_by(|a, b| a.ext_total_cmp(b));
    x2_sorted.dedup_by(|a, b| a == b);

    let access1 = LineAccess::new(x1, l1, order)?;
    let m = x2_sorted.len();
    for (i, c_slot) in c.iter_mut().take(access1.len()).enumerate() {
        let v = access1.get(i);
        let found = if v.ext_is_nan() {
            // NaN is never equal under `==` (NumPy: isin([nan], [nan]) is
            // false); complex NaN-bearing keys likewise match nothing
            false
        } else {
            let mut lo = 0_usize;
            let mut hi = m;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                match x2_sorted[mid].ext_total_cmp(v) {
                    Ordering::Less => lo = mid + 1,
                    _ => hi = mid,
                }
            }
            lo < m && x2_sorted[lo].ext_total_cmp(v) == Ordering::Equal
        };
        c_slot.write(found);
    }
    Ok(())
}

/// Isin, general path: linear membership scan over `x2` (value equality `==`;
/// no ordering required). Writes `bool`s into `c` in the `order` visit
/// sequence.
pub fn isin_naive_cpu_serial<T, D1>(
    c: &mut [MaybeUninit<bool>],
    x1: &[T],
    l1: &Layout<D1>,
    x2: &[T],
    l2: &Layout<IxD>,
    order: FlagOrder,
) -> Result<()>
where
    T: PartialEq,
    D1: DimAPI,
{
    let access1 = LineAccess::new(x1, l1, order)?;
    let access2 = LineAccess::new(x2, l2, order)?;
    let m = access2.len();
    for (i, c_slot) in c.iter_mut().take(access1.len()).enumerate() {
        let v = access1.get(i);
        // NaN (and NaN-bearing complex) never compare equal — NumPy parity
        c_slot.write((0..m).any(|j| access2.get(j) == v));
    }
    Ok(())
}

/// Fast unique: copy all values (in the accessor's visit order) into
/// `values` (capacity n), sort with the total order of [`ExtSortCmp`], dedupe
/// adjacent in place. Returns the unique count `u`; values are in ascending
/// order (NumPy parity).
pub fn unique_values_sorted_cpu_serial<T>(values: &mut [MaybeUninit<T>], access: &LineAccess<'_, T>) -> Result<usize>
where
    T: Clone + ExtSortCmp + PartialEq,
{
    let n = access.len();
    for (i, v_slot) in values.iter_mut().take(n).enumerate() {
        v_slot.write(access.get(i).clone());
    }
    // SAFETY-CONTRACT: values[0..n] all written above.
    let slice = unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut T, n) };
    // stable sort with NaN-last total order; equal (±0) keep input order
    slice.sort_by(|a, b| a.ext_total_cmp(b));
    // adjacent dedupe (NaN entries are mutually Equal and collapse; each NaN
    // stays distinct via ext_is_nan, matching the naive path's contract)
    let mut u = if n > 0 { 1 } else { 0 };
    for i in 1..n {
        let prev_is_nan = slice[u - 1].ext_is_nan();
        let cur_is_nan = slice[i].ext_is_nan();
        if !prev_is_nan && !cur_is_nan && slice[u - 1] == slice[i] {
            continue;
        }
        if u != i {
            slice[u] = slice[i].clone();
        }
        u += 1;
    }
    Ok(u)
}

/// Fast unique_all: values sorted ascending (NaN tail distinct), `indices`
/// = first-occurrence flat index of each unique value (visit position under
/// the caller's order), `inverse` = unique-entry slot per input element,
/// `counts` = multiplicities. Returns the unique count `u`.
#[allow(clippy::too_many_arguments)]
pub fn unique_all_sorted_cpu_serial<T>(
    values: &mut [MaybeUninit<T>],
    indices: &mut [MaybeUninit<usize>],
    inverse: &mut [MaybeUninit<usize>],
    counts: &mut [MaybeUninit<usize>],
    access: &LineAccess<'_, T>,
    flat_c: &dyn Fn(usize) -> usize,
) -> Result<usize>
where
    T: Clone + ExtSortCmp + PartialEq,
{
    let n = access.len();
    // sort (value, row-major position) pairs by the total order; stability
    // keeps first occurrence first among equals
    let mut pairs: Vec<(T, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        pairs.push((access.get(i).clone(), i));
    }
    pairs.sort_by(|a, b| a.0.ext_total_cmp(&b.0));

    let mut u = 0_usize;
    let mut prev_is_nan = false;
    for (j, (v, pos)) in pairs.iter().enumerate() {
        let cur_is_nan = v.ext_is_nan();
        // new entry iff first, NaN-class changes, finite values differ, or
        // both NaN (each NaN is a distinct entry)
        let new_entry = j == 0
            || (prev_is_nan != cur_is_nan)
            || (!prev_is_nan && !cur_is_nan && !(pairs[j - 1].0 == *v))
            || (prev_is_nan && cur_is_nan);
        if new_entry {
            values[u].write(v.clone());
            indices[u].write(flat_c(*pos));
            counts[u].write(1);
            prev_is_nan = cur_is_nan;
            u += 1;
        } else {
            let c = unsafe { counts[u - 1].assume_init_mut() };
            *c += 1;
        }
        inverse[*pos].write(u - 1);
    }
    Ok(u)
}
