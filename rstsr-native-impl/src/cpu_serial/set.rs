//! Set-operation kernels (serial): naive unique (PartialEq, general bound)
//! and unique_all sweeps; binary-search isin.

use crate::prelude_dev::*;
use core::cmp::Ordering;

use rstsr_dtype_traits::ExtSortCmp;

/// Element accessor over a layout: offset + stride walk in row-major order
/// through an arbitrary input layout.
pub struct LineAccess<'a, T> {
    data: &'a [T],
    offsets: Vec<usize>,
}

impl<'a, T> LineAccess<'a, T> {
    /// Collect the row-major offsets of every element of `la` into `data`.
    pub fn new<D: DimAPI>(data: &'a [T], la: &Layout<D>) -> Result<Self> {
        let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&la.to_dim()?, RowMajor)?;
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

/// Naive unique: first-occurrence order in row-major sequence.
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

/// Naive unique_all sweep over the row-major sequence.
///
/// Writes: unique values into `values` (first-occurrence order), their
/// first-occurrence flat C-order index into `indices` (length `u`), the
/// unique-entry index for every input element into `inverse` (length `n`),
/// and the multiplicity of each unique value into `counts` (length `u`).
/// Returns `u`.
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

/// Isin: for each element of `x1`, whether it appears in the sorted unique
/// sequence `x2_sorted` (binary search). Writes `bool`s into `c`.
///
/// Membership is value equality (`==`, array-api `isin`'s contract via
/// `equal`): a NaN key is never a member — NumPy parity — so NaN keys short-
/// circuit to `false` without searching.
pub fn isin_cpu_serial<T>(
    c: &mut [MaybeUninit<bool>],
    x2_sorted: &[T],
    x1_access: &LineAccess<'_, T>,
    is_nan: &dyn Fn(&T) -> bool,
) -> Result<()>
where
    T: ExtSortCmp,
{
    let n = x1_access.len();
    let m = x2_sorted.len();
    for (i, c_slot) in c.iter_mut().take(n).enumerate() {
        let v = x1_access.get(i);
        let found = if is_nan(v) {
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

/// Fast unique: copy all values (row-major) into `values` (capacity n), sort
/// with the total order of [`ExtSortCmp`], dedupe adjacent in place. Returns
/// the unique count `u`; values are in ascending order (NumPy parity).
pub fn unique_values_sorted_cpu_serial<T>(
    values: &mut [MaybeUninit<T>],
    access: &LineAccess<'_, T>,
    is_nan: &dyn Fn(&T) -> bool,
) -> Result<usize>
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
    // adjacent dedupe (NaN entries are mutually Equal and collapse; use
    // is_nan to keep each NaN distinct, matching the naive path's contract)
    let mut u = if n > 0 { 1 } else { 0 };
    for i in 1..n {
        let prev_is_nan = is_nan(&slice[u - 1]);
        let cur_is_nan = is_nan(&slice[i]);
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
/// = first-occurrence flat C-order index of each unique value, `inverse` =
/// unique-entry slot per input element, `counts` = multiplicities. Returns
/// the unique count `u`.
#[allow(clippy::too_many_arguments)]
pub fn unique_all_sorted_cpu_serial<T>(
    values: &mut [MaybeUninit<T>],
    indices: &mut [MaybeUninit<usize>],
    inverse: &mut [MaybeUninit<usize>],
    counts: &mut [MaybeUninit<usize>],
    access: &LineAccess<'_, T>,
    flat_c: &dyn Fn(usize) -> usize,
    is_nan: &dyn Fn(&T) -> bool,
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
        let cur_is_nan = is_nan(v);
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
