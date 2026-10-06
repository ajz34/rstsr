//! nonzero kernels (serial): two-pass count + fill over the row-major
//! visit sequence; per-dimension coordinate split is host layout math.

use crate::prelude_dev::*;

/// Count the nonzero elements of `a` (row-major visit order; bool: true;
/// complex: either component nonzero — `is_nonzero` decides).
pub fn nonzero_count_cpu_serial<T, D>(a: &[T], la: &Layout<D>, is_nonzero: &dyn Fn(&T) -> bool) -> Result<usize>
where
    T: Clone,
    D: DimAPI,
{
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&la.to_dim()?, RowMajor)?;
    let mut count = 0_usize;
    for (_, off) in iter {
        if is_nonzero(&a[off]) {
            count += 1;
        }
    }
    Ok(count)
}

/// Fill the flat C-order indices of every nonzero element into `out`
/// (capacity = the count from [`nonzero_count_cpu_serial`]).
pub fn nonzero_fill_cpu_serial<T, D>(
    out: &mut [MaybeUninit<usize>],
    a: &[T],
    la: &Layout<D>,
    is_nonzero: &dyn Fn(&T) -> bool,
) -> Result<usize>
where
    T: Clone,
    D: DimAPI,
{
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&la.to_dim()?, RowMajor)?;
    let mut count = 0_usize;
    for (flat, (_, off)) in iter.enumerate() {
        if is_nonzero(&a[off]) {
            // the row-major visit position IS the flat C-order index
            out[count].write(flat);
            count += 1;
        }
    }
    Ok(count)
}
