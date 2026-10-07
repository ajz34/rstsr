//! nonzero kernels (serial): two-pass count + per-dimension coordinate fill
//! over the row-major visit sequence.

use rstsr_dtype_traits::ExtZero;

use crate::prelude_dev::*;

/// Count the nonzero elements of `a` (`!= 0`; bool: `true`; complex: either
/// component nonzero; NaN is nonzero) in row-major visit order.
pub fn nonzero_count_cpu_serial<T, D>(a: &[T], la: &Layout<D>) -> Result<usize>
where
    T: ExtZero + PartialEq,
    D: DimAPI,
{
    let zero = T::ext_zero();
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&la.to_dim()?, RowMajor)?;
    let mut count = 0_usize;
    for (_, off) in iter {
        if a[off] != zero {
            count += 1;
        }
    }
    Ok(count)
}

/// Fill the coordinates of every nonzero element into `out` (one buffer per
/// dimension, capacity = the count from [`nonzero_count_cpu_serial`]) in
/// row-major visit order.
pub fn nonzero_fill_cpu_serial<T, D>(out: &mut [&mut Vec<MaybeUninit<usize>>], a: &[T], la: &Layout<D>) -> Result<()>
where
    T: ExtZero + PartialEq,
    D: DimAPI,
{
    let zero = T::ext_zero();
    let iter: IndexedIterLayout<IxD> = IndexedIterLayout::new(&la.to_dim()?, RowMajor)?;
    let mut k = 0_usize;
    for (index, off) in iter {
        if a[off] != zero {
            for (d, buffer) in out.iter_mut().enumerate() {
                buffer[k].write(index[d]);
            }
            k += 1;
        }
    }
    Ok(())
}
