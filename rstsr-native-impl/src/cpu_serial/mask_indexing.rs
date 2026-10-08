//! Whole-tensor boolean-mask indexing kernels (serial).
//!
//! The mask has `dm <= da` axes matching `a`'s leading axes; a `true` selects
//! the trailing block `a[i_0, .., i_{dm-1}, ..]` of size
//! `B = prod(a.shape()[dm..])`. Mask entries are visited in `order` (the
//! device default order), and each selected block is emitted in that same
//! order — the array-API / NumPy masked-selection sequence.

use crate::prelude_dev::*;

/// Relative offsets of the trailing block, in `order` visit sequence.
///
/// The block shape/stride are those of `la.shape()[dm..]` / `la.stride()[dm..]`;
/// offsets are relative to the block base. Empty block (a zero-size axis)
/// yields no offsets.
fn trailing_offsets(la: &Layout<IxD>, dm: usize, order: FlagOrder) -> Vec<isize> {
    let shape: &[usize] = &la.shape()[dm..];
    let stride: &[isize] = &la.stride()[dm..];
    let n = shape.len();
    if shape.contains(&0) {
        return Vec::new();
    }
    if n == 0 {
        return vec![0];
    }
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0_usize; n];
    loop {
        out.push((0..n).map(|j| idx[j] as isize * stride[j]).sum());
        let mut carry = true;
        match order {
            // row-major visits the last axis fastest, column-major the first
            RowMajor => {
                for j in (0..n).rev() {
                    idx[j] += 1;
                    if idx[j] < shape[j] {
                        carry = false;
                        break;
                    }
                    idx[j] = 0;
                }
            },
            ColMajor => {
                for j in 0..n {
                    idx[j] += 1;
                    if idx[j] < shape[j] {
                        carry = false;
                        break;
                    }
                    idx[j] = 0;
                }
            },
        }
        if carry {
            break;
        }
    }
    out
}

/// Gather: for every `true` in `mask`, copy the trailing block of `a` into `c`.
///
/// `c` has capacity `count * B` (`count` = number of `true` entries) and is
/// written front-to-back in the mask visit order.
pub fn mask_select_cpu_serial<T>(
    c: &mut [MaybeUninit<T>],
    a: &[T],
    la: &Layout<IxD>,
    mask: &[bool],
    lm: &Layout<IxD>,
    order: FlagOrder,
) -> Result<()>
where
    T: Clone,
{
    let dm = lm.ndim();
    let rel = trailing_offsets(la, dm, order);
    let la_base = la.offset() as isize;
    let lm_dim = lm.to_dim::<IxD>()?;
    let mut w = 0_usize;
    for (index, moff) in IndexedIterLayout::<IxD>::new(&lm_dim, order)? {
        if !mask[moff] {
            continue;
        }
        let mut base = la_base;
        for (d, &ix) in index.iter().enumerate() {
            base += ix as isize * la.stride()[d];
        }
        for &r in rel.iter() {
            c[w].write(a[(base + r) as usize].clone());
            w += 1;
        }
    }
    Ok(())
}

/// Scatter a scalar: write `value` into every element selected by `mask`.
pub fn mask_fill_cpu_serial<T>(
    a: &mut [T],
    la: &Layout<IxD>,
    mask: &[bool],
    lm: &Layout<IxD>,
    value: T,
    order: FlagOrder,
) -> Result<()>
where
    T: Clone,
{
    let dm = lm.ndim();
    let rel = trailing_offsets(la, dm, order);
    let la_base = la.offset() as isize;
    let lm_dim = lm.to_dim::<IxD>()?;
    for (index, moff) in IndexedIterLayout::<IxD>::new(&lm_dim, order)? {
        if !mask[moff] {
            continue;
        }
        let mut base = la_base;
        for (d, &ix) in index.iter().enumerate() {
            base += ix as isize * la.stride()[d];
        }
        for &r in rel.iter() {
            a[(base + r) as usize] = value.clone();
        }
    }
    Ok(())
}
