//! Array indexing (fancy indexing) kernel (serial): gather by integer arrays.
//!
//! The index arrays broadcast together (trailing-aligned) into `fancy_ndim`
//! bulk dimensions, which sit in the output at `consec`; the remaining output
//! dimensions form the subspace described by `base_layout` (whose strides are
//! those of `la`, with integer selections already folded into its offset).
//! Every layout involved may be strided.

use crate::prelude_dev::*;

/// Gather `a` into `c` by integer arrays on selected axes.
///
/// `indexers` carries, per array indexer, its source axis, its resolved
/// (non-negative, in-bounds) values in C order, and the layout of its own
/// shape.
#[allow(clippy::too_many_arguments)]
pub fn array_index_cpu_serial<T>(
    c: &mut [MaybeUninit<T>],
    lc: &Layout<IxD>,
    a: &[T],
    la: &Layout<IxD>,
    base_layout: &Layout<IxD>,
    indexers: &[(usize, &[usize], Layout<IxD>)],
    consec: usize,
) -> Result<()>
where
    T: Clone,
{
    let ndim_c = lc.ndim();
    let ndim_base = base_layout.ndim();
    rstsr_assert!(ndim_c >= ndim_base, InvalidLayout, "Output layout rank is smaller than the subspace rank.")?;
    let fancy_ndim = ndim_c - ndim_base;

    let bulk_shape: &[usize] = &lc.shape()[consec..consec + fancy_ndim];
    let base_shape: &[usize] = &base_layout.shape()[..];
    let n_bulk: usize = bulk_shape.iter().product();
    let n_base: usize = base_shape.iter().product();
    // a zero-sized dimension means nothing to write (and guards the unravel
    // below against a zero divisor)
    if n_bulk == 0 || n_base == 0 {
        return Ok(());
    }


    let la_stride: &[isize] = la.stride().as_ref();

    let mut out_multi = vec![0_usize; ndim_c];
    let mut base_multi = vec![0_usize; ndim_base];
    let mut bulk_multi = vec![0_usize; fancy_ndim];

    for bulk_flat in 0..n_bulk {
        let mut rem = bulk_flat;
        for d in (0..fancy_ndim).rev() {
            bulk_multi[d] = rem % bulk_shape[d];
            rem /= bulk_shape[d];
        }
        out_multi[consec..consec + fancy_ndim].copy_from_slice(&bulk_multi[..fancy_ndim]);
        for base_flat in 0..n_base {
            let mut rem = base_flat;
            for d in (0..ndim_base).rev() {
                base_multi[d] = rem % base_shape[d];
                rem /= base_shape[d];
            }
            out_multi[..consec].copy_from_slice(&base_multi[..consec]);
            out_multi[consec + fancy_ndim..ndim_base + fancy_ndim]
                .copy_from_slice(&base_multi[consec..ndim_base]);
            let out_off = lc.index_uncheck(&out_multi) as usize;
            let mut src_off: isize = base_layout.index_uncheck(&base_multi);
            for (src_axis, indices, layout) in indexers {
                let ndim_idx = layout.ndim();
                let idx_shape: &[usize] = &layout.shape()[..];
                let idx_stride: &[isize] = &layout.stride()[..];
                let mut idx_off = layout.offset() as isize;
                for d in 0..ndim_idx {
                    // dim `d` of an index array aligns with the trailing bulk
                    // dimensions; a size-1 dim reuses its single slice
                    let m = if idx_shape[d] == 1 { 0 } else { bulk_multi[fancy_ndim - ndim_idx + d] };
                    idx_off += idx_stride[d] * m as isize;
                }
                src_off += la_stride[*src_axis] * indices[idx_off as usize] as isize;
            }
            // SAFETY: the tensor level validated every index against its
            // axis; the subspace offset is an input-stride dot-product over
            // in-range positions (see the layout bounds checks), and the
            // output position is the layout's own index.
            let src = a[src_off as usize].clone();
            c[out_off].write(src);
        }
    }
    Ok(())
}
