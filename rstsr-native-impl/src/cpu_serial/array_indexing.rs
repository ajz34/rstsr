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
/// (non-negative, in-bounds) entries, and the layout of its own shape (the
/// entries are addressed through that layout, so any strides/offset work). The
/// broadcast index dimensions are visited in `order`, the device default order;
/// the two per-bulk tables are paired per visit, so the visit order does not
/// affect which value lands where.
#[allow(clippy::too_many_arguments)]
pub fn array_index_cpu_serial<T>(
    c: &mut [MaybeUninit<T>],
    lc: &Layout<IxD>,
    a: &[T],
    la: &Layout<IxD>,
    base_layout: &Layout<IxD>,
    indexers: &[(usize, &[usize], Layout<IxD>)],
    consec: usize,
    order: FlagOrder,
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

    let lc_stride: &[isize] = &lc.stride()[..];
    let la_stride: &[isize] = &la.stride()[..];

    // Per-bulk tables (independent of the subspace): the source offset
    // contributed by the index arrays, and the output offset contributed by the
    // bulk dimensions.
    let mut src_bulk = vec![0_isize; n_bulk];
    let mut out_bulk = vec![0_isize; n_bulk];
    let mut bulk_multi = vec![0_usize; fancy_ndim];
    for (bulk_flat, (src, out)) in src_bulk.iter_mut().zip(out_bulk.iter_mut()).enumerate() {
        // unravel the flat bulk position in the device default order:
        // row-major varies the last axis fastest, column-major the first
        let mut rem = bulk_flat;
        match order {
            RowMajor => {
                for d in (0..fancy_ndim).rev() {
                    bulk_multi[d] = rem % bulk_shape[d];
                    rem /= bulk_shape[d];
                }
            },
            ColMajor => {
                for d in 0..fancy_ndim {
                    bulk_multi[d] = rem % bulk_shape[d];
                    rem /= bulk_shape[d];
                }
            },
        }
        for (d, &m) in bulk_multi.iter().enumerate() {
            *out += lc_stride[consec + d] * m as isize;
        }
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
            *src += la_stride[*src_axis] * indices[idx_off as usize] as isize;
        }
    }

    // Outer loop over the subspace, inner over the bulk dimensions; the
    // subspace-side offsets are computed once per subspace position.
    let mut base_multi = vec![0_usize; ndim_base];
    for base_flat in 0..n_base {
        let mut rem = base_flat;
        for d in (0..ndim_base).rev() {
            base_multi[d] = rem % base_shape[d];
            rem /= base_shape[d];
        }
        let src_base = base_layout.index_uncheck(&base_multi);
        // `lc.offset()` is part of the output layout: it must reach the write
        // offset as well (the sibling ops never assume a zero-offset layout)
        let mut out_base = lc.offset() as isize;
        for (d, &m) in base_multi.iter().enumerate() {
            let stride = if d < consec { lc_stride[d] } else { lc_stride[d + fancy_ndim] };
            out_base += stride * m as isize;
        }
        for bulk_flat in 0..n_bulk {
            let out_off = (out_base + out_bulk[bulk_flat]) as usize;
            let src_off = (src_base + src_bulk[bulk_flat]) as usize;
            // SAFETY: the tensor level validated every index against its axis;
            // both offsets are layout dot-products over in-range positions
            // (see the layouts' own bounds checks), so they address initialized
            // elements of `a` and slots of the fresh output buffer.
            c[out_off].write(a[src_off].clone());
        }
    }
    Ok(())
}

/// Scatter (inverse of [`array_index_cpu_serial`]): write `value` into `a` at
/// the positions the index arrays select, casting `TA` to `TC` on the fly.
///
/// `la` / `base_layout` / `indexers` address the destination exactly as the
/// gather addresses its source; `lvalue` is the value layout broadcast to the
/// gather output shape (its broadcast block sits at `consec`). Duplicate index
/// targets write the same slot more than once; the visit order (`order`) decides
/// the winner — the last write wins.
#[allow(clippy::too_many_arguments)]
pub fn array_index_assign_promote_cpu_serial<TC, TA>(
    a: &mut [TC],
    la: &Layout<IxD>,
    base_layout: &Layout<IxD>,
    indexers: &[(usize, &[usize], Layout<IxD>)],
    value: &[TA],
    lvalue: &Layout<IxD>,
    consec: usize,
    order: FlagOrder,
) -> Result<()>
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
{
    let ndim_l = lvalue.ndim();
    let ndim_base = base_layout.ndim();
    rstsr_assert!(ndim_l >= ndim_base, InvalidLayout, "Value layout rank is smaller than the subspace rank.")?;
    let fancy_ndim = ndim_l - ndim_base;

    let bulk_shape: &[usize] = &lvalue.shape()[consec..consec + fancy_ndim];
    let base_shape: &[usize] = &base_layout.shape()[..];
    let n_bulk: usize = bulk_shape.iter().product();
    let n_base: usize = base_shape.iter().product();
    // a zero-sized dimension means nothing to write (and guards the unravel
    // below against a zero divisor)
    if n_bulk == 0 || n_base == 0 {
        return Ok(());
    }

    let lv_stride: &[isize] = &lvalue.stride()[..];
    let la_stride: &[isize] = &la.stride()[..];

    // Per-bulk tables: the destination offset contributed by the index arrays,
    // and the value offset contributed by the broadcast dimensions.
    let mut dst_bulk = vec![0_isize; n_bulk];
    let mut val_bulk = vec![0_isize; n_bulk];
    let mut bulk_multi = vec![0_usize; fancy_ndim];
    for (bulk_flat, (dst, val)) in dst_bulk.iter_mut().zip(val_bulk.iter_mut()).enumerate() {
        // unravel the flat bulk position in the device default order:
        // row-major varies the last axis fastest, column-major the first
        let mut rem = bulk_flat;
        match order {
            RowMajor => {
                for d in (0..fancy_ndim).rev() {
                    bulk_multi[d] = rem % bulk_shape[d];
                    rem /= bulk_shape[d];
                }
            },
            ColMajor => {
                for d in 0..fancy_ndim {
                    bulk_multi[d] = rem % bulk_shape[d];
                    rem /= bulk_shape[d];
                }
            },
        }
        for (d, &m) in bulk_multi.iter().enumerate() {
            *val += lv_stride[consec + d] * m as isize;
        }
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
            *dst += la_stride[*src_axis] * indices[idx_off as usize] as isize;
        }
    }

    // Outer loop over the subspace, inner over the broadcast dimensions; the
    // subspace-side offsets are computed once per subspace position.
    let mut base_multi = vec![0_usize; ndim_base];
    for base_flat in 0..n_base {
        let mut rem = base_flat;
        for d in (0..ndim_base).rev() {
            base_multi[d] = rem % base_shape[d];
            rem /= base_shape[d];
        }
        let dst_base = base_layout.index_uncheck(&base_multi);
        let mut val_base = lvalue.offset() as isize;
        for (d, &m) in base_multi.iter().enumerate() {
            let stride = if d < consec { lv_stride[d] } else { lv_stride[d + fancy_ndim] };
            val_base += stride * m as isize;
        }
        for bulk_flat in 0..n_bulk {
            let dst_off = (dst_base + dst_bulk[bulk_flat]) as usize;
            let val_off = (val_base + val_bulk[bulk_flat]) as usize;
            a[dst_off] = value[val_off].clone().into_cast();
        }
    }
    Ok(())
}
