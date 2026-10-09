//! Array indexing (fancy indexing) kernel (rayon): parallel gather.
//!
//! Both the offset tables and the gather are built in parallel; the trait
//! contract (write every output element exactly once) makes the writes disjoint,
//! so no locking or reduction is needed.

use crate::prelude_dev::*;

// a gather is memory bound; parallelize above ~128 KiB of f64 output
const PARALLEL_SWITCH: usize = 16384;

/// Rayon twin of [`crate::cpu_serial::array_indexing::array_index_cpu_serial`]:
/// identical output; the offset tables and the flattened (subspace × broadcast)
/// gather are built in parallel when the output is large enough.
///
/// Four tables (source and output offsets for the broadcast dims and for the
/// subspace) keep the per-element cost equal to the serial inner loop without
/// an `O(n_out)` scratch buffer.
#[allow(clippy::too_many_arguments)]
pub fn array_index_cpu_rayon<T>(
    c: &mut [MaybeUninit<T>],
    lc: &Layout<IxD>,
    a: &[T],
    la: &Layout<IxD>,
    base_layout: &Layout<IxD>,
    indexers: &[(usize, &[usize], Layout<IxD>)],
    consec: usize,
    order: FlagOrder,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
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
    let pool = match pool {
        Some(pool) if n_base * n_bulk >= PARALLEL_SWITCH => pool,
        _ => return array_index_cpu_serial(c, lc, a, la, base_layout, indexers, consec, order),
    };

    let lc_stride: &[isize] = &lc.stride()[..];
    let la_stride: &[isize] = &la.stride()[..];

    let mut src_bulk = vec![0_isize; n_bulk];
    let mut out_bulk = vec![0_isize; n_bulk];
    let mut src_base = vec![0_isize; n_base];
    let mut out_base = vec![0_isize; n_base];

    pool.install(|| {
        // Per-broadcast tables: the source offset contributed by the index
        // arrays and the output offset contributed by the bulk dimensions.
        src_bulk.par_iter_mut().zip(out_bulk.par_iter_mut()).enumerate().for_each_init(
            || vec![0_usize; fancy_ndim],
            |bulk_multi, (bulk_flat, (src, out))| {
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
                        // dim `d` of an index array aligns with the trailing
                        // bulk dimensions; a size-1 dim reuses its single slice
                        let m = if idx_shape[d] == 1 { 0 } else { bulk_multi[fancy_ndim - ndim_idx + d] };
                        idx_off += idx_stride[d] * m as isize;
                    }
                    *src += la_stride[*src_axis] * indices[idx_off as usize] as isize;
                }
            },
        );

        // Per-subspace tables: the source offset and the output offset (with the
        // broadcast dimensions removed).
        src_base.par_iter_mut().zip(out_base.par_iter_mut()).enumerate().for_each_init(
            || vec![0_usize; ndim_base],
            |base_multi, (base_flat, (src, out))| {
                let mut rem = base_flat;
                for d in (0..ndim_base).rev() {
                    base_multi[d] = rem % base_shape[d];
                    rem /= base_shape[d];
                }
                *src = base_layout.index_uncheck(base_multi);
                // `lc.offset()` is part of the output layout: it must reach the
                // write offset as well (the sibling ops never assume a zero
                // offset layout)
                let mut off = lc.offset() as isize;
                for (d, &m) in base_multi.iter().enumerate() {
                    let stride = if d < consec { lc_stride[d] } else { lc_stride[d + fancy_ndim] };
                    off += stride * m as isize;
                }
                *out = off;
            },
        );

        // Flatten the (subspace × broadcast) loop over disjoint output offsets,
        // nesting the smaller dimension inside (so no per-element division).
        let c_ptr = AtomicPtr::new(c.as_mut_ptr());
        if n_bulk >= n_base {
            (0..n_bulk).into_par_iter().for_each(|q| {
                let cp = c_ptr.load(Ordering::Relaxed);
                for b in 0..n_base {
                    let out_off = (out_base[b] + out_bulk[q]) as usize;
                    let src_off = (src_base[b] + src_bulk[q]) as usize;
                    // SAFETY: the tensor level validated every index; both
                    // offsets are layout dot-products over in-range positions.
                    // Distinct `(base, bulk)` pairs map to disjoint output
                    // offsets (write-every-element-exactly-once contract).
                    unsafe {
                        cp.add(out_off).write(MaybeUninit::new(a[src_off].clone()));
                    }
                }
            });
        } else {
            (0..n_base).into_par_iter().for_each(|b| {
                let cp = c_ptr.load(Ordering::Relaxed);
                for q in 0..n_bulk {
                    let out_off = (out_base[b] + out_bulk[q]) as usize;
                    let src_off = (src_base[b] + src_bulk[q]) as usize;
                    // SAFETY: as above.
                    unsafe {
                        cp.add(out_off).write(MaybeUninit::new(a[src_off].clone()));
                    }
                }
            });
        }
    });
    Ok(())
}
