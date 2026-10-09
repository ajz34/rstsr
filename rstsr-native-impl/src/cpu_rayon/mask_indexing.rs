//! Whole-tensor boolean-mask indexing kernels (rayon): parallel gather / scatter.
//!
//! The selected trailing blocks are disjoint, so both directions parallelize
//! over the mask visit order after the output offsets are known: `mask_fill`
//! needs no more than the write itself, `mask_select` a per-chunk exclusive
//! prefix of the chunk true-counts.

use crate::cpu_serial::mask_indexing::trailing_offsets;
use crate::prelude_dev::*;

// a mask copy/gather is memory bound; parallelize above ~128 KiB of f64 data
const PARALLEL_SWITCH: usize = 16384;

/// Rayon twin of [`crate::cpu_serial::mask_indexing::mask_select_cpu_serial`]:
/// identical output; the mask visit order is split into chunks and the selected
/// blocks are copied in parallel at their prefix-summed output offsets.
pub fn mask_select_cpu_rayon<T>(
    c: &mut [MaybeUninit<T>],
    a: &[T],
    la: &Layout<IxD>,
    mask: &[bool],
    lm: &Layout<IxD>,
    order: FlagOrder,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
{
    let pool = match pool {
        Some(pool) => pool,
        None => return mask_select_cpu_serial(c, a, la, mask, lm, order),
    };
    let dm = lm.ndim();
    let rel = trailing_offsets(la, dm, order);
    let block = rel.len();
    let lm_dim = lm.to_dim::<IxD>()?;
    let nm = lm_dim.size();
    // `nm * block` is `la`'s element count; below the switch the serial scan wins
    if nm == 0 || block == 0 || nm * block < PARALLEL_SWITCH {
        return mask_select_cpu_serial(c, a, la, mask, lm, order);
    }

    let la_base = la.offset() as isize;
    let la_stride: &[isize] = &la.stride()[..];
    let rel = &rel;

    // Split the visit order into chunks; a chunk's output base is the exclusive
    // prefix of the chunk true-counts, so the blocks land in visit order.
    let n_chunks = (pool.current_num_threads() * 4).min(nm);
    let chunk_len = nm.div_ceil(n_chunks);
    let mut chunks: Vec<IndexedIterLayout<IxD>> = Vec::with_capacity(n_chunks);
    let mut cur = IndexedIterLayout::<IxD>::new(&lm_dim, order)?;
    let mut done = 0_usize;
    while done < nm {
        let take = chunk_len.min(nm - done);
        let (lhs, rhs) = cur.split_at(take);
        chunks.push(lhs);
        cur = rhs;
        done += take;
    }

    let counts: Vec<usize> =
        chunks.par_iter().map(|chunk| chunk.clone().filter(|&(_, moff)| mask[moff]).count()).collect();
    let mut offsets = vec![0_usize; counts.len()];
    let mut acc = 0_usize;
    for (offset, &count) in offsets.iter_mut().zip(counts.iter()) {
        *offset = acc;
        acc += count;
    }

    let c_ptr = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        chunks.par_iter().zip(offsets.par_iter()).for_each(|(chunk, &rank0)| {
            let cp = c_ptr.load(Ordering::Relaxed);
            let mut rank = rank0;
            for (index, moff) in chunk.clone() {
                if !mask[moff] {
                    continue;
                }
                let mut base = la_base;
                for (d, &ix) in index.iter().enumerate() {
                    base += ix as isize * la_stride[d];
                }
                // SAFETY: block `rank` writes the disjoint output range
                // `[rank*block, (rank+1)*block)`; selected mask positions have
                // disjoint source blocks.
                let p = unsafe { cp.add(rank * block) };
                for (j, &r) in rel.iter().enumerate() {
                    unsafe {
                        p.add(j).write(MaybeUninit::new(a[(base + r) as usize].clone()));
                    }
                }
                rank += 1;
            }
        })
    };
    pool.install(task);
    Ok(())
}

/// Rayon twin of [`crate::cpu_serial::mask_indexing::mask_fill_cpu_serial`]:
/// identical output; the selected trailing blocks are written in parallel.
pub fn mask_fill_cpu_rayon<T>(
    a: &mut [T],
    la: &Layout<IxD>,
    mask: &[bool],
    lm: &Layout<IxD>,
    value: T,
    order: FlagOrder,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
{
    if pool.is_none() || a.len() < PARALLEL_SWITCH {
        return mask_fill_cpu_serial(a, la, mask, lm, value, order);
    }

    let dm = lm.ndim();
    let rel = trailing_offsets(la, dm, order);
    let la_base = la.offset() as isize;
    let la_stride: &[isize] = &la.stride()[..];
    let lm_dim = lm.to_dim::<IxD>()?;
    let iter = IndexedIterLayout::<IxD>::new(&lm_dim, order)?;

    let a_ptr = AtomicPtr::new(a.as_mut_ptr());
    let rel = &rel;
    let task = || {
        iter.into_par_iter().for_each(|(index, moff)| {
            if !mask[moff] {
                return;
            }
            let mut base = la_base;
            for (d, &ix) in index.iter().enumerate() {
                base += ix as isize * la_stride[d];
            }
            let p = a_ptr.load(Ordering::Relaxed);
            for &r in rel.iter() {
                // SAFETY: each selected mask position writes its own trailing
                // block; distinct positions have disjoint blocks.
                unsafe {
                    p.add((base + r) as usize).write(value.clone());
                }
            }
        })
    };
    pool.expect("pool checked Some above").install(task);
    Ok(())
}
