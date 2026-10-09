//! take_along_axis kernel (rayon): parallel gather over the rest positions.

use crate::prelude_dev::*;

// a gather is memory bound; parallelize above ~128 KiB of f64 output
const PARALLEL_SWITCH: usize = 16384;

/// Rayon twin of
/// [`crate::cpu_serial::adv_indexing_take_along::take_along_axis_cpu_serial`]:
/// identical output; the (rest × axis) nested loop is run in parallel when the
/// output is large enough, the smaller dimension nested inside.
///
/// Three precomputed tables (input, index and output offsets per rest position)
/// replace the serial per-position unravel, so the parallel loop carries no
/// allocation.
#[allow(clippy::too_many_arguments)]
pub fn take_along_axis_cpu_rayon<T, DA, DI>(
    c: &mut [MaybeUninit<T>],
    out_strides: &[usize],
    a: &[T],
    la: &Layout<DA>,
    idx: &[usize],
    lidx: &Layout<DI>,
    axis: usize,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
    DA: DimAPI,
    DI: DimAPI,
{
    if pool.is_none() || c.len() < PARALLEL_SWITCH {
        return take_along_axis_cpu_serial(c, out_strides, a, la, idx, lidx, axis);
    }

    let ndim = la.ndim();
    rstsr_check_axis!(axis as isize, ndim)?;
    let axis_stride_in = la.stride()[axis];
    let base_in = la.offset() as isize;
    let idx_stride_in = lidx.stride()[axis];
    let idx_base_in = lidx.offset() as isize;
    let axis_size_idx = lidx.shape()[axis];

    // rest axes (ascending order); the walk covers the broadcast rest shape:
    // a dim-1 rest axis of a tensor keeps its single slice (index 0)
    let rest_slots: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let rest_shape: Vec<usize> = rest_slots.iter().map(|&s| la.shape()[s].max(lidx.shape()[s])).collect();
    let stride_ref_a: &[isize] = la.stride().as_ref();
    let stride_ref_i: &[isize] = lidx.stride().as_ref();
    let total: usize = rest_shape.iter().product();

    // Per-rest tables: input offset, index offset, and the output block base.
    let mut a_base = vec![0_isize; total];
    let mut i_base = vec![0_isize; total];
    let mut out_base = vec![0_usize; total];
    let mut rest_multi = vec![0_usize; rest_shape.len()];
    for r in 0..total {
        let mut rem = r;
        for k in (0..rest_shape.len()).rev() {
            rest_multi[k] = rem % rest_shape[k];
            rem /= rest_shape[k];
        }
        let mut a_off = base_in;
        let mut i_off = idx_base_in;
        let mut o_off = 0_usize;
        for (k, &slot) in rest_slots.iter().enumerate() {
            let m = rest_multi[k];
            // per-tensor rest offsets: clamp broadcast (dim-1) axes to slice 0
            if la.shape()[slot] != 1 {
                a_off += stride_ref_a[slot] * m as isize;
            }
            if lidx.shape()[slot] != 1 {
                i_off += stride_ref_i[slot] * m as isize;
            }
            o_off += m * out_strides[slot];
        }
        a_base[r] = a_off;
        i_base[r] = i_off;
        out_base[r] = o_off;
    }

    // Flatten the (rest × axis) loop over disjoint output offsets, nesting the
    // smaller dimension inside (so no per-element division); both loops are
    // indexed, so either nesting covers the single-position case.
    let c_ptr = AtomicPtr::new(c.as_mut_ptr());
    let out_axis_stride = out_strides[axis];
    let (a_base, i_base, out_base) = (&a_base, &i_base, &out_base);
    let task = || {
        if total >= axis_size_idx {
            (0..total).into_par_iter().for_each(|r| {
                let cp = c_ptr.load(Ordering::Relaxed);
                let (base_a, base_i, base_out) = (a_base[r], i_base[r], out_base[r]);
                for j in 0..axis_size_idx {
                    let idx_pos = (base_i + idx_stride_in * j as isize) as usize;
                    let src_pos = (base_a + axis_stride_in * idx[idx_pos] as isize) as usize;
                    let dst = base_out + j * out_axis_stride;
                    // SAFETY: the tensor level validated every index within
                    // `0..la.shape()[axis]`; distinct `(rest, axis)` pairs map
                    // to disjoint output offsets.
                    unsafe {
                        cp.add(dst).write(MaybeUninit::new(a[src_pos].clone()));
                    }
                }
            });
        } else {
            (0..axis_size_idx).into_par_iter().for_each(|j| {
                let cp = c_ptr.load(Ordering::Relaxed);
                let dj = j * out_axis_stride;
                let ij = idx_stride_in * j as isize;
                for r in 0..total {
                    let idx_pos = (i_base[r] + ij) as usize;
                    let src_pos = (a_base[r] + axis_stride_in * idx[idx_pos] as isize) as usize;
                    let dst = out_base[r] + dj;
                    // SAFETY: as above.
                    unsafe {
                        cp.add(dst).write(MaybeUninit::new(a[src_pos].clone()));
                    }
                }
            });
        }
    };
    pool.expect("pool checked Some above").install(task);
    Ok(())
}
