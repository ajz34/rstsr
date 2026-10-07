//! Sorting kernels (rayon): parallel per-line sort/argsort; each line is
//! sorted serially, lines are distributed over the pool.

use crate::cpu_serial::sorting::{sort_axes_cpu_serial, sort_line_cpu_serial};
use crate::prelude_dev::*;
use core::cmp::Ordering;
use core::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};

// Sort workload is per-line `O(axis_size log axis_size)`; parallelize when
// the total element count passes this (8 KiB of f64), mirroring reduction.
const PARALLEL_SWITCH: usize = 1024;

/// Rayon twin of [`sort_axes_cpu_serial`]: identical output; lines are
/// visited in parallel when the workload is large enough.
#[allow(clippy::too_many_arguments)]
pub fn sort_axes_cpu_rayon<T, D, F>(
    c: Option<&mut [MaybeUninit<T>]>,
    idx: Option<&mut [MaybeUninit<usize>]>,
    layout_c: &Layout<IxD>,
    a: &[T],
    la: &Layout<D>,
    axis: usize,
    f: &F,
    descending: bool,
    is_nan: &(dyn Fn(&T) -> bool + Send + Sync),
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering + Send + Sync,
{
    // if not in pool environment or too small, use serial
    if pool.is_none() || la.size() < PARALLEL_SWITCH {
        return sort_axes_cpu_serial(c, idx, layout_c, a, la, axis, f, descending, is_nan);
    }

    let ndim = la.ndim();
    rstsr_check_axis!(axis as isize, ndim)?;
    let axis_size = la.shape()[axis];

    // split the layout into the sorted axis and the rest; parallelize over
    // the rest multi-index (row-major), each line's storage offset computed
    // from the input's own strides
    let (layout_axes, _layout_rest) = la.dim_split_axes(&[axis as isize])?;
    let rest_slots: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let rest_shape: Vec<usize> = rest_slots.iter().map(|&s| la.shape()[s]).collect();
    let stride_ref_in: &[isize] = la.stride().as_ref();
    let line_base = la.offset();
    let rest_total: usize = rest_shape.iter().product();

    // contiguous output strides for output position computation
    let stride_ref: &[isize] = layout_c.stride().as_ref();
    let out_strides: Vec<usize> = stride_ref.iter().map(|&s| s.unsigned_abs()).collect();
    let axis_stride = out_strides[axis];

    // pass mutable references through the parallel region as raw pointers
    // (house pattern: base-pointer hoist through AtomicPtr; disjoint writes);
    // None is encoded as a null pointer
    let c_ptr = AtomicPtr::new(c.map_or(core::ptr::null_mut(), |s| s.as_mut_ptr()));
    let idx_ptr = AtomicPtr::new(idx.map_or(core::ptr::null_mut(), |s| s.as_mut_ptr()));

    // per-rest-digit strides, hoisted once (input's own strides over rest slots)
    let rest_strides_in: Vec<isize> = rest_slots.iter().map(|&s| stride_ref_in[s]).collect();

    let task = || {
        (0..rest_total).into_par_iter().try_for_each(|rest_flat: usize| -> Result<()> {
            // row-major unravel of the rest position, digits computed on the fly
            let mut rest_flat_rem = rest_flat;
            let idx_rest: isize = rest_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    let v = rest_flat_rem % dim;
                    rest_flat_rem /= dim;
                    rest_strides_in[i] * v as isize
                })
                .sum::<isize>()
                + line_base as isize;
            let mut layout_line = layout_axes.clone();
            // SAFETY: input-stride dot-product over in-range rest indices.
            unsafe { layout_line.set_offset(idx_rest as usize) };
            let iter_line = IndexedIterLayout::new(&layout_line, RowMajor)?;
            let mut pairs: Vec<(T, usize)> = Vec::with_capacity(axis_size);
            for (index, off) in iter_line {
                let index_ref: &[usize] = index.as_ref();
                let axis_position = index_ref[0];
                pairs.push((a[off].clone(), axis_position));
            }
            sort_line_cpu_serial(&mut pairs, f, descending, is_nan);

            let mut rest_flat_rem = rest_flat;
            let rest_part: usize = rest_shape
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    let v = rest_flat_rem % dim;
                    rest_flat_rem /= dim;
                    v * out_strides[rest_slots[i]]
                })
                .sum();
            for (j, (value, axis_position)) in pairs.iter().enumerate() {
                let out_pos = rest_part + j * axis_stride;
                // SAFETY: base pointers hoisted through AtomicPtr (relaxed
                // load; never reassigned). Lines write disjoint output
                // blocks: `rest_part` is unique per line and positions
                // `rest_part + j * axis_stride` within it are disjoint.
                unsafe {
                    let cp = c_ptr.load(AtomicOrdering::Relaxed);
                    if !cp.is_null() {
                        cp.add(out_pos).write(MaybeUninit::new(value.clone()));
                    }
                    let ip = idx_ptr.load(AtomicOrdering::Relaxed);
                    if !ip.is_null() {
                        ip.add(out_pos).write(MaybeUninit::new(*axis_position));
                    }
                }
            }
            Ok(())
        })
    };
    // pool is Some here (checked above); install into it
    pool.expect("pool checked Some above").install(task)
}
