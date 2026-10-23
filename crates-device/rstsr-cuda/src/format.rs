//! Element addressing for printing CUDA tensors.
//!
//! The traversal must mirror `rstsr-native-impl`'s
//! `print_vec_with_layout_dfs_inner` exactly (row-major DFS with
//! `min_print`/`max_print` truncation).
//!
//! DEFER: whole-module stopgap. Printing fetches elements individually on the
//! host path; to be reworked later in a not-copy-to-CPU way.

use rstsr_core::prelude_dev::*;

/// Flat (layout-relative) indices the CPU walker prints for `layout` under
/// the given limits, in visit order.
pub(crate) fn touched_element_indices<D>(layout: &Layout<D>, max_print: usize, min_print: usize) -> Vec<usize>
where
    D: DimAPI,
{
    let mut out = vec![];
    let offset = layout.offset();
    let shape = layout.shape().as_ref().to_vec();
    let stride = layout.stride().as_ref().to_vec();
    let ndim = shape.len();
    if ndim == 0 {
        out.push(offset);
        return out;
    }
    // DFS over all but the last axis, mirroring the walker's truncation on
    // the last axis; non-last axes are printed fully by the CPU walker.
    let mut idx = vec![0usize; ndim.saturating_sub(1)];
    let nlast = shape[ndim - 1];
    let stride_last = stride[ndim - 1];
    let full_last = nlast <= max_print.max(2 * min_print + 1);
    loop {
        // prefix offset for this multi-index
        let mut base = 0isize;
        for (i, &v) in idx.iter().enumerate() {
            base += v as isize * stride[i];
        }
        let flat = |i: usize| -> usize { (offset as isize + base + i as isize * stride_last) as usize };
        if full_last || nlast == 0 {
            for i in 0..nlast {
                out.push(flat(i));
            }
        } else {
            for i in 0..min_print {
                out.push(flat(i));
            }
            for i in (nlast - min_print)..nlast {
                out.push(flat(i));
            }
        }
        // increment the multi-index (row-major over the prefix axes), but
        // skip truncated ranges on non-last axes the same way the walker does
        // (the CPU walker prints all rows; only the last axis truncates).
        let mut done = false;
        for ax in (0..idx.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < shape[ax] {
                done = true;
                break;
            }
            idx[ax] = 0;
        }
        if !done {
            break;
        }
    }
    out
}

/// The original layout, dimensioned dynamically (offset/strides preserved)
/// so the shared walker walks it identically over the full-size staged vec.
pub(crate) fn staged_layout_of<D: DimAPI>(layout: &Layout<D>) -> Layout<IxD> {
    unsafe { Layout::new_unchecked(layout.shape().clone().into(), layout.stride().clone().into(), layout.offset()) }
}
