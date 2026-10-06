//! Sorting kernels (serial): per-axis-line sort and argsort.

use crate::prelude_dev::*;
use core::cmp::Ordering;

/// Order a 1-D line of `(value, original_axis_position)` pairs by `f`.
///
/// Stable: ties (under `f`) keep the original position order. `descending`
/// reverses the value comparison for non-NaN values only; NaN (as flagged by
/// `is_nan`) keeps its ascending relation (last), matching NumPy, which sorts
/// NaN to the end in both directions.
#[allow(clippy::type_complexity)]
pub fn sort_line_cpu_serial<T, F>(pairs: &mut [(T, usize)], f: &F, descending: bool, is_nan: &dyn Fn(&T) -> bool)
where
    F: Fn(&T, &T) -> Ordering,
{
    if descending {
        pairs.sort_by(|a, b| {
            let (a_nan, b_nan) = (is_nan(&a.0), is_nan(&b.0));
            match (a_nan, b_nan) {
                // NaN keeps its ascending "greater than everything" relation
                (true, true) => a.1.cmp(&b.1),
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                // non-NaN values: flip the comparison, ties keep input order
                (false, false) => match f(&a.0, &b.0) {
                    Ordering::Equal => a.1.cmp(&b.1),
                    Ordering::Less => Ordering::Greater,
                    Ordering::Greater => Ordering::Less,
                },
            }
        });
    } else {
        pairs.sort_by(|a, b| match f(&a.0, &b.0) {
            Ordering::Equal => a.1.cmp(&b.1),
            other => other,
        });
    }
}

/// Multi-index increment in row-major order; returns false when the index
/// wraps past the last position (iteration complete).
fn ndindex_next(index: &mut [usize], shape: &[usize]) -> bool {
    for i in (0..index.len()).rev() {
        index[i] += 1;
        if index[i] < shape[i] {
            return true;
        }
        index[i] = 0;
    }
    false
}

/// Per-line sort/argsort over `la` along `axis`.
///
/// For every line (fixing all non-`axis` positions), the `(value,
/// axis_position)` pairs are sorted stably by `f` (descending reverses the
/// value comparison only; the NaN policy lives in the comparator). Sorted
/// values are written to `c`, sorted original axis positions to `idx` — both
/// slices are addressed by `layout_c`, a contiguous layout of the same shape
/// as `la` (output positions computed through its actual strides, so any
/// contiguous order works).
pub fn sort_axes_cpu_serial<T, D, F>(
    mut c: Option<&mut [MaybeUninit<T>]>,
    mut idx: Option<&mut [MaybeUninit<usize>]>,
    layout_c: &Layout<IxD>,
    a: &[T],
    la: &Layout<D>,
    axis: usize,
    f: &F,
    descending: bool,
    is_nan: &(dyn Fn(&T) -> bool + Send + Sync),
) -> Result<()>
where
    T: Clone,
    D: DimAPI,
    F: Fn(&T, &T) -> Ordering,
{
    let ndim = la.ndim();
    rstsr_check_axis!(axis as isize, ndim)?;
    let axis_size = la.shape()[axis];
    // contiguous output strides; the sorted-axis slot is `axis`
    let stride_ref: &[isize] = layout_c.stride().as_ref();
    let out_strides: Vec<usize> = stride_ref.iter().map(|&s| s.unsigned_abs()).collect();

    // the rest layout of the input: all non-axis positions, walked row-major
    // over the rest multi-index; each line's storage offset is the input's
    // own stride dot-product over the rest slots
    let (layout_axes, _layout_rest) = la.dim_split_axes(&[axis as isize])?;
    let rest_slots: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let rest_shape: Vec<usize> = rest_slots.iter().map(|&s| la.shape()[s]).collect();
    let stride_ref_in: &[isize] = la.stride().as_ref();
    let line_base = la.offset();
    let mut rest_multi: Vec<usize> = vec![0; rest_shape.len()];
    let mut layout_line = layout_axes.clone();
    let mut pairs: Vec<(T, usize)> = Vec::with_capacity(axis_size);

    loop {
        let idx_rest: isize =
            rest_slots.iter().zip(rest_multi.iter()).map(|(&slot, &v)| stride_ref_in[slot] * v as isize).sum::<isize>()
                + line_base as isize;
        // SAFETY: `idx_rest` is the input's own stride dot-product over
        // in-range rest indices + offset — an in-bounds element address.
        unsafe { layout_line.set_offset(idx_rest as usize) };
        let iter_line = IndexedIterLayout::new(&layout_line, RowMajor)?;
        pairs.clear();
        for (index, off) in iter_line {
            let index_ref: &[usize] = index.as_ref();
            let axis_position = index_ref[0];
            pairs.push((a[off].clone(), axis_position));
        }
        sort_line_cpu_serial(&mut pairs, f, descending, is_nan);

        let rest_part: usize = rest_slots.iter().zip(rest_multi.iter()).map(|(&slot, &v)| v * out_strides[slot]).sum();
        let axis_stride = out_strides[axis];
        for (j, (value, axis_position)) in pairs.iter().enumerate() {
            let out_pos = rest_part + j * axis_stride;
            if let Some(c) = c.as_deref_mut() {
                c[out_pos].write(value.clone());
            }
            if let Some(idx_out) = idx.as_deref_mut() {
                idx_out[out_pos].write(*axis_position);
            }
        }
        if !ndindex_next(&mut rest_multi, &rest_shape) {
            break;
        }
    }
    Ok(())
}
