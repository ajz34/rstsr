//! take_along_axis kernel (serial): gather with an index tensor along one
//! axis. Output shape = broadcast of the rest shapes with the indices' axis
//! length; `layout_c` is a contiguous layout of that shape.
//!
//! The index tensor `idx` must have the same rank as `a`; non-axis dims are
//! broadcast-compatible (dim 1 broadcasts; checked by the tensor level).

use crate::prelude_dev::*;

/// Gather `a` along `axis` using per-position indices from `idx`.
///
/// Rest positions are enumerated over the broadcast rest shape (dim-1 rest
/// axes of either tensor reuse their single slice); for each rest position
/// the line's gathered values are written into the output block for that
/// position (via `out_strides`).
#[allow(clippy::too_many_arguments)]
pub fn take_along_axis_cpu_serial<T, DA, DI>(
    c: &mut [MaybeUninit<T>],
    out_strides: &[usize],
    a: &[T],
    la: &Layout<DA>,
    idx: &[usize],
    lidx: &Layout<DI>,
    axis: usize,
) -> Result<()>
where
    T: Clone,
    DA: DimAPI,
    DI: DimAPI,
{
    let ndim = la.ndim();
    rstsr_check_axis!(axis as isize, ndim)?;
    let axis_stride_in = la.stride()[axis];
    let base_in = la.offset();
    let idx_stride_in = lidx.stride()[axis];
    let idx_base_in = lidx.offset();
    let axis_size_idx = lidx.shape()[axis];

    // rest axes (ascending order); the walk covers the broadcast rest shape:
    // a dim-1 rest axis of a tensor keeps its single slice (index 0)
    let rest_slots: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let rest_shape: Vec<usize> = rest_slots.iter().map(|&s| la.shape()[s].max(lidx.shape()[s])).collect();
    let stride_ref_a: &[isize] = la.stride().as_ref();
    let stride_ref_i: &[isize] = lidx.stride().as_ref();
    let total: usize = rest_shape.iter().product();

    for rest_index in 0..total {
        // unravel the rest position (row-major)
        let mut rem = rest_index;
        let mut rest_multi: Vec<usize> = vec![0; rest_shape.len()];
        for i in (0..rest_shape.len()).rev() {
            rest_multi[i] = rem % rest_shape[i];
            rem /= rest_shape[i];
        }
        // per-tensor rest offsets: clamp broadcast (dim-1) axes to slice 0
        let mut a_multi = rest_multi.clone();
        let mut i_multi = rest_multi.clone();
        for (k, &slot) in rest_slots.iter().enumerate() {
            if la.shape()[slot] == 1 {
                a_multi[k] = 0;
            }
            if lidx.shape()[slot] == 1 {
                i_multi[k] = 0;
            }
        }
        let a_off: isize =
            rest_slots.iter().zip(a_multi.iter()).map(|(&slot, &v)| stride_ref_a[slot] * v as isize).sum::<isize>()
                + base_in as isize;
        let i_off: isize =
            rest_slots.iter().zip(i_multi.iter()).map(|(&slot, &v)| stride_ref_i[slot] * v as isize).sum::<isize>()
                + idx_base_in as isize;
        let out_base: usize = rest_slots.iter().zip(rest_multi.iter()).map(|(&slot, &v)| v * out_strides[slot]).sum();
        let out_axis_stride = out_strides[axis];
        for j in 0..axis_size_idx {
            // SAFETY: the tensor level validated every index within
            // `0..la.shape()[axis]`; line offsets are input-stride
            // dot-products over in-range rest indices (broadcast dims
            // clamped to their single slice).
            let idx_pos = (i_off + idx_stride_in * j as isize) as usize;
            let src_pos = (a_off + axis_stride_in * idx[idx_pos] as isize) as usize;
            let src = a[src_pos].clone();
            let dst = out_base + j * out_axis_stride;
            c[dst].write(src);
        }
    }
    Ok(())
}

/// Scatter (inverse of [`take_along_axis_cpu_serial`]): for every position,
/// write the value from `values` into `a` at `indices` along `axis`, casting
/// `TA` to `TC`.
///
/// `a` and `indices` share their rest shape; `lvalues` is the values layout
/// broadcast to the indices' shape (a dim-1 rest axis reuses its single slice).
/// Duplicate targets write the same slot more than once; the visit order decides
/// the winner — the last write wins.
#[allow(clippy::too_many_arguments)]
pub fn put_along_axis_promote_cpu_serial<TC, TA, DA, DI>(
    a: &mut [TC],
    la: &Layout<DA>,
    idx: &[usize],
    lidx: &Layout<DI>,
    values: &[TA],
    lvalues: &Layout<DI>,
    axis: usize,
) -> Result<()>
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
    DA: DimAPI,
    DI: DimAPI,
{
    let ndim = la.ndim();
    rstsr_check_axis!(axis as isize, ndim)?;
    let axis_stride_a = la.stride()[axis];
    let base_a = la.offset();
    let idx_stride = lidx.stride()[axis];
    let idx_base = lidx.offset();
    let val_stride = lvalues.stride()[axis];
    let val_base = lvalues.offset();
    let axis_size_idx = lidx.shape()[axis];

    // rest axes (ascending order); the walk covers the indices' rest shape
    let rest_slots: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let rest_shape: Vec<usize> = rest_slots.iter().map(|&s| lidx.shape()[s]).collect();
    let stride_a = la.stride().as_ref();
    let stride_i = lidx.stride().as_ref();
    let stride_v = lvalues.stride().as_ref();
    let total: usize = rest_shape.iter().product();

    for rest_index in 0..total {
        let mut rem = rest_index;
        let mut rest_multi: Vec<usize> = vec![0; rest_shape.len()];
        for i in (0..rest_shape.len()).rev() {
            rest_multi[i] = rem % rest_shape[i];
            rem /= rest_shape[i];
        }
        let a_off: isize =
            rest_slots.iter().zip(rest_multi.iter()).map(|(&s, &v)| stride_a[s] * v as isize).sum::<isize>()
                + base_a as isize;
        let i_off: isize =
            rest_slots.iter().zip(rest_multi.iter()).map(|(&s, &v)| stride_i[s] * v as isize).sum::<isize>()
                + idx_base as isize;
        // values broadcast to the indices' shape: a dim-1 rest axis reuses its
        // single slice
        let v_off: isize = rest_slots
            .iter()
            .zip(rest_multi.iter())
            .map(|(&s, &v)| {
                let m = if lvalues.shape()[s] == 1 { 0 } else { v };
                stride_v[s] * m as isize
            })
            .sum::<isize>()
            + val_base as isize;
        for j in 0..axis_size_idx {
            let idx_pos = (i_off + idx_stride * j as isize) as usize;
            // SAFETY: the tensor level validated every index within
            // `0..la.shape()[axis]`; the destination offset is a
            // destination-stride dot-product over in-range rest indices.
            let dst = (a_off + axis_stride_a * idx[idx_pos] as isize) as usize;
            let src = (v_off + val_stride * j as isize) as usize;
            a[dst] = values[src].clone().into_cast();
        }
    }
    Ok(())
}
