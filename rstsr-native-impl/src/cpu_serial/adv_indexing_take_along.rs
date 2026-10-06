//! take_along_axis kernel (serial): gather with an index tensor along one
//! axis. Output shape = input shape with the axis length replaced by the
//! indices' axis length; `layout_c` is a contiguous layout of that shape.
//!
//! The index tensor `idx` must have the same rank as `a`, with every
//! non-axis dimension matching `a`'s shape (checked by the tensor level).

use crate::prelude_dev::*;

/// Gather `a` along `axis` using per-position indices from `idx`.
///
/// Both `a` and `idx` are visited in row-major order over their rest axes;
/// for each rest position the line's gathered values are written contiguously
/// into the output block for that rest position (via `out_strides`).
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

    // rest axes of the input (ascending order); both tensors share the rest
    // shape (tensor-level check)
    let rest_slots: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let rest_shape: Vec<usize> = rest_slots.iter().map(|&s| la.shape()[s]).collect();
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
        let a_off: isize =
            rest_slots.iter().zip(rest_multi.iter()).map(|(&slot, &v)| stride_ref_a[slot] * v as isize).sum::<isize>()
                + base_in as isize;
        let i_off: isize =
            rest_slots.iter().zip(rest_multi.iter()).map(|(&slot, &v)| stride_ref_i[slot] * v as isize).sum::<isize>()
                + idx_base_in as isize;
        let out_base: usize = rest_slots.iter().zip(rest_multi.iter()).map(|(&slot, &v)| v * out_strides[slot]).sum();
        let out_axis_stride = out_strides[axis];
        for j in 0..axis_size_idx {
            // SAFETY: the tensor level validated every index within
            // `0..la.shape()[axis]`; line offsets are input-stride
            // dot-products over in-range rest indices.
            let idx_pos = (i_off + idx_stride_in * j as isize) as usize;
            let src_pos = (a_off + axis_stride_in * idx[idx_pos] as isize) as usize;
            let src = a[src_pos].clone();
            let dst = out_base + j * out_axis_stride;
            c[dst].write(src);
        }
    }
    Ok(())
}
