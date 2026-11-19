//! `index_select` and `index_put` on CUDA.
//!
//! ```text
//! index_select   c[..., k, ...] = a[..., indices[k], ...]
//! index_put      a[..., indices[k], ...] = v[..., k, ...]
//! ```
//!
//! Both are one kernel (the `kernels::adv_index` module) because both are "one axis addressed
//! indirectly, everything else copied". What differs is the direction of the store and the checks
//! this module performs before launching.
//!
//! # Why the checks are on the host, and why they are not optional
//!
//! The trait hands us `indices` as a host slice, so the two properties that decide whether the kernel
//! is *safe* — every index in range, and (for the scatter) no index repeated — can be established
//! before a single thread runs:
//!
//! - **range**: an out-of-range index is an out-of-bounds read (select) or **write** (put), which on
//!   a GPU silently touches whatever the caching allocator put next door. There is no fault for it.
//!   The CPU raises `IndexError`; this raises before launching.
//! - **duplicates**: the CPU's `index_put` loops over `k` and writes, so a repeated index means
//!   *last write wins*. Threads have no order, so instead of a race that resolves differently on
//!   each run, duplicates are refused by name. `index_select` is exempt: gathering the same element
//!   twice is well defined.
//!
//! The duplicate check costs a sort of the index vector, `O(n log n)` on the host, once per call.
//! That is the same order as the H2D that uploads it and small next to any real scatter, but it is
//! a real cost and the reason `index_select` does not pay it.
//!
//! # What is absent
//!
//! - `take_along_axis` / `put_along_axis`: the index buffer lives on the *device*, so the range check
//!   above would need a read-back on every call. Deferred until something needs it — the same rule
//!   that keeps `sort`/`unique` out.
//! - `mask_select` / `mask_fill`: `mask_fill` is a `where` and could reuse the elementwise path;
//!   `mask_select` is a compaction and needs a prefix sum over the mask to be deterministic. Both are
//!   a separate increment.

#![allow(clippy::type_complexity)]

use core::mem::MaybeUninit;

use crate::dtype::CudaDType;
use crate::kernels::adv_index::{gather_launch, scatter_launch, upload_indices};
use crate::prelude_dev::*;

/// Shapes must agree on every axis except the indirectly addressed one, or the copy would read and
/// write unrelated elements. The CPU relies on the caller for this; the GPU checks it, because the
/// failure mode here is memory corruption rather than an error.
fn check_rest_shapes(dst: &[usize], src: &[usize], axis: usize, what: &str) -> Result<()> {
    if dst.len() != src.len() {
        rstsr_raise!(InvalidLayout, "{what}: rank mismatch {} vs {}", dst.len(), src.len())?;
    }
    for ax in 0..dst.len() {
        if ax != axis && dst[ax] != src[ax] {
            rstsr_raise!(
                InvalidLayout,
                "{what}: axis {ax} is {} on one side and {} on the other; only axis {axis} may differ",
                dst[ax],
                src[ax]
            )?;
        }
    }
    Ok(())
}

/// `index_select`: no ordering requirement (a gather has no side effects), so duplicates are fine.
impl<T, D> DeviceIndexSelectAPI<T, D> for DeviceCuda
where
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType,
{
    fn index_select(
        &self,
        c: &mut CudaBuffer<MaybeUninit<T>>,
        lc: &Layout<D>,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axis: usize,
        indices: &[usize],
    ) -> Result<()> {
        let (lc, la) = (lc.to_dim::<IxD>()?, la.to_dim::<IxD>()?);
        let axis = rstsr_check_axis!(axis as isize, lc.ndim())?;
        check_rest_shapes(lc.shape(), la.shape(), axis, "cuda index_select")?;
        if lc.shape()[axis] != indices.len() {
            rstsr_raise!(
                InvalidLayout,
                "cuda index_select: axis {axis} of the output is {} but {} indices were given",
                lc.shape()[axis],
                indices.len()
            )?;
        }
        // The CPU's own check, verbatim: an empty index list selects nothing, so there is no maximum
        // to test and an empty axis is legal.
        if let Some(&max) = indices.iter().max() {
            if max >= la.shape()[axis] {
                rstsr_raise!(
                    IndexError,
                    "cuda index_select: index {max} is out of range for axis {axis} of extent {}",
                    la.shape()[axis]
                )?;
            }
        }

        let n = lc.size();
        if n == 0 {
            return Ok(());
        }
        let idx = upload_indices(self, indices)?;
        gather_launch(self, c, &lc, a, &la, axis, &idx, n)
    }
}

/// `index_put`: writes in place, so duplicate indices are refused rather than raced.
impl<TC, TA> DeviceIndexPutAPI<TC, TA> for DeviceCuda
where
    Self: DeviceRawAPI<TC, Raw = CudaBuffer<TC>> + DeviceRawAPI<TA, Raw = CudaBuffer<TA>>,
    TC: CudaDType,
    TA: CudaDType,
{
    fn index_put(
        &self,
        a: &mut CudaBuffer<TC>,
        la: &Layout<IxD>,
        axis: usize,
        indices: &[usize],
        value: &CudaBuffer<TA>,
        lvalue: &Layout<IxD>,
    ) -> Result<()> {
        let axis = rstsr_check_axis!(axis as isize, la.ndim())?;
        check_rest_shapes(lvalue.shape(), la.shape(), axis, "cuda index_put")?;
        if lvalue.shape()[axis] != indices.len() {
            rstsr_raise!(
                InvalidLayout,
                "cuda index_put: the value has {} on axis {axis} but {} indices were given",
                lvalue.shape()[axis],
                indices.len()
            )?;
        }
        if let Some(&max) = indices.iter().max() {
            if max >= la.shape()[axis] {
                rstsr_raise!(
                    IndexError,
                    "cuda index_put: index {max} is out of range for axis {axis} of extent {}",
                    la.shape()[axis]
                )?;
            }
        }
        // The CPU writes sequentially over `k`, so a repeat means last-wins. Threads have no order.
        if indices.len() > 1 {
            let mut sorted = indices.to_vec();
            sorted.sort_unstable();
            if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
                rstsr_raise!(
                    InvalidValue,
                    "cuda index_put: index {} appears more than once; the CPU resolves repeats as \
                     last-write-wins, which is not expressible without an ordering between threads. \
                     Deduplicate the indices, or use the CPU device.",
                    w[0]
                )?;
            }
        }

        let n = lvalue.size();
        if n == 0 {
            return Ok(());
        }
        let idx = upload_indices(self, indices)?;
        scatter_launch(self, a, la, axis, &idx, value, lvalue, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rest-shape check is what stands between a wrong call and memory corruption, so its
    /// accept/reject boundary is pinned rather than inferred.
    #[test]
    fn rest_shapes_must_match_off_the_axis() {
        assert!(check_rest_shapes(&[4, 5, 6], &[4, 9, 6], 1, "t").is_ok());
        assert!(check_rest_shapes(&[4, 5, 6], &[4, 9, 7], 1, "t").is_err());
        assert!(check_rest_shapes(&[4, 5], &[4, 5, 6], 1, "t").is_err());
        // the axis itself may differ by any amount, including to zero
        assert!(check_rest_shapes(&[4, 0, 6], &[4, 9, 6], 1, "t").is_ok());
    }
}
