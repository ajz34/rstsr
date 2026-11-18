//! Gather/scatter along one axis: `index_select` and `index_put`.
//!
//! # What this is
//!
//! ```text
//! gather   c[..., k, ...] = a[..., indices[k], ...]      (index_select)
//! scatter  a[..., indices[k], ...] = v[..., k, ...]      (index_put)
//! ```
//!
//! Both are "one axis is addressed indirectly, every other axis is copied verbatim". That is why
//! they share a file: the loop is identical and only the direction of the final store changes.
//!
//! # Why the index vector is a host slice, and why that is the honest shape of this kernel
//!
//! The trait takes `indices: &[usize]` — the indices are already on the host when the call arrives.
//! So they are uploaded once per call and the kernel indexes them from device memory. That costs
//! `8 * n_indices` bytes of H2D, and it buys the two checks that a device-side index buffer cannot
//! give cheaply:
//!
//! - **range**: `max(indices) < a.shape[axis]`. Without it, an out-of-range index is an out-of-bounds
//!   read (gather) or write (scatter), and a GPU has no trap for that — it reads whatever is next in
//!   the pool. The CPU raises `IndexError` here; so do we, before launching anything.
//! - **duplicates**: only for the scatter. The CPU applies `index_put` sequentially over `k`, so a
//!   repeated index means *last write wins*. On a GPU those writes race, so the answer would depend
//!   on scheduling. Rather than pretend, duplicate indices are refused by name. `index_select` has
//!   no such problem — a repeated gather simply reads the same element twice.
//!
//! `take_along_axis` is *not* here, and the reason is the same checks: its index buffer lives on the
//! device, so validating range would need a full read-back on every call. That belongs with a caller
//! that needs it; nothing in the current programs does.
//!
//! # Addressing
//!
//! One thread per output element. The linear index is decomposed against the output shape and the
//! per-axis strides of *both* operands, with the indirect axis read from the index vector:
//!
//! ```text
//! for ax in (0..rank).rev():
//!     i  = lin % shape[ax]; lin /= shape[ax]
//!     off_out += i * out_stride[ax]
//!     if ax != axis: off_in += i * in_stride[ax]
//!     else:          k = i
//! off_in += indices[k] * axis_stride
//! ```
//!
//! That is `O(rank)` per element, against `O(1)` for a contiguous fast path. The elementwise kernels
//! in this crate make the same trade in their rank-generic fallback (`rstsr_index_to_offset`), and
//! here it is the only path: the axis being indirect means the "contiguous" case still has to split
//! the index into outer/axis/inner. A blocked fast path is a later optimisation, not a correctness
//! question.

use crate::dtype::CudaDType;
use crate::prelude_dev::*;

use super::reduce_unary::{offset_ptr, render};

/// `@TO@` is the element type; the index vector is `usize`, which `CudaDType` spells
/// `unsigned long long` — the same width as `rstsr_usz`, so no conversion is needed on device.
const GATHER_ENTRY: &str = r#"
extern "C" __global__ void @NAME@(
    @TO@* __restrict__ c,
    const @TO@* __restrict__ a,
    const unsigned long long* __restrict__ idx,
    int rank,
    const rstsr_usz* shape,
    const rstsr_usz* cstrides,   // output strides, one per axis
    rstsr_usz coff,
    const rstsr_usz* astrides,   // input strides, one per axis
    rstsr_usz aoff,
    int axis,                    // the indirectly addressed axis
    rstsr_isz axis_step,         // input element step along that axis
    rstsr_usz n)                 // number of output elements
{
    const rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) { return; }

    rstsr_usz rest = lin, oc = coff, oa = aoff;
    rstsr_usz k = 0;
    for (int ax = rank - 1; ax >= 0; --ax) {
        const rstsr_usz i = rest % shape[ax];
        rest /= shape[ax];
        oc += i * (rstsr_usz)cstrides[ax];
        if (ax == axis) {
            k = i;                    // the axis position indexes the index vector
        } else {
            oa += i * (rstsr_usz)astrides[ax];
        }
    }
    oa += (rstsr_usz)idx[k] * (rstsr_usz)axis_step;
    c[oc] = a[oa];
}
"#;

/// The scatter is the same loop with the store reversed and a cast on the value, because
/// `DeviceIndexPutAPI<TC, TA>` allows the value to have a different dtype than the destination.
const SCATTER_ENTRY: &str = r#"
extern "C" __global__ void @NAME@(
    @TC@* __restrict__ a,
    const @TV@* __restrict__ v,
    const unsigned long long* __restrict__ idx,
    int rank,
    const rstsr_usz* shape,
    const rstsr_usz* vstrides,
    rstsr_usz voff,
    const rstsr_usz* astrides,
    rstsr_usz aoff,
    int axis,
    rstsr_isz axis_step,
    rstsr_usz n)
{
    const rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) { return; }

    rstsr_usz rest = lin, ov = voff, oa = aoff;
    rstsr_usz k = 0;
    for (int ax = rank - 1; ax >= 0; --ax) {
        const rstsr_usz i = rest % shape[ax];
        rest /= shape[ax];
        ov += i * (rstsr_usz)vstrides[ax];
        if (ax == axis) {
            k = i;
        } else {
            oa += i * (rstsr_usz)astrides[ax];
        }
    }
    oa += (rstsr_usz)idx[k] * (rstsr_usz)axis_step;
    a[oa] = @CAST@;
}
"#;

/// Upload a host index vector. Stream-ordered, so the kernel that follows sees it.
pub(crate) fn upload_indices(device: &DeviceCuda, indices: &[usize]) -> Result<CudaBuffer<usize>> {
    crate::transfer::h2d_full(device, indices.to_vec())
}

/// Copy `n` elements of `a` into `c`, addressing `axis` through `indices`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_launch<T>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<T>>,
    lc: &Layout<IxD>,
    a: &CudaBuffer<T>,
    la: &Layout<IxD>,
    axis: usize,
    indices: &CudaBuffer<usize>,
    n: usize,
) -> Result<()>
where
    T: CudaDType,
{
    use crate::kernels_ops::{FlatLayout, LayoutPack};

    if n == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let _ga = crate::kernel::order_input(a, exec)?;

    let body = render(
        GATHER_ENTRY,
        &[
            ("@NAME@", "rstsr_index_select"),
            ("@TO@", T::CNAME),
        ],
    );
    let src = crate::kernel::assemble_source(T::INCLUDES, &body);
    let f = device.ordinal_obj().kernel_function(&crate::kernel::make_key(
        device.ordinal_obj(),
        "rstsr_index_select",
        &src,
        &[],
    )?)?;

    let lc_f = FlatLayout::of(lc);
    let la_f = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lc_f, true), (&la_f, false)])?;

    // SAFETY: layout-identical `MaybeUninit` transmute; each output element is written once.
    let c_typed: &mut CudaBuffer<T> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<T>> as *mut CudaBuffer<T>) };

    crate::kernel::FlatLaunch::new(exec, &f, n)
        .arg_buf(c_typed)
        .arg_buf(a)
        .arg_u64(offset_ptr(exec, indices, 0))
        .arg_i32(lc.ndim() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lc_f.offset)
        .arg_u64(pack.strides[1])
        .arg_usize(la_f.offset)
        .arg_i32(axis as i32)
        .arg_bytes(&la.stride()[axis].to_ne_bytes())
        .arg_usize(n)
        .launch()?;
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

/// Copy `n` elements of `v` into `a`, addressing `axis` through `indices`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_launch<TC, TV>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<TC>,
    la: &Layout<IxD>,
    axis: usize,
    indices: &CudaBuffer<usize>,
    v: &CudaBuffer<TV>,
    lv: &Layout<IxD>,
    n: usize,
) -> Result<()>
where
    TC: CudaDType,
    TV: CudaDType,
{
    use crate::kernels_ops::{FlatLayout, LayoutPack};

    if n == 0 {
        return Ok(());
    }
    let exec = device.stream();
    // The destination is written in place, so it needs both markers: `order_input` before, in case
    // the buffer came from another stream, and `order_output` after, so a later read knows it was
    // produced here (the convention `binary_assign_generic` follows).
    let _ga = crate::kernel::order_input(a, exec)?;
    let _gv = crate::kernel::order_input(v, exec)?;

    let cast = crate::kernels_ops::cast_expr_c(TV::CNAME, TC::CNAME, "v[ov]")?;
    let body = render(
        SCATTER_ENTRY,
        &[
            ("@NAME@", "rstsr_index_put"),
            ("@TC@", TC::CNAME),
            ("@TV@", TV::CNAME),
            ("@CAST@", cast.as_str()),
        ],
    );
    let mut includes: Vec<&'static str> = TV::INCLUDES.to_vec();
    for inc in TC::INCLUDES.iter().copied() {
        if !includes.contains(&inc) {
            includes.push(inc);
        }
    }
    let src = crate::kernel::assemble_source(&includes, &body);
    let f = device.ordinal_obj().kernel_function(&crate::kernel::make_key(
        device.ordinal_obj(),
        "rstsr_index_put",
        &src,
        &[],
    )?)?;

    let lv_f = FlatLayout::of(lv);
    let la_f = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lv_f, true), (&la_f, false)])?;

    crate::kernel::FlatLaunch::new(exec, &f, n)
        .arg_buf(a)
        .arg_buf(v)
        .arg_u64(offset_ptr(exec, indices, 0))
        .arg_i32(lv.ndim() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lv_f.offset)
        .arg_u64(pack.strides[1])
        .arg_usize(la_f.offset)
        .arg_i32(axis as i32)
        .arg_bytes(&la.stride()[axis].to_ne_bytes())
        .arg_usize(n)
        .launch()?;
    let _go = crate::kernel::order_output(a, exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(template: &str, subs: &[(&str, &str)]) -> String {
        render(template, subs)
    }

    /// Render both kernels and read them: no unsubstituted token, and the indirect axis must appear
    /// in exactly the two places that matter (the `k` capture and the final offset add).
    #[test]
    fn kernels_render_completely() {
        let g = rendered(GATHER_ENTRY, &[("@NAME@", "rstsr_index_select"), ("@TO@", "double")]);
        assert!(!g.contains('@'), "gather: unsubstituted token");
        // `make_key` checks this at launch; asserting it here means the check is verified without a GPU.
        assert!(g.contains("void rstsr_index_select("), "gather: entry symbol missing");
        assert_eq!(g.matches("k = i;").count(), 1, "gather: axis capture");
        assert_eq!(g.matches("idx[k]").count(), 1, "gather: index read");

        let s = rendered(
            SCATTER_ENTRY,
            &[("@NAME@", "rstsr_index_put"), ("@TC@", "double"), ("@TV@", "float"), ("@CAST@", "(double)(v[ov])")],
        );
        assert!(!s.contains('@'), "scatter: unsubstituted token");
        assert!(s.contains("void rstsr_index_put("), "scatter: entry symbol missing");
        assert_eq!(s.matches("k = i;").count(), 1, "scatter: axis capture");
        assert!(s.contains("(double)(v[ov])"), "scatter: the value cast must be the emitted one");
    }

    /// The scatter's cast must come from `cast_expr_c`, not a bare C cast — the two disagree on
    /// float→int and on complex, and the crate already made that choice in one place.
    #[test]
    fn scatter_casts_through_the_shared_expression() {
        let cast = crate::kernels_ops::cast_expr_c("double", "float", "v[ov]").unwrap();
        assert!(cast.contains("v[ov]"), "{cast}");
        assert!(cast.contains("float"), "{cast}");
    }
}
