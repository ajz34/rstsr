//! `cumsum`/`cumprod`: the one reduction-shaped op whose every prefix is an output.
//!
//! # Why this is not a `ReducePlan` job
//!
//! A reduction produces one value per output and can reorder freely — that freedom is what the
//! butterfly and the `lanes` knob are for. A scan produces `n` values per line and reorders *nothing*:
//! `out[k]` must be `x[0] op … op x[k]`. So there is no plan here, and no `classify_reduce`: the
//! geometry is one block per line, and the arithmetic inside the block is a Hillis–Steele scan
//! followed by a per-warp carry.
//!
//! ```text
//! warp 0: [x0 x1 x2 ... x15]  -> inclusive scan within the warp
//! warp 1: [x16 ... x31]       -> inclusive scan within the warp
//!         wsum[w] = last lane of warp w            (each warp's total)
//!         wsum  := exclusive prefix of wsum, seeded by `carry`  (thread 0, serially)
//!         out   := x + wsum[warp]                  (each element gains every earlier warp)
//!         carry := total of the whole block        (seed for the next chunk)
//! ```
//!
//! The `carry` is what makes an arbitrarily long axis work: the block walks the line in chunks of
//! `BLOCK` and each chunk starts from the total of the previous one. There is no length cap and no
//! second kernel — the price is that the per-chunk carry is serial, so a 10⁶-long scan is ~4000
//! serial steps per line rather than a decoupled-scan's ~2. That is the honest trade for the shapes
//! REST uses (a few dozen elements along the scan axis), and it is recorded rather than hidden.
//!
//! # The line offset is computed, not assumed
//!
//! A line is "everything but the scan axis", so its base offset is a function of the rest
//! multi-index, and the input and output have *different* strides there (the output is a fresh
//! array with the scan axis possibly grown). So the kernel takes both rest layouts as
//! `(shape, strides, offset)` triples and decomposes `blockIdx.x` twice — the same
//! `rstsr_index_to_offset` the elementwise kernels use. That is why this handles any axis of any
//! layout without a contiguity precondition: rank-2, rank-3, transposed views and a nonzero offset
//! all resolve through the same helper.
//!
//! # What is deliberately not supported
//!
//! A complex operand. The instantaneous scan value is shuffled as a primitive register, and a
//! complex value is two registers, so it would need a second shuffle path in four places. The CPU
//! accepts complex; this refuses at compile time through the `CudaReal` bound rather than giving a
//! wrong answer.

use crate::dtype::CudaDType;
use crate::prelude_dev::*;

use super::reduce_unary::render;

/// Which scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScanOp {
    Sum,
    Prod,
}

impl ScanOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Sum => "cumsum",
            Self::Prod => "cumprod",
        }
    }

    /// The identity, which is what the out-of-range tail lanes carry and what `include_initial`
    /// writes at position 0. It must be the operation's neutral element or the carry is wrong.
    fn init(self) -> &'static str {
        match self {
            Self::Sum => "(@TO@)0",
            Self::Prod => "(@TO@)1",
        }
    }

    /// `RSTSR_SCAN_OP(x, y)`. Written as a macro body over the *parameters*, because a body that
    /// names something else is exactly the failure that cost a hardware round trip in
    /// [`super::reduce_pair`].
    fn op(self) -> &'static str {
        match self {
            Self::Sum => "((x) + (y))",
            Self::Prod => "((x) * (y))",
        }
    }
}

const SCAN_ENTRY: &str = r#"
#define RSTSR_SCAN_OP(x, y) (@OP@)

extern "C" __global__ void @NAME@(
    @TO@* __restrict__ out,
    const @TO@* __restrict__ a,
    int rrank,                       // rank of the "everything but the scan axis" layout
    const rstsr_usz* oshape,         // its shape (shared by input and output)
    const rstsr_usz* ostrides,       // output rest strides
    rstsr_usz ooffset,               // output rest base offset
    const rstsr_usz* istrides,       // input rest strides
    rstsr_usz ioffset,               // input rest base offset
    rstsr_usz size_m,                // scan length
    rstsr_isz step_in,               // input element step along the scan axis
    rstsr_isz step_out,              // output element step along the scan axis
    int include_initial)
{
    const rstsr_usz lin = (rstsr_usz)blockIdx.x;
    const rstsr_usz obase = rstsr_index_to_offset(lin, rrank, oshape, ostrides, ooffset);
    const rstsr_usz ibase = rstsr_index_to_offset(lin, rrank, oshape, istrides, ioffset);

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nw = (int)(blockDim.x >> 5);
    const rstsr_usz span = (rstsr_usz)blockDim.x;

    __shared__ @TO@ wsum[@MAXW@];
    __shared__ @TO@ carry;

    if (tid == 0) { carry = @INIT@; }

    // The prefix at position 0 is the identity, and it is written once — writing it inside the loop
    // would rewrite it for every chunk.
    if (include_initial && tid == 0) { out[obase] = @INIT@; }
    __syncthreads();

    for (rstsr_usz base = 0; base < size_m; base += span) {
        const rstsr_usz idx = base + (rstsr_usz)tid;
        // Tail lanes carry the identity and still take part: the warp scan is a shuffle, so a
        // `return` here would leave the warp's remaining lanes reading undefined values.
        @TO@ x = (idx < size_m) ? a[ibase + (rstsr_isz)idx * step_in] : (@TO@)(@INIT@);

        // Inclusive scan within the warp (`__shfl_up_sync` gives lane-off's value; lanes below
        // `off` must not take it).
        for (int off = 1; off < 32; off <<= 1) {
            const @TO@ y = __shfl_up_sync(0xffffffffu, x, off);
            if (lane >= off) { x = RSTSR_SCAN_OP(x, y); }
        }

        // Turn the per-warp totals into exclusive prefixes, seeded by the previous chunks' total.
        if (lane == 31) { wsum[wid] = x; }
        __syncthreads();
        if (tid == 0) {
            @TO@ s = carry;
            for (int w = 0; w < nw; ++w) {
                const @TO@ t = wsum[w];
                wsum[w] = s;
                s = RSTSR_SCAN_OP(s, t);
            }
            carry = s;
        }
        __syncthreads();
        x = RSTSR_SCAN_OP(x, wsum[wid]);

        if (idx < size_m) {
            const rstsr_usz o = obase
                + (rstsr_isz)(idx + (include_initial ? (rstsr_usz)1 : (rstsr_usz)0)) * step_out;
            out[o] = x;
        }
        // `wsum` and `carry` are reused by the next chunk; the barrier keeps a fast warp from
        // overwriting them while a slow one is still reading.
        __syncthreads();
    }
}
"#;

/// Render the entry. The substitution **order** is part of the contract: `@INIT@` expands to
/// `(@TO@)0`, so it must be substituted *before* `@TO@` or the inner token is never expanded and
/// NVRTC sees `@TO@`. This is the same rule `reduce_unary` documents, and it is why the launcher and
/// the unit test share this one function instead of each keeping its own list.
fn render_scan(op: ScanOp, name: &str, cname: &str) -> String {
    let maxw = super::reduce_unary::MAX_WARPS_PER_BLOCK.to_string();
    render(
        SCAN_ENTRY,
        &[
            ("@NAME@", name),
            ("@OP@", op.op()),
            ("@INIT@", op.init()),
            ("@TO@", cname),
            ("@MAXW@", maxw.as_str()),
        ],
    )
}

/// Launch one scan.
///
/// `lm` is the input's "everything but the scan axis" layout, `lo_rest` the output's; both carry
/// the offset of their own base pointer, and both describe the *same* multi-index set, which is what
/// makes `blockIdx.x` mean the same line on the two sides.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_launch<T>(
    device: &DeviceCuda,
    op: ScanOp,
    c: &mut CudaBuffer<MaybeUninit<T>>,
    lo_rest: &Layout<IxD>,
    a: &CudaBuffer<T>,
    lm: &Layout<IxD>,
    size_m: usize,
    step_in: isize,
    step_out: isize,
    include_initial: bool,
) -> Result<()>
where
    T: CudaDType,
{
    use crate::kernels_ops::{FlatLayout, LayoutPack};

    let n_lines = lm.size();
    if n_lines == 0 {
        return Ok(());
    }

    let exec = device.stream();
    let _ga = crate::kernel::order_input(a, exec)?;

    let entry = format!("rstsr_{}", op.name());
    let body = render_scan(op, &entry, T::CNAME);
    let src = crate::kernel::assemble_source(T::INCLUDES, &body);
    // `make_key`'s name is the **symbol** looked up in the module, not a cache label: passing the
    // bare `cumsum` here compiled the kernel and then failed to find it
    // (`CUDA_ERROR_NOT_FOUND, "named symbol not found"`), because the entry is `rstsr_cumsum`.
    let key = crate::kernel::make_key(device.ordinal_obj(), &entry, &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;

    let lo_f = FlatLayout::of(lo_rest);
    let li_f = FlatLayout::of(lm);
    let pack = LayoutPack::new(exec, &[(&lo_f, true), (&li_f, false)])?;

    // SAFETY: `MaybeUninit<T>` and `T` are layout-identical; every element of the output is written
    // exactly once (each line writes its `size_m` positions, plus the initial one when asked).
    let c_typed: &mut CudaBuffer<T> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<T>> as *mut CudaBuffer<T>) };

    let block = super::reduce_plan::BLOCK;
    let grid = n_lines;
    if grid > u32::MAX as usize {
        rstsr_raise!(UnImplemented, "cuda {}: grid exceeds u32", op.name())?;
    }

    crate::kernel::FlatLaunch::new(exec, &f, grid)
        .arg_buf(c_typed)
        .arg_buf(a)
        .arg_i32(lo_rest.ndim() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo_f.offset)
        .arg_u64(pack.strides[1])
        .arg_usize(li_f.offset)
        .arg_usize(size_m)
        .arg_bytes(&step_in.to_ne_bytes())
        .arg_bytes(&step_out.to_ne_bytes())
        .arg_i32(include_initial as i32)
        .explicit_shape((grid as u32, 1, 1), (block as u32, 1, 1))
        .launch()?;

    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identities are what the tail lanes and the `include_initial` slot carry, so a wrong one
    /// is a wrong answer everywhere rather than in an edge case.
    #[test]
    fn identities_are_neutral() {
        assert_eq!(ScanOp::Sum.init(), "(@TO@)0");
        assert_eq!(ScanOp::Prod.init(), "(@TO@)1");
        assert!(ScanOp::Sum.op().contains("(x)") && ScanOp::Sum.op().contains("(y)"));
        assert!(ScanOp::Prod.op().contains('*'));
    }

    /// Render with real substitutions and read it — the cheap substitute for a GPU round trip.
    #[test]
    fn rendered_source_is_complete() {
        for op in [ScanOp::Sum, ScanOp::Prod] {
            let src = render_scan(op, &format!("rstsr_{}", op.name()), "double");
            assert!(!src.contains('@'), "{}: unsubstituted token", op.name());
            // Three uses of the operator macro: the warp scan, the carry fold, and the per-warp
            // addition. A missing one is a scan that silently drops a chunk.
            assert_eq!(src.matches("RSTSR_SCAN_OP(").count(), 4, "{}: scan op sites", op.name());
            // The tail lanes must not return early: that is what makes the shuffle legal.
            assert!(!src.contains("if (idx >= size_m) return"), "{}: early return", op.name());
            assert_eq!(src.matches("__syncthreads()").count(), 4, "{}: barriers", op.name());
        }
    }
}
