//! argmin/argmax: the reduction that cannot be split into independent slots.
//!
//! # The shape of the problem
//!
//! [`super::reduce_pair`] reduces two independent quantities in one tree: `Σx` and `Σx²` never
//! consult each other, and only the writer combines them. An arg-reduction is the opposite — the
//! accumulator is a *pair that moves together*. When a new element wins, its index must come with
//! it, so the combine reads both slots and writes both. That is why this is a separate kernel and
//! not a third row in the pair table.
//!
//! Concretely the accumulator is `(value, index)`; it is packed into a struct and shuffled as two
//! registers, because `__shfl_down_sync` has no struct overload.
//!
//! # The tie-break is pinned by the CPU, not invented here
//!
//! The CPU keeps the **smaller logical index**
//! (`rstsr-native-impl/src/cpu_serial/reduction.rs`, `reduce_all_unraveled_arg_fold_cpu_serial`,
//! "cond 2: current value is same with previous value, return smaller index"), where "logical" means
//! the C-order raveled index over the reduced axes — independent of memory layout. Sequential
//! left-to-right traversal over C-order can never see a smaller index after a larger one, so on the
//! CPU the rule reduces to *first occurrence wins*.
//!
//! Here it is written out as a lexicographic comparison instead, because a tree visits its blocks
//! in an order the fold does not see: `(v, i)` replaces `(v', i')` iff the value is strictly better,
//! or equal and `i < i'`. Under the flat-index enumeration the two rules agree, and the comparison
//! form stays correct if the tree ever changes shape.
//!
//! # NaN, exactly as the CPU does it
//!
//! The CPU's four policies are asymmetric on purpose, and the kernel copies them rather than
//! "doing the sane thing":
//!
//! - **plain `argmin`/`argmax`**: the comparison is `y > x` and the tie test `y == x`. Both are
//!   false when either operand is NaN, so a NaN that arrives first **wins forever**
//!   (`argmax([NaN, 5]) == 0`), and a NaN that arrives later is ignored. No special case is written
//!   here: the two false answers fall out of the lexicographic test.
//! - **`nanargmin`/`nanargmax`**: a NaN element is skipped entirely *before* it can seed the
//!   accumulator (`f_comp_nan_max` returns `None`, meaning "leave the accumulator alone"). So an
//!   all-NaN contraction has **no** answer, which is why the accumulator needs an explicit empty
//!   state rather than an identity value.
//!
//! # The empty state, and why the identity trick does not work here
//!
//! The unary kernel's tail lanes keep the operation's identity and take part in the tree, which is
//! free because `0` cannot change a sum. There is no index that cannot change an argmax: index `0`
//! is a real position. So the accumulator carries `RSTSR_ARG_NONE` — an index that is out of range
//! by construction — and the combine treats it as "no element yet". Both the per-lane fold and the
//! tree use the *same* combine, so the rule is written once.
//!
//! A contraction that stays empty to the very end (only possible for the NaN policies) is written
//! as `n_contract`, an out-of-range index, and the caller turns that into an error — see
//! [`crate::arg_ops`].
//!
//! # Enumeration order is the logical index, and that is checked before launching
//!
//! The kernel walks the contracted part as `offset = base + m * step`, so `m` *is* the raveled
//! logical index only when the contracted part is contiguous in C-order with the axes in their
//! logical order. [`contracted_is_logical`] is that check, and the launcher refuses rather than
//! returning indices that are raveled over the wrong order. The plan's own `step` is compared
//! against the layout's innermost stride as a second, independent guard: two derivations of the
//! same number, so a mismatch is a bug in one of them and not a silently wrong index.

use crate::dtype::CudaDType;
use crate::prelude_dev::*;
use super::reduce_plan::ReducePlan;
use super::reduce_unary::{
    offset_ptr, reduce_unary_function, render as render_for, MAX_WARPS_PER_BLOCK as MAX_WARPS,
};

/// One of the four arg policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArgOp {
    Max,
    Min,
    NanMax,
    NanMin,
}

impl ArgOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Max => "argmax",
            Self::Min => "argmin",
            Self::NanMax => "nanargmax",
            Self::NanMin => "nanargmin",
        }
    }

    /// The strict comparison, mirrored per policy: `max`/`nanmax` accept a *larger* value.
    fn cmp(self) -> &'static str {
        match self {
            Self::Max | Self::NanMax => ">",
            Self::Min | Self::NanMin => "<",
        }
    }

    /// `true` for the policies that drop NaN elements before they can seed the accumulator.
    pub(crate) fn skips_nan(self) -> bool {
        matches!(self, Self::NanMax | Self::NanMin)
    }

    /// The NaN guard, or a branch that never runs.
    ///
    /// A plain policy must **not** skip NaN — a leading NaN has to win — so the guard is emitted as
    /// `if (0)`, which the compiler removes. Writing the guard unconditionally and passing a flag
    /// would work too, but then the plain rows would carry a comparison against a NaN with different
    /// semantics (`b.v != b.v` is *true* for NaN, so it would have to be inverted), and the two
    /// policies would differ by a negation rather than by a branch.
    fn nan_guard(self) -> &'static str {
        match self {
            Self::NanMax | Self::NanMin => "if (b.v != b.v) { return a; }",
            Self::Max | Self::Min => "if (0) { return a; }",
        }
    }
}

/// The kernel. `@NAME@` per policy so a profiler shows which one is running.
///
/// The accumulator is a struct of `(value, index)`. The index is `rstsr_usz` — wide enough for any
/// contraction — and travels through the tree beside the value in every stage, which is the whole
/// difference from the pair kernel.
const REDUCE_ARG_ENTRY: &str = r#"
struct rstsr_argacc {
    @TIN@ v;
    rstsr_usz i;
};

#define RSTSR_ARG_NONE ((rstsr_usz)(~(rstsr_usz)0))

// The one rule. Both the per-lane fold and the tree call this, so "better" cannot mean two
// different things in the two places.
__device__ __forceinline__ rstsr_argacc rstsr_arg_comb(rstsr_argacc a, rstsr_argacc b) {
    @NAN_GUARD@
    if (a.i == RSTSR_ARG_NONE) { return b; }
    if (b.i == RSTSR_ARG_NONE) { return a; }
    if ((b.v @CMP@ a.v) || (b.v == a.v && b.i < a.i)) { return b; }
    return a;
}

extern "C" __global__ void @NAME@(
    @TOUT@* __restrict__ c,
    const @TIN@* __restrict__ a,
    rstsr_usz n_out,
    rstsr_usz n_contract,
    rstsr_isz step,      // element step per contracted index
    rstsr_isz out_step)  // element step per output index (unused when n_out == 1)
{
    const rstsr_usz lane = (rstsr_usz)threadIdx.x & (@LANES@ - 1u);
    const rstsr_usz j = (rstsr_usz)blockIdx.x * @OPB@ + ((rstsr_usz)threadIdx.x >> @LOG2_LANES@);

    // Out-of-range threads keep an empty accumulator and still reach the shuffles: the combine has
    // no identity, so "empty" has to be an explicit state rather than a value.
    rstsr_argacc acc;
    acc.v = (@TIN@)0;
    acc.i = RSTSR_ARG_NONE;

    if (j < n_out) {
        const @TIN@* ap = a + (rstsr_isz)j * out_step;
        for (rstsr_usz m = lane; m < n_contract; m += @LANES@) {
            rstsr_argacc t;
            t.v = ap[(rstsr_isz)m * step];
            t.i = m;                       // the logical index: enumeration order is C-order
            acc = rstsr_arg_comb(acc, t);
        }
    }

    __syncwarp();
    const int ws = warpSize;
    for (int off = (@LANES@ <= 32 ? (@LANES@ >> 1) : (ws >> 1)); off > 0; off >>= 1) {
        rstsr_argacc other;
        other.v = __shfl_down_sync(__activemask(), acc.v, off);
        other.i = __shfl_down_sync(__activemask(), acc.i, off);
        acc = ((((rstsr_usz)off + lane) < @LANES@) ? rstsr_arg_comb(acc, other) : acc);
    }

    const int nw = @LANES@ / ws;
    if (@LANES@ > 32 && nw > 1) {
        __shared__ rstsr_argacc wpart[@MAXW@];
        const int tl = (int)((unsigned)threadIdx.x & (unsigned)(ws - 1));
        const int wid = (int)((unsigned)threadIdx.x / (unsigned)ws);
        if (tl == 0) wpart[wid] = acc;
        __syncthreads();

        if (wid % nw == 0) {
            rstsr_argacc v;
            if (tl < nw) {
                v = wpart[(wid / nw) * nw + tl];
            } else {
                v.v = (@TIN@)0;
                v.i = RSTSR_ARG_NONE;
            }
            for (int off = nw >> 1; off > 0; off >>= 1) {
                rstsr_argacc o;
                o.v = __shfl_down_sync(__activemask(), v.v, off);
                o.i = __shfl_down_sync(__activemask(), v.i, off);
                v = ((((int)tl + off) < nw) ? rstsr_arg_comb(v, o) : v);
            }
            // An empty result is written out of range; the caller raises on it.
            if (tl == 0 && j < n_out) c[j] = (@TOUT@)(v.i == RSTSR_ARG_NONE ? n_contract : v.i);
        }
        return;
    }

    if (lane == 0 && j < n_out) c[j] = (@TOUT@)(acc.i == RSTSR_ARG_NONE ? n_contract : acc.i);
}
"#;

/// Is the contracted part contiguous in **logical** C-order, and therefore enumerable by one
/// arithmetic step?
///
/// The condition is `stride[i] == stride[j] * shape[j]` for every adjacent pair of axes that are
/// not of extent 1 (a size-1 axis imposes no order, so its stride is arbitrary). That is exactly
/// "the C-order raveled index `m` maps to `base + m * stride[last]`", which is what makes `m` the
/// number the CPU would return.
///
/// Returns the step when it holds, so the launcher can compare it with the plan's.
pub(crate) fn contracted_is_logical(las: &Layout<IxD>) -> Option<isize> {
    let (shape, stride) = (las.shape(), las.stride());
    let live: Vec<usize> = (0..shape.len()).filter(|k| shape[*k] > 1).collect();
    for w in live.windows(2) {
        let (outer, inner) = (w[0], w[1]);
        if stride[outer] != stride[inner] * shape[inner] as isize {
            return None;
        }
    }
    live.last().map(|k| stride[*k])
}

/// Launch one arg policy for a plan from [`super::reduce_plan::classify_reduce`].
pub(crate) fn reduce_arg_launch<TIn, TOut>(
    device: &DeviceCuda,
    op: ArgOp,
    c: &mut CudaBuffer<MaybeUninit<TOut>>,
    lc: &Layout<IxD>,
    a: &CudaBuffer<TIn>,
    las: &Layout<IxD>,
    plan: &ReducePlan,
) -> Result<()>
where
    TIn: CudaDType,
    TOut: CudaDType,
{
    let ReducePlan { lanes, block, n_contract, step_a, n_out, out_step_a, .. } = *plan;

    if n_out == 0 || n_contract == 0 {
        return Ok(());
    }

    // The index the kernel writes is `m`, so `m` has to be the logical index. Checked here rather
    // than assumed: a layout that fails this is one whose reduced axes are not in C-order, and the
    // answer would be a correct value with a wrong index — the one failure mode that would look
    // plausible.
    //
    // A contraction of one element has no order to check (`contracted_is_logical` returns `None`
    // when there is nothing to order), and its index is `0` regardless of how the strides fall.
    let logical_step = if n_contract <= 1 {
        step_a
    } else {
        match contracted_is_logical(las) {
        Some(s) => s,
        None => rstsr_raise!(
            UnImplemented,
            "cuda {}: the contracted axes are not contiguous in C-order (shape {:?}, stride {:?}), \
             so the element enumeration order is not the order the returned index is raveled in; \
             call `to_contig` on the operand first",
            op.name(),
            las.shape(),
            las.stride()
        )?,
        }
    };
    if logical_step != step_a {
        rstsr_raise!(
            InvalidLayout,
            "cuda {}: plan step {} disagrees with the layout's innermost stride {}; the index would \
             be raveled over the wrong order",
            op.name(),
            step_a,
            logical_step
        )?;
    }

    let grid_blocks = plan.grid_blocks();
    if grid_blocks > u32::MAX as usize {
        rstsr_raise!(UnImplemented, "cuda {}: grid exceeds u32", op.name())?;
    }
    if !lanes.is_power_of_two() || block % lanes != 0 {
        rstsr_raise!(InvalidLayout, "cuda {}: plan has lanes={lanes}, block={block}", op.name())?;
    }

    let exec = device.stream();
    let _ga = crate::kernel::order_input(a, exec)?;

    let name = format!("rstsr_{}", op.name());
    let disc = format!("{name}|{}|{}|L{lanes}|B{block}", TIn::CNAME, TOut::CNAME);

    let mut includes: Vec<&'static str> = TIn::INCLUDES.to_vec();
    for inc in TOut::INCLUDES.iter().copied() {
        if !includes.contains(&inc) {
            includes.push(inc);
        }
    }

    let opb = plan.outputs_per_block.to_string();
    let lanes_s = lanes.to_string();
    let log2_s = lanes.trailing_zeros().to_string();
    let maxw_s = MAX_WARPS.to_string();

    let f = reduce_unary_function(device, &name, &disc, &includes, || {
        render_for(
            REDUCE_ARG_ENTRY,
            &[
                ("@NAN_GUARD@", op.nan_guard()),
                ("@NAME@", name.as_str()),
                ("@CMP@", op.cmp()),
                ("@TIN@", TIn::CNAME),
                ("@TOUT@", TOut::CNAME),
                ("@LANES@", lanes_s.as_str()),
                ("@LOG2_LANES@", log2_s.as_str()),
                ("@OPB@", opb.as_str()),
                ("@MAXW@", maxw_s.as_str()),
            ],
        )
    })?;

    // SAFETY: `MaybeUninit<TOut>` and `TOut` are layout-identical; the kernel writes every visited
    // element (exactly once per `j < n_out`) and `c` holds `n_out`.
    let c_typed: &mut CudaBuffer<TOut> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOut>> as *mut CudaBuffer<TOut>) };

    crate::kernel::FlatLaunch::new(exec, &f, grid_blocks)
        .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
        .arg_u64(offset_ptr(exec, a, las.offset()))
        .arg_usize(n_out)
        .arg_usize(n_contract)
        .arg_bytes(&step_a.to_ne_bytes())
        .arg_bytes(&out_step_a.to_ne_bytes())
        .explicit_shape((grid_blocks as u32, 1, 1), (block, 1, 1))
        .launch()?;

    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], stride: &[isize]) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), 0).unwrap()
    }

    /// The check must accept every C-contiguous arrangement and reject the transposed ones.
    #[test]
    fn logical_order_is_the_c_order() {
        // (2,3,4) contiguous, whole block
        assert_eq!(contracted_is_logical(&layout(&[2, 3, 4], &[12, 4, 1])), Some(1));
        // a stepped block: still one arithmetic progression
        assert_eq!(contracted_is_logical(&layout(&[2, 3], &[6, 2])), Some(2));
        // size-1 axes impose no order: (1,4,1) over strides (99, 5, 3)
        assert_eq!(contracted_is_logical(&layout(&[1, 4, 1], &[99, 5, 3])), Some(5));
        // rank-0 and single-element blocks have no order to check; the launcher skips the check
        // when the contraction holds one element, because its index is 0 however the strides fall.
        assert_eq!(contracted_is_logical(&layout(&[], &[])), None);
        assert_eq!(contracted_is_logical(&layout(&[1], &[7])), None);
        // transposed: axis 0 is the fast one, so `m` would count in memory order, not logical order
        assert_eq!(contracted_is_logical(&layout(&[3, 2], &[1, 3])), None);
        // a gap between the contracted axes (the tensordot refusal case)
        assert_eq!(contracted_is_logical(&layout(&[2, 4], &[12, 1])), None);
    }

    /// Each policy's comparison and guard must be the pair that matches the CPU, not a plausible
    /// pair: `nanargmax` compares with `>` and skips NaN, `argmax` compares with `>` and does not.
    #[test]
    fn policies_match_the_cpu() {
        assert_eq!(ArgOp::Max.cmp(), ">");
        assert_eq!(ArgOp::Min.cmp(), "<");
        assert_eq!(ArgOp::NanMax.cmp(), ">");
        assert_eq!(ArgOp::NanMin.cmp(), "<");
        assert!(!ArgOp::Max.skips_nan() && !ArgOp::Min.skips_nan());
        assert!(ArgOp::NanMax.skips_nan() && ArgOp::NanMin.skips_nan());
        // A plain policy's guard must not mention NaN at all: skipping it would make
        // `argmax([NaN, 5])` return 1 where the CPU returns 0.
        assert!(!ArgOp::Max.nan_guard().contains("b.v"), "{}", ArgOp::Max.nan_guard());
        assert!(ArgOp::NanMax.nan_guard().contains("b.v != b.v"));
    }

    /// Every token in the template must be substituted — an unsubstituted one reaches NVRTC and
    /// fails at launch, on hardware, not here.
    #[test]
    fn every_token_is_substituted() {
        let subs = ["@NAN_GUARD@", "@NAME@", "@CMP@", "@TIN@", "@TOUT@", "@LANES@", "@LOG2_LANES@", "@OPB@", "@MAXW@"];
        let bytes = REDUCE_ARG_ENTRY.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'@' {
                if let Some(j) = REDUCE_ARG_ENTRY[i + 1..].find('@') {
                    let tok = &REDUCE_ARG_ENTRY[i..i + j + 2];
                    assert!(subs.contains(&tok), "template uses {tok}, which is not substituted");
                    i += j + 2;
                    continue;
                }
            }
            i += 1;
        }
    }

    /// Render the source the way the launcher does, for every policy, and check the result is a
    /// complete program rather than a template with holes in it.
    ///
    /// This is the local half of what NVRTC does at launch: the previous kernel in this crate
    /// (`reduce_pair`) was rejected on its first hardware run because a macro body named a variable
    /// the macro did not have, which no amount of token counting would have caught. Rendering and
    /// asserting on the *text* is the cheap version of that round trip.
    #[test]
    fn rendered_source_is_complete() {
        for (op, cmp, guard) in [
            (ArgOp::Max, ">", "if (0) { return a; }"),
            (ArgOp::Min, "<", "if (0) { return a; }"),
            (ArgOp::NanMax, ">", "if (b.v != b.v) { return a; }"),
            (ArgOp::NanMin, "<", "if (b.v != b.v) { return a; }"),
        ] {
            let src = render_for(
                REDUCE_ARG_ENTRY,
                &[
                    ("@NAN_GUARD@", op.nan_guard()),
                    ("@NAME@", &format!("rstsr_{}", op.name())),
                    ("@CMP@", op.cmp()),
                    ("@TIN@", "double"),
                    ("@TOUT@", "unsigned long long"),
                    ("@LANES@", "256"),
                    ("@LOG2_LANES@", "8"),
                    ("@OPB@", "1"),
                    ("@MAXW@", "8"),
                ],
            );
            assert!(!src.contains('@'), "{}: unsubstituted token left in the source", op.name());
            assert_eq!(src.matches(guard).count(), 1, "{}: guard", op.name());
            assert_eq!(op.cmp(), cmp);
            // One definition and three uses: the per-lane fold and the two butterfly stages. A
            // fourth use (or a missing one) means a stage was added or lost, which is exactly the
            // kind of edit that leaves a slot uncombined.
            assert_eq!(src.matches("rstsr_arg_comb(").count(), 4, "{}: combine sites", op.name());
            // The empty state must be defined before its first use.
            let def = src.find("#define RSTSR_ARG_NONE").unwrap();
            let first_use = src.find("acc.i = RSTSR_ARG_NONE").unwrap();
            assert!(def < first_use, "{}: RSTSR_ARG_NONE used before it is defined", op.name());
        }
    }
}
