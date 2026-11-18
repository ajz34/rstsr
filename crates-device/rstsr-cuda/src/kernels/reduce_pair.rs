//! Two-slot unary reductions: `var`/`std` here, the arg family next.
//!
//! # Why a second entry point rather than a second accumulator in [`super::reduce_unary`]
//!
//! Those nine rows fold into **one** value, and their generated source is the thing that currently
//! passes on hardware. Threading an optional second register through that template — a declaration,
//! a fold statement, a shuffle, a shared slot, and a different writer, each expanding to nothing for
//! ten of the twelve rows — would make the reviewed source of the working rows depend on a decision
//! that only matters for the other two. The plan is shared verbatim; the body is not; so this is a
//! second entry point, exactly as `reduce_unary` itself is a second entry point beside `reduce`.
//!
//! # What the pair actually is
//!
//! `var` needs `(Σx, Σ|x|²)` in one pass, because the CPU computes `Σ|x|²/n − |mean|²` in one pass
//! and a second pass would double the bytes read for an op that is memory-bound. The two slots are
//! **independent**: both fold with `+`, both combine with `+`, and the only interaction is at the
//! writer, where the mean has to be formed before squaring it. That is what makes this cheap — no
//! lexicographic combine, no joint decision inside the tree.
//!
//! ```text
//! op    INIT1  INIT2  FOLD1(A,X)  FOLD2(A2,X)   COMB1   COMB2   POST(V, V2)
//! var   0      0      A + X       A2 + X*X      A + B   A + B   V2/n - (V/n)^2
//! std   0      0      A + X       A2 + X*X      A + B   A + B   sqrt(V2/n - (V/n)^2)
//! ```
//!
//! `n` is `n_contract`, the same divisor the CPU uses (`layout_axes.size()`).
//!
//! # Real dtypes only, for the same reason as `l2_norm`
//!
//! The CPU accumulates `(T, T::Real)`: a complex `Σx` beside a real `Σ|x|²`. Here both slots are the
//! same real type, which is exact for real operands and is why `TOut = T` (`Real = Self`). A complex
//! operand needs a two-type accumulator — one struct, or a third template — and the call site refuses
//! it because `Complex` is not `num::Float`.
//!
//! The empty-input fill is `NaN`, matching the CPU: `0/0 − 0` is `NaN`, so `var([])` is `NaN` rather
//! than an uninitialised read. Same convention as `mean`.
//!
//! # The tree
//!
//! Identical shape to [`super::reduce_unary`]: warp butterfly, then a shared stage only when
//! `LANES > warpSize`. Both slots take part in both stages. The guard rule from there applies here
//! too — threads past `n_out` must not `return`, and lanes past `n_contract` keep the identity (zero
//! for both slots of an addition), which is what makes the tail block harmless.

use std::sync::Arc;

use cudarc::driver::CudaFunction;

use super::reduce_plan::{ReducePlan, BLOCK};
use super::reduce_unary::{offset_ptr, reduce_unary_function, AccSpec};
use crate::dtype::CudaDType;
use crate::prelude_dev::*;

/// Largest number of warps a block can contain, assuming a warp is at least 32 lanes.
const MAX_WARPS_PER_BLOCK: usize = BLOCK / 32;

/// The reduction to perform. One variant per row of the table in the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PairOp {
    Var,
    Std,
}

impl PairOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Var => "var",
            Self::Std => "std",
        }
    }

    /// Both slots start at zero: the identity of the two additions.
    fn init(self) -> &'static str {
        "(@TACC@)0"
    }

    /// `A = FOLD1(A, X)`. `X` is already the accumulator type — the loader casts once.
    fn fold(self) -> &'static str {
        match self {
            Self::Var | Self::Std => "((A) + (X))",
        }
    }

    /// `A = FOLD2(A, X)`, where `A` is the *second* accumulator. `X*X`, not `X*conj(X)`: the
    /// operand is real by the impl bound.
    fn fold2(self) -> &'static str {
        match self {
            Self::Var | Self::Std => "((A) + (X) * (X))",
        }
    }

    fn comb(self) -> &'static str {
        match self {
            Self::Var | Self::Std => "((A) + (B))",
        }
    }

    /// The second slot's combine. **Delegates** to [`Self::comb`] because the separable rows use one
    /// operation for both slots; a *joint* combine (the arg family) cannot share this and needs its
    /// own tree.
    fn comb2(self) -> &'static str {
        self.comb()
    }

    /// `@TOUT@` value from the finished pair. Applied once, at the writer.
    ///
    /// The mean appears twice because it is written as an expression; it is the same expression both
    /// times, so the result is the CPU's `let mean = …; acc_2/size − mean*mean` exactly.
    fn post(self) -> &'static str {
        match self {
            Self::Var => {
                "((V2) / (@TACC@)n_contract \
                  - ((V) / (@TACC@)n_contract) * ((V) / (@TACC@)n_contract))"
            },
            Self::Std => {
                "sqrt((V2) / (@TACC@)n_contract \
                  - ((V) / (@TACC@)n_contract) * ((V) / (@TACC@)n_contract))"
            },
        }
    }
}

const REDUCE_PAIR_ENTRY: &str = r#"
#define RSTSR_INIT1       (@INIT@)
#define RSTSR_INIT2       (@INIT2@)
#define RSTSR_FOLD1(A, X) (@FOLD@)
#define RSTSR_FOLD2(A, X) (@FOLD2@)
#define RSTSR_COMB1(A, B) (@COMB@)
#define RSTSR_COMB2(A, B) (@COMB2@)
#define RSTSR_POST(V, V2) (@POST@)

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

    // Out-of-range threads and lanes past `n_contract` keep the identity in *both* slots and still
    // reach the shuffles below.
    @TACC@ acc = RSTSR_INIT1;
    @TACC@ acc2 = RSTSR_INIT2;
    if (j < n_out) {
        const @TIN@* ap = a + (rstsr_isz)j * out_step;
        for (rstsr_usz m = lane; m < n_contract; m += @LANES@) {
            const @TACC@ x_ = (@TACC@)ap[(rstsr_isz)m * step];
            acc = RSTSR_FOLD1(acc, x_);
            acc2 = RSTSR_FOLD2(acc2, x_);
        }
    }

    // Stage 1: within a warp. `__syncwarp()` forces reconvergence after the divergent guard, which
    // is what makes `__activemask()` meaningful here.
    __syncwarp();
    const int ws = warpSize;
    for (int off = (@LANES@ <= 32 ? (@LANES@ >> 1) : (ws >> 1)); off > 0; off >>= 1) {
        @TACC@ other = __shfl_down_sync(__activemask(), acc, off);
        acc = (((rstsr_usz)off + lane < @LANES@) ? RSTSR_COMB1(acc, other) : acc);
        @TACC@ other2 = __shfl_down_sync(__activemask(), acc2, off);
        acc2 = (((rstsr_usz)off + lane < @LANES@) ? RSTSR_COMB2(acc2, other2) : acc2);
    }

    // Stage 2: only when one output spans several warps. Statically dead for LANES <= 32.
    const int nw = @LANES@ / ws;
    if (@LANES@ > 32 && nw > 1) {
        __shared__ @TACC@ wpart[@MAXW@];
        __shared__ @TACC@ wpart2[@MAXW@];
        const int tl = (int)((unsigned)threadIdx.x & (unsigned)(ws - 1));
        const int wid = (int)((unsigned)threadIdx.x / (unsigned)ws);
        if (tl == 0) {
            wpart[wid] = acc;
            wpart2[wid] = acc2;
        }
        __syncthreads();

        if (wid % nw == 0) {
            @TACC@ v = (tl < nw) ? wpart[(wid / nw) * nw + tl] : RSTSR_INIT1;
            @TACC@ v2 = (tl < nw) ? wpart2[(wid / nw) * nw + tl] : RSTSR_INIT2;
            for (int off = nw >> 1; off > 0; off >>= 1) {
                @TACC@ o = __shfl_down_sync(__activemask(), v, off);
                @TACC@ o2 = __shfl_down_sync(__activemask(), v2, off);
                v = (((int)tl + off < nw) ? RSTSR_COMB1(v, o) : v);
                v2 = (((int)tl + off < nw) ? RSTSR_COMB2(v2, o2) : v2);
            }
            if (tl == 0 && j < n_out) c[j] = (@TOUT@)RSTSR_POST(v, v2);
        }
        return;
    }

    if (lane == 0 && j < n_out) c[j] = (@TOUT@)RSTSR_POST(acc, acc2);
}
"#;

fn render(template: &str, subs: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (name, text) in subs {
        s = s.replace(name, text);
    }
    s
}

/// Launch a two-slot reduction for a plan produced by [`super::reduce_plan::classify_reduce`].
///
/// Mirrors [`super::reduce_unary::reduce_unary_launch`] argument for argument: the plan is the same
/// one, and both slots carry the accumulator's type, so `acc` supplies the single C name.
pub(crate) fn reduce_pair_launch<TIn, TOut>(
    device: &DeviceCuda,
    op: PairOp,
    acc: AccSpec,
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
    let ReducePlan {
        lanes,
        block,
        n_contract,
        step_a,
        n_out,
        out_step_a,
        ..
    } = *plan;

    if n_out == 0 || n_contract == 0 {
        return Ok(());
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

    let name = format!("rstsr_reduce_pair_{}", op.name());
    let disc = format!("{name}|{}|{}|{}|L{lanes}|B{block}", TIn::CNAME, acc.cname, TOut::CNAME);

    // A mixed-dtype instantiation may need both types' includes; the same-dtype case is a no-op.
    let mut includes: Vec<&'static str> = TIn::INCLUDES.to_vec();
    for inc in TOut::INCLUDES.iter().copied() {
        if !includes.contains(&inc) {
            includes.push(inc);
        }
    }

    let init = op.init();
    let opb = plan.outputs_per_block.to_string();
    let lanes_s = lanes.to_string();
    let log2_s = lanes.trailing_zeros().to_string();
    let maxw_s = MAX_WARPS_PER_BLOCK.to_string();

    let f: Arc<CudaFunction> = reduce_unary_function(device, &name, &disc, &includes, || {
        render(
            REDUCE_PAIR_ENTRY,
            &[
                // Producers before consumers: `FOLD`/`COMB`/`POST` contain `@TACC@`, and both are
                // *inside* the macro definitions above the body.
                ("@NAME@", name.as_str()),
                ("@FOLD@", op.fold()),
                ("@FOLD2@", op.fold2()),
                ("@COMB@", op.comb()),
                ("@COMB2@", op.comb2()),
                ("@POST@", op.post()),
                ("@INIT@", init),
                ("@INIT2@", init),
                ("@TIN@", TIn::CNAME),
                ("@TACC@", acc.cname),
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

    /// Every row, so a new op cannot be added without the checks below seeing it.
    const ALL: [PairOp; 2] = [PairOp::Var, PairOp::Std];

    /// The two slots must fold and combine with the *same* operation, or the tree is not a valid
    /// reordering of the sequential fold.
    #[test]
    fn both_slots_use_one_operation() {
        for op in ALL {
            assert!(op.fold().contains("+"), "{}: fold must add", op.name());
            assert!(op.comb().contains("+"), "{}: comb must add", op.name());
            assert!(op.fold2().contains('*'), "{}: slot 2 must square", op.name());
            assert!(op.comb2().contains("+"), "{}: slot 2 comb must add", op.name());
        }
    }

    /// The bodies of `FOLD2`/`COMB2` are pasted into `#define RSTSR_FOLD2(A, X)` / `(A, B)`, so they
    /// may name `A`, `B`, `X` and nothing else.
    ///
    /// `A2` reads like "the second accumulator" and is precisely what NVRTC rejected on the first
    /// hardware run (`identifier "A2" is undefined`, three sites). A local check, because the
    /// alternative is discovering it one round trip at a time.
    #[test]
    fn second_slot_bodies_name_only_the_macro_parameters() {
        for op in ALL {
            for (what, body) in [
                ("fold", op.fold()),
                ("fold2", op.fold2()),
                ("comb", op.comb()),
                ("comb2", op.comb2()),
            ] {
                for bad in ["A2", "B2", "X2"] {
                    assert!(
                        !body.contains(bad),
                        "{}: {what} names `{bad}`, which is not a macro parameter: {body}",
                        op.name()
                    );
                }
            }
        }
    }

    /// `std` is `sqrt` applied to `var`'s expression and nothing else, so the two cannot drift.
    ///
    /// No extra pair of parentheses around the argument: `Var::post` already *is* a parenthesised
    /// expression, which is why this is `sqrt{}` and not `sqrt({})`.
    #[test]
    fn std_is_the_root_of_var() {
        assert_eq!(PairOp::Std.post(), format!("sqrt{}", PairOp::Var.post()));
    }

    /// The mean must be formed *from the first slot* and squared; a stray `V2/n - V*V/n/n` would
    /// still compile and would be a different (wrong) statistic.
    #[test]
    fn post_subtracts_the_squared_mean() {
        let p = PairOp::Var.post();
        assert!(p.contains("(V2) / (@TACC@)n_contract"), "sum of squares over n: {p}");
        assert!(p.contains("((V) / (@TACC@)n_contract) * ((V) / (@TACC@)n_contract)"), "mean squared: {p}");
    }

    /// Every token the template uses must be substituted, or the source fails inside NVRTC on a
    /// rented GPU instead of here.
    #[test]
    fn every_token_is_substituted() {
        let acc = AccSpec { cname: "double", pos_inf: "", neg_inf: "" };
        let src = render(
            REDUCE_PAIR_ENTRY,
            &[
                ("@NAME@", "k"),
                ("@FOLD@", PairOp::Var.fold()),
                ("@FOLD2@", PairOp::Var.fold2()),
                ("@COMB@", PairOp::Var.comb()),
                ("@COMB2@", PairOp::Var.comb2()),
                ("@POST@", PairOp::Var.post()),
                ("@INIT@", PairOp::Var.init()),
                ("@INIT2@", PairOp::Var.init()),
                ("@TIN@", "double"),
                ("@TACC@", acc.cname),
                ("@TOUT@", "double"),
                ("@LANES@", "32"),
                ("@LOG2_LANES@", "5"),
                ("@OPB@", "8"),
                ("@MAXW@", "8"),
            ],
        );
        assert!(!src.contains('@'), "unsubstituted token in:\n{src}");
        let _ = src.len();
    }
}
