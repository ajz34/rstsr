//! The tiled SIMT GEMM driver — the kernel behind `matmul` when cuBLAS declines.
//!
//! It takes what the classic cuBLAS entries cannot express (a stepped output, a reversed
//! operand) and a machine without `libcublas`, and it is the on-device step before the naive
//! kernel. It is kept for those cases and as a test oracle, not because it competes — on an A100,
//! `dgemm` reaches the FP64 tensor-core pipe at 19.5 TFLOPS where a SIMT tile tops out at 9.7.
//!
//! # What this kernel is
//!
//! One block computes a `BM x BN` tile of `C = alpha * A B + beta * C` by streaming the contraction
//! axis through shared memory in `BK`-deep slabs. Per block that is `BM*BN*K` fused multiply-adds
//! for `(BM+BN)*K` element loads, i.e. `2*64*64 / (128*8) = 8` flop/byte once the tile is resident.
//! The untiled form is `2/16 = 0.125`, which is DRAM-bound on every machine — the whole reason this
//! file exists.
//!
//! 8 flop/byte is only just above the A100 fp64 ridge point (`9.7 TFLOPS / 1.55 TB/s` = 6.3), so at
//! this tile size the kernel is *marginally* compute-bound and the memory system still costs nearly
//! as much as the arithmetic. Doubling the tile to 128x128 gives `2*128*128 / (256*8) = 16` and is
//! the first thing to try after measuring this one; `BM`/`BN` are plain constants and every derived
//! count (slab depth, thread count, shared sizes) follows from them, so the change is a re-tune
//! rather than a rewrite.
//!
//! # The two things that are actually decisions
//!
//! **1. The lane map follows a unit-stride axis, not a fixed axis.** A 2-D operand arrives with
//! arbitrary strides: `A` may be `[M, K]` with `K` contiguous (`rt::matmul(a, b)`) or with `M`
//! contiguous (`rt::matmul(a.T, b)`), and `layout_matmul` admits both because it only checks
//! shapes. Instead of transposing an operand — a copy — the block's linear thread id is mapped to
//! the tile along whichever axis is unit-stride, chosen at **NVRTC time**. Four variants result (A
//! on `k` or `m`, B on `k` or `n`) and each has both global loads coalesced.
//!
//! **2. Per-thread tiles are interleaved, not blocked.** Thread `(tx, ty)` owns rows `ty + i*TY`
//! and columns `tx + j*TX`, so one shared-memory read instruction has its 16 lanes on 16
//! consecutive doubles (and the other row group in the warp reads the same address, which
//! broadcasts). Blocking would instead make the `Bs` read stride `TN` doubles per lane — a 4-way
//! bank conflict on every one of the `TM*TN*BK` inner reads. Interleaving costs something on the
//! store side, but not bandwidth: for a fixed `(i, j)` the 16 lanes span 16 sectors holding 8
//! useful bytes each, and the four values of `j` then fill those same sectors, so all the bytes are
//! consumed — the request count is 4x, not the traffic. Staging `C` through shared memory is the
//! fix if that ever measures badly.
//!
//! # Order
//!
//! `ColMajor` is not special-cased in the kernel. The geometry module reverses every layout and
//! swaps the operands for that order, so the kernel only ever sees row-major-decomposed blocks with
//! leading dimensions.
//!
//! # What is deliberately absent
//!
//! No double buffering, no 128-bit vectorised loads, no `C` staging, no split-K, no cuBLAS. Batch
//! axes ride in `grid.z` and must be *affine* (one step per operand); a non-affine batch, like an
//! operand with no unit-stride axis, is declined here and runs on the naive driver.
//!
//! The kernel is single-`@T@` (it spells its zero as `(@T@)0`), so it serves `f32` and `f64` only.

use std::any::TypeId;

use super::super::geom::GemmGeom;
use super::{Buffers as Uniform, Problem, Refusal, Verdict};
use crate::dtype::CudaDType;
use crate::kernel::{order_input, order_output, FlatLaunch};
use crate::kernels::elementwise::planned_function;
use crate::prelude_dev::*;

/// Tile of `C` computed by one block.
const BM: u32 = 64;
const BN: u32 = 64;
/// Depth of one shared-memory slab along the contraction.
const BK: u32 = 16;
/// Outputs per thread, per axis.
const TM: u32 = 4;
const TN: u32 = 4;
const TX: u32 = BN / TN;
const TY: u32 = BM / TM;
const NTHREADS: u32 = TX * TY;

/// Elements per thread when filling one slab: `BM*BK` and `BK*BN` over the block size.
const A_SLAB: u32 = BM * BK / NTHREADS;
const B_SLAB: u32 = BK * BN / NTHREADS;

const _: () = assert!(TX * TY == 256, "the block must be 256 threads");
const _: () = assert!(BM * BK % NTHREADS == 0 && BK * BN % NTHREADS == 0);
const _: () = assert!(TM * TY == BM && TN * TX == BN);

/// Which axis of a 2-D operand the linear thread id enumerates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaneAxis {
    /// Contraction axis (`k`).
    K,
    /// The other one: `m` for the left operand, `n` for the right.
    Other,
}

const GEMM_ENTRY: &str = r#"
extern "C" __global__ void rstsr_gemm(
    @T@* c,     rstsr_isz scm, rstsr_isz scn, rstsr_isz scb,
    const @T@* a, rstsr_isz sam, rstsr_isz sak, rstsr_isz sab,
    const @T@* b, rstsr_isz sbk, rstsr_isz sbn, rstsr_isz sbb,
    int M, int N, int K,
    @T@ alpha, @T@ beta)
{
    __shared__ @T@ As[@BK@][@BM@ + 1];
    __shared__ @T@ Bs[@BK@][@BN@ + 1];

    const int m0 = (int)blockIdx.y * @BM@;
    const int n0 = (int)blockIdx.x * @BN@;
    const int tid = (int)threadIdx.y * @TX@ + (int)threadIdx.x;
    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    // batch: one grid slice per output matrix, affine in the slice index
    const @T@* ap = a + (rstsr_isz)blockIdx.z * sab;
    const @T@* bp = b + (rstsr_isz)blockIdx.z * sbb;
    @T@*       cp = c + (rstsr_isz)blockIdx.z * scb;

    @T@ acc[@TM@][@TN@];
    #pragma unroll
    for (int i = 0; i < @TM@; ++i)
        #pragma unroll
        for (int j = 0; j < @TN@; ++j) acc[i][j] = (@T@)0;

    for (int k0 = 0; k0 < K; k0 += @BK@) {
        // A slab, [BM x BK], stored transposed as As[k][m] so the inner loop reads one shared
        // double per (i, k) instead of a strided column.
        #pragma unroll
        for (int t = 0; t < @A_SLAB@; ++t) {
            rstsr_usz lin = (rstsr_usz)(tid + t * @NTHREADS@);
            @A_IDX@
            const int gm = m0 + (int)mm;
            const int gk = k0 + (int)kk;
            As[kk][mm] = (gm < M && gk < K)
                ? ap[(rstsr_isz)gm * sam + (rstsr_isz)gk * sak]
                : (@T@)0;
        }
        // B slab, [BK x BN], stored as Bs[k][n].
        #pragma unroll
        for (int t = 0; t < @B_SLAB@; ++t) {
            rstsr_usz lin = (rstsr_usz)(tid + t * @NTHREADS@);
            @B_IDX@
            const int gk = k0 + (int)kk;
            const int gn = n0 + (int)nn;
            Bs[kk][nn] = (gk < K && gn < N)
                ? bp[(rstsr_isz)gk * sbk + (rstsr_isz)gn * sbn]
                : (@T@)0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < @BK@; ++kk) {
            @T@ av[@TM@], bv[@TN@];
            #pragma unroll
            for (int i = 0; i < @TM@; ++i) av[i] = As[kk][ty + i * @TY@];
            #pragma unroll
            for (int j = 0; j < @TN@; ++j) bv[j] = Bs[kk][tx + j * @TX@];
            #pragma unroll
            for (int i = 0; i < @TM@; ++i)
                #pragma unroll
                for (int j = 0; j < @TN@; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }

    // `beta == 0` must not read `c`: that is the contract of `matmul_uninit`, whose buffer is
    // `MaybeUninit` and has no meaningful contents.
    #pragma unroll
    for (int i = 0; i < @TM@; ++i) {
        const int gm = m0 + ty + i * @TY@;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < @TN@; ++j) {
            const int gn = n0 + tx + j * @TX@;
            if (gn >= N) continue;
            @T@* p = cp + (rstsr_isz)gm * scm + (rstsr_isz)gn * scn;
            *p = (beta == (@T@)0) ? (alpha * acc[i][j]) : (alpha * acc[i][j] + beta * (*p));
        }
    }
}
"#;

fn render(template: &str, subs: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (name, text) in subs {
        s = s.replace(name, text);
    }
    s
}

/// Pick the axis the linear thread id enumerates, preferring a unit-stride one.
///
/// `s_other`/`s_k` are the strides of the non-contracted and contracted axes, `n_other`/`n_k` their
/// extents. A degenerate axis (extent 1) is free to take: it contributes no address variation, so
/// following the other axis is coalesced whatever its stride happens to be.
fn lane_axis(s_other: isize, s_k: isize, n_other: usize, n_k: usize) -> Option<LaneAxis> {
    if s_k == 1 {
        Some(LaneAxis::K)
    } else if s_other == 1 {
        Some(LaneAxis::Other)
    } else if n_other == 1 {
        Some(LaneAxis::K)
    } else if n_k == 1 {
        Some(LaneAxis::Other)
    } else {
        None
    }
}

/// One attempt of the tiled driver.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    // the single-type kernel covers f32 and f64 only
    if TypeId::of::<TC>() == TypeId::of::<f32>() {
        return typed(p.uniform::<f32>(), p);
    }
    if TypeId::of::<TC>() == TypeId::of::<f64>() {
        return typed(p.uniform::<f64>(), p);
    }
    Ok(Verdict::Declined(Refusal::Cannot("the single-type tiled kernel covers f32 and f64 only")))
}

/// Launch `c = alpha * a @ b + beta * c`, or decline with the reason. Errors are launch failures.
///
/// `c` is read by address only (its element type is `T` or `MaybeUninit<T>`), and it is not read
/// at all on the device when `beta == 0`: that is the contract of `matmul_uninit`.
fn typed<T: CudaDType, TA, TB, TC, TBuf>(
    u: Option<Uniform<'_, T, T, T>>,
    p: &Problem<'_, TA, TB, TC, TBuf>,
) -> Result<Verdict> {
    let Some(u) = u else {
        return Ok(Verdict::Declined(Refusal::Cannot("the operands and the output must share one dtype")));
    };
    let geom = &p.geom;
    let GemmGeom { m, n, k, n_batch, .. } = geom;
    let (m, n, k, n_batch) = (*m, *n, *k, *n_batch);
    let decline = |what: &'static str| Ok(Verdict::Declined(Refusal::Cannot(what)));

    if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
        return decline("a dimension exceeds i32");
    }
    let grid_x = (n as u64).div_ceil(BN as u64);
    let grid_y = (m as u64).div_ceil(BM as u64);
    if grid_y > u16::MAX as u64 || n_batch as u64 > u16::MAX as u64 {
        return decline("more than 65535 row tiles or batch slices (the grid limit on y and z)");
    }
    // Batch: one grid slice per output matrix, so the batch offset must be affine.
    if geom.affine_batch().is_none() {
        return decline("the batch part is not one linear run (grid.z carries an affine batch only)");
    }
    let [sa_b, sb_b, sc_b] = geom.affine_batch().unwrap_or([0; 3]);
    let (a_off, sam, sak) = geom.a;
    let (b_off, sbk, sbn) = geom.b;
    let (c_off, scm, scn) = geom.c;
    // An operand with no unit-stride axis: the lanes have nothing to follow.
    let Some(lane_a) = lane_axis(sam, sak, m, k) else {
        return decline("an operand has no unit-stride axis (strides of the left operand)");
    };
    let Some(lane_b) = lane_axis(sbn, sbk, n, k) else {
        return decline("the right operand has no unit-stride axis");
    };

    let exec = p.device.stream();
    let (buf_a, buf_b) = if geom.swapped { (u.b, u.a) } else { (u.a, u.b) };
    let _ga = order_input(buf_a, exec)?;
    let _gb = order_input(buf_b, exec)?;
    let _gc = order_output(u.c, exec)?;

    // The two fragments differ only in which axis the linear id follows; that is what makes each
    // variant coalesced on both operands' loads.
    let a_idx = match lane_a {
        LaneAxis::K => "rstsr_usz kk = lin % @BK@; rstsr_usz mm = lin / @BK@;",
        LaneAxis::Other => "rstsr_usz mm = lin % @BM@; rstsr_usz kk = lin / @BM@;",
    };
    let b_idx = match lane_b {
        LaneAxis::K => "rstsr_usz kk = lin % @BK@; rstsr_usz nn = lin / @BK@;",
        LaneAxis::Other => "rstsr_usz nn = lin % @BN@; rstsr_usz kk = lin / @BN@;",
    };

    // Order matters: the fragments are text that still contains the numeric tokens, so they must be
    // expanded *before* the numbers, or an `@BK@` inside a fragment survives into the CUDA source.
    let subs: Vec<(&str, String)> = vec![
        ("@A_IDX@", a_idx.to_string()),
        ("@B_IDX@", b_idx.to_string()),
        ("@T@", T::CNAME.to_string()),
        ("@BM@", BM.to_string()),
        ("@BN@", BN.to_string()),
        ("@BK@", BK.to_string()),
        ("@TM@", TM.to_string()),
        ("@TN@", TN.to_string()),
        ("@TX@", TX.to_string()),
        ("@TY@", TY.to_string()),
        ("@NTHREADS@", NTHREADS.to_string()),
        ("@A_SLAB@", A_SLAB.to_string()),
        ("@B_SLAB@", B_SLAB.to_string()),
    ];
    let subs: Vec<(&str, &str)> = subs.iter().map(|(k, v)| (*k, v.as_str())).collect();

    // The discriminator must name every substitution the source depends on, or two variants collide
    // in the cache and one launches the other's code.
    let disc = format!(
        "{}|{}|{}|BM{BM}|BN{BN}|BK{BK}|TM{TM}|TN{TN}|N{NTHREADS}",
        T::CNAME,
        match lane_a {
            LaneAxis::K => "ak",
            LaneAxis::Other => "am",
        },
        match lane_b {
            LaneAxis::K => "bk",
            LaneAxis::Other => "bn",
        },
    );
    let f = planned_function(p.device, "rstsr_gemm", T::INCLUDES, &disc, || render(GEMM_ENTRY, &subs))?;

    let nb = core::mem::size_of::<T>();
    let abytes = super::scalar_bytes(&p.alpha);
    let bbytes = super::scalar_bytes(&p.beta);

    FlatLaunch::new(exec, &f, 0)
        .explicit_shape((grid_x as u32, grid_y as u32, n_batch as u32), (TX, TY, 1))
        .arg_ptr(u.c, c_off)
        .arg_bytes(&scm.to_ne_bytes())
        .arg_bytes(&scn.to_ne_bytes())
        .arg_bytes(&sc_b.to_ne_bytes())
        .arg_ptr(buf_a, a_off)
        .arg_bytes(&sam.to_ne_bytes())
        .arg_bytes(&sak.to_ne_bytes())
        .arg_bytes(&sa_b.to_ne_bytes())
        .arg_ptr(buf_b, b_off)
        .arg_bytes(&sbk.to_ne_bytes())
        .arg_bytes(&sbn.to_ne_bytes())
        .arg_bytes(&sb_b.to_ne_bytes())
        .arg_i32(m as i32)
        .arg_i32(n as i32)
        .arg_i32(k as i32)
        .arg_bytes(&abytes[..nb])
        .arg_bytes(&bbytes[..nb])
        .launch()?;

    Ok(Verdict::Ran)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane map prefers the unit-stride axis, and falls back to the axis that actually varies
    /// when the other is degenerate.
    #[test]
    fn lane_axis_prefers_the_unit_stride() {
        assert!(matches!(lane_axis(64, 1, 64, 64), Some(LaneAxis::K)));
        assert!(matches!(lane_axis(1, 64, 64, 64), Some(LaneAxis::Other)));
        // [1, K] with a strided k axis: nothing is unit-stride, but m is degenerate, so following
        // k is still the right map.
        assert!(matches!(lane_axis(0, 2, 1, 64), Some(LaneAxis::K)));
        // genuinely stepped on both axes
        assert!(lane_axis(2, 64, 64, 64).is_none());
    }

    /// Tile geometry has to stay self-consistent: the kernel derives slab counts from it and the
    /// shared arrays from `BM`/`BN`.
    #[test]
    fn tile_geometry_is_consistent() {
        assert_eq!(TX * TY, NTHREADS);
        assert_eq!(TM * TY, BM);
        assert_eq!(TN * TX, BN);
        assert_eq!(BM * BK, A_SLAB * NTHREADS);
        assert_eq!(BK * BN, B_SLAB * NTHREADS);
        // static shared memory for both slabs must fit the 48 KB static limit
        let smem = (BK * (BM + 1) + BK * (BN + 1)) * 8;
        assert!(smem <= 48 * 1024, "static shared memory {smem} B is over the 48 KB limit");
    }
}
