//! Host-side launch planning for elementwise kernels.
//!
//! Each operand layout is analyzed once per launch into an [`OperandPlan`]:
//! a flat run (pointer-only indexing), a rank-≤4 magic-division walk, or a
//! signal to fall back to the rank-generic scratch kernel. Magic division
//! multipliers are precomputed here so device-side index math never runs
//! the 64-bit hardware-division emulation.

use crate::prelude_dev::*;

/// One decomposition axis of a parameterized operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AxisPlan {
    /// Divisor: the axis shape (product of merged shapes).
    pub d: u64,
    /// Element-unit stride (wrapping; negative strides wrap exactly like
    /// the rank-generic kernel's u64 arithmetic).
    pub s: u64,
    /// Magic multiplier for `d` (0 = power-of-two sentinel: shift only).
    pub m: u64,
    /// Magic shift (`floor(log2 d)`, or `log2 d` when `m == 0`).
    pub sh: u32,
}

impl AxisPlan {
    /// Magic numbers for one axis of shape `d >= 2`.
    ///
    /// Round-up method (Granlund–Montgomery): with `l = floor(log2 d)` and
    /// `m = floor(2^(64+l)/d) + 1`, `q = mulhi(n, m) >> l` equals `n / d`
    /// for all `n < 2^63` (proof: writing `m*d = 2^(64+l) + e` with
    /// `e <= d`, the quotient is short by `e*n/2^(64+l) < 1`).
    fn new(d: u64) -> Self {
        debug_assert!(d >= 2, "shape-1 axes are dropped before planning");
        if d.is_power_of_two() {
            Self { d, s: 0, m: 0, sh: d.trailing_zeros() }
        } else {
            let l = 63 - d.leading_zeros();
            let m = (((1u128 << (64 + l)) / d as u128) as u64).wrapping_add(1);
            Self { d, s: 0, m, sh: l }
        }
    }

    /// `n / d` mirroring the device-side math (test oracle).
    #[cfg(test)]
    fn div(&self, n: u64) -> u64 {
        if self.m == 0 {
            n >> self.sh
        } else {
            (((n as u128 * self.m as u128) >> 64) as u64) >> self.sh
        }
    }
}

/// Fixed-width parameter pack for a `Params` operand (rank ≤ 4 after
/// squash; longer walks fall back to the rank-generic kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AxesPack {
    pub axes: [AxisPlan; 4],
    pub rank: usize,
}

/// One axis of a 32-bit parameterized operand: divisors/strides fit
/// `u32`/`i32` and the accumulated offset range stays within `i32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AxisPlanN {
    pub d: u32,
    pub s: i32,
    pub m: u32,
    pub sh: u32,
}

impl AxisPlanN {
    /// 32-bit round-up magic (`__umulhi`); valid for `lin < 2^31`.
    fn new(d: u32) -> Self {
        debug_assert!(d >= 2, "shape-1 axes are dropped before planning");
        if d.is_power_of_two() {
            Self { d, s: 0, m: 0, sh: d.trailing_zeros() }
        } else {
            let l = 31 - d.leading_zeros();
            let m = (((1u64 << (32 + l)) / d as u64) as u32).wrapping_add(1);
            Self { d, s: 0, m, sh: l }
        }
    }

    /// `n / d` mirroring the device-side narrow math (test oracle).
    #[cfg(test)]
    fn div(&self, n: u32) -> u32 {
        if self.m == 0 {
            n >> self.sh
        } else {
            (((n as u64 * self.m as u64) >> 32) as u32) >> self.sh
        }
    }
}

/// Fixed-width 32-bit parameter pack (device struct is 64 B of axes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AxesPackN {
    pub axes: [AxisPlanN; 4],
    pub rank: usize,
}

/// Per-operand index plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandPlan {
    /// Flat run: element offsets fold into the base pointer at launch;
    /// the device index is the linear id itself.
    Flat,
    /// Magic-divmod walk over `rank` axes (decomposition order).
    Params(AxesPack),
    /// 32-bit walk: divisors/strides narrow, offsets accumulate in `i32`
    /// (wrapping arithmetic makes negative strides exact).
    ParamsN(AxesPackN),
    /// No plan: use the rank-generic scratch kernel.
    Generic,
}

impl OperandPlan {
    pub fn is_generic(&self) -> bool {
        matches!(self, OperandPlan::Generic)
    }
}

/// Plan one operand of a launch against `n` output elements.
///
/// Axes are visited in the device's decomposition order (row-major walks
/// last-axis-first), shape-1 axes are dropped, and adjacent axes merge
/// when the walk stays linear in the merged index (inner stride equals
/// the outer stride, or steps by exactly one inner period). The result
/// covers every layout the elementwise tier sees: operands arrive at the
/// device layer broadcast-padded to the output shape and rank.
pub(crate) fn plan_operand<D: DimAPI>(layout: &Layout<D>, order: FlagOrder, n: usize) -> OperandPlan {
    if n == 0 {
        return OperandPlan::Flat; // unused; callers return before planning
    }
    let shape = layout.shape().as_ref();
    let stride = layout.stride().as_ref();
    // decomposition order, innermost first
    let range: Box<dyn Iterator<Item = usize>> = match order {
        RowMajor => Box::new((0..shape.len()).rev()),
        ColMajor => Box::new(0..shape.len()),
    };
    // squashed walk, innermost first
    let mut sq: Vec<(u64, u64)> = Vec::with_capacity(shape.len());
    for ax in range {
        let (sh, st) = (shape[ax] as u64, stride[ax] as u64);
        if sh == 1 {
            continue;
        }
        let (mut csh, mut cst) = (sh, st);
        while let Some(&(lsh, lst)) = sq.last() {
            // merged walk stays a valid divmod level: (d0,s0)+(d1,s1) ->
            // (d0*d1, s0) iff s1 == d0*s0, i.e. the outer stride continues
            // the inner run; both-zero strides (broadcast pairs) merge too.
            // The merged level keeps the INNER stride.
            if cst == lst.wrapping_mul(lsh) || (cst == 0 && lst == 0) {
                sq.pop();
                csh = lsh.wrapping_mul(csh);
                cst = lst;
            } else {
                break;
            }
        }
        sq.push((csh, cst));
    }
    match sq.len() {
        0 => OperandPlan::Flat, // n == 1: every axis dropped
        // genuinely flat: a single stride-1 run spanning exactly n elements
        // (a broadcast operand keeps its divmod level even when stride == 1;
        // the walk discards the top quotient, implementing the broadcast)
        1 if sq[0].1 == 1 && sq[0].0 == n as u64 => OperandPlan::Flat,
        rank if rank <= 4 => {
            if n < (1 << 31) {
                if let Some(pack) = narrow_pack(&sq) {
                    return OperandPlan::ParamsN(pack);
                }
            }
            let mut axes = [AxisPlan { d: 1, s: 0, m: 0, sh: 0 }; 4];
            for (i, &(d, s)) in sq.iter().enumerate() {
                let mut plan = AxisPlan::new(d);
                plan.s = s;
                axes[i] = plan;
            }
            OperandPlan::Params(AxesPack { axes, rank })
        },
        _ => OperandPlan::Generic,
    }
}

/// Narrow-pack construction: every divisor < 2^31, strides fit `i32`, and
/// the whole offset walk (including base offsets of at most `n` elements)
/// stays within `i32` so signed 32-bit accumulation is exact (negative
/// strides wrap exactly in two's complement).
fn narrow_pack(sq: &[(u64, u64)]) -> Option<AxesPackN> {
    let mut max_off: u64 = 0;
    let mut axes = [AxisPlanN { d: 1, s: 0, m: 0, sh: 0 }; 4];
    for (i, &(d, s)) in sq.iter().enumerate() {
        // stride magnitude: negative strides arrive wrapping as > i64::MAX
        let mag = if s > i64::MAX as u64 { s.wrapping_neg() } else { s };
        if d >= (1 << 31) || mag > i32::MAX as u64 {
            return None;
        }
        max_off = max_off.saturating_add((d - 1).saturating_mul(mag));
        if max_off > i32::MAX as u64 {
            return None;
        }
        let mut plan = AxisPlanN::new(d as u32);
        plan.s = s as i32;
        axes[i] = plan;
    }
    Some(AxesPackN { axes, rank: sq.len() })
}

#[cfg(test)]
mod test {
    use super::*;

    fn plan_of(shape: &[usize], stride: &[isize], order: FlagOrder, n: usize) -> OperandPlan {
        // SAFETY: test fixture; shapes and strides are consistent by construction
        let layout: Layout<Vec<usize>> = unsafe { Layout::new_unchecked(shape.to_vec(), stride.to_vec(), 0) };
        plan_operand(&layout, order, n)
    }

    /// (rank, strides outermost-first, divisors outermost-first) of a
    /// parameterized plan, either width; narrow strides are sign-extended.
    fn pack_view(p: &OperandPlan) -> (usize, Vec<i64>, Vec<u64>) {
        match p {
            OperandPlan::Params(a) => (
                a.rank,
                a.axes[..a.rank].iter().map(|ax| ax.s as i64).collect(),
                a.axes[..a.rank].iter().map(|ax| ax.d).collect(),
            ),
            OperandPlan::ParamsN(a) => (
                a.rank,
                a.axes[..a.rank].iter().map(|ax| ax.s as i64).collect(),
                a.axes[..a.rank].iter().map(|ax| ax.d as u64).collect(),
            ),
            other => panic!("expected parameterized plan, got {other:?}"),
        }
    }

    #[test]
    fn magic_division_matches_hardware() {
        // every divisor up to 2^16 against boundary and random numerators
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let numerators: [u64; 12] =
            [0, 1, 2, 3, 63, 64, 65, 1000, (1 << 31) - 1, 1 << 31, (1 << 62) + 1, (1 << 63) - 1];
        for d in 2u64..=(1 << 16) {
            let ax = AxisPlan::new(d);
            for &n in &numerators {
                assert_eq!(ax.div(n), n / d, "magic div failed for d={d} n={n}");
            }
            for _ in 0..8 {
                let n = next() >> 1; // < 2^63
                assert_eq!(ax.div(n), n / d, "magic div failed for d={d} n={n}");
            }
        }
        // large divisors, powers of two, and near-maximum numerators
        for d in [(1u64 << 20) + 1, (1 << 32) + 7, (1 << 40) - 3, 1 << 40, (1 << 62) + 1] {
            let ax = AxisPlan::new(d);
            for &n in &numerators {
                assert_eq!(ax.div(n), n / d, "magic div failed for d={d} n={n}");
            }
        }
    }

    #[test]
    fn pow2_uses_shift_sentinel() {
        let ax = AxisPlan::new(512);
        assert_eq!(ax.m, 0);
        assert_eq!(ax.sh, 9);
        assert_eq!(ax.div(12345), 12345 / 512);
    }

    #[test]
    fn narrow_magic_division_matches_hardware() {
        let mut state = 0x9E3779B9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let numerators: [u32; 10] = [0, 1, 2, 3, 63, 64, 65, 1000, (1 << 16) + 1, (1 << 31) - 1];
        for d in 2u32..=(1 << 14) {
            let ax = AxisPlanN::new(d);
            for &n in &numerators {
                assert_eq!(ax.div(n), n / d, "narrow magic div failed for d={d} n={n}");
            }
            for _ in 0..8 {
                let n = next() & ((1 << 31) - 1);
                assert_eq!(ax.div(n), n / d, "narrow magic div failed for d={d} n={n}");
            }
        }
    }

    #[test]
    fn narrow_pack_bound_checks() {
        // ordinary 2-D strides fit
        let pack = narrow_pack(&[(10, 30), (30, 1)]);
        assert!(pack.is_some());
        // negative stride fits (wrapping representation)
        let neg: u64 = (-6isize) as u64;
        let pack = narrow_pack(&[(10, neg), (30, 1)]);
        assert!(pack.is_some());
        // oversized stride rejected
        assert!(narrow_pack(&[(10, 1 << 40), (30, 1)]).is_none());
        // accumulated offset beyond i32 rejected
        assert!(narrow_pack(&[(1 << 30, 4), (4, 1)]).is_none());
        // oversized divisor rejected
        assert!(narrow_pack(&[(1 << 31, 1)]).is_none());
    }

    #[test]
    fn contiguous_and_sliced_rows() {
        // fully contiguous 3-D
        let p = plan_of(&[16, 256, 512], &[131072, 512, 1], RowMajor, 16 * 256 * 512);
        assert_eq!(p, OperandPlan::Flat);
        // row-sliced 2-D: one outer axis + one stride-1 run stays parameterized
        let p = plan_of(&[1024, 1000], &[1280, 1], RowMajor, 1024 * 1000);
        let (rank, strides, divs) = pack_view(&p);
        assert_eq!(rank, 2);
        assert_eq!(strides, [1, 1280]);
        assert_eq!(divs, [1000, 1024]);
    }

    #[test]
    fn transposed_and_broadcast() {
        // transposed 2-D operand: two axes, no merge
        let p = plan_of(&[1000, 1000], &[1, 1000], RowMajor, 1_000_000);
        let (rank, strides, _) = pack_view(&p);
        assert_eq!(rank, 2);
        assert_eq!(strides, [1000, 1]);
        // broadcast axis of stride 0 (a[:, None, :] padded to output rank):
        // a stride-0 axis is NOT a shape-1 axis — it stays a divmod axis
        let p = plan_of(&[16, 256, 512], &[131072, 0, 1], RowMajor, 16 * 256 * 512);
        let (rank, strides, _) = pack_view(&p);
        assert_eq!(rank, 3);
        assert_eq!(strides, [1, 0, 131072]);
        // shape-1 axis between contig axes: the whole thing is contiguous
        let p = plan_of(&[8, 1, 6], &[6, 6, 1], RowMajor, 48);
        assert_eq!(p, OperandPlan::Flat);
        // fully broadcast operand collapses flat (n == 1 shape or all-zero
        // strides merge into a stride-0 single axis)
        let p = plan_of(&[4, 4], &[0, 0], RowMajor, 16);
        let (rank, strides, _) = pack_view(&p);
        assert_eq!(rank, 1);
        assert_eq!(strides, [0]);
    }

    #[test]
    fn col_major_order_flips_decomposition() {
        // column-major device: axis 0 is innermost; a col-contiguous 2-D
        // tensor is flat
        let p = plan_of(&[8, 6], &[1, 8], ColMajor, 48);
        assert_eq!(p, OperandPlan::Flat);
        // same strides on a row-major device: transposed, two axes
        let p = plan_of(&[8, 6], &[1, 8], RowMajor, 48);
        assert!(matches!(p, OperandPlan::Params(_) | OperandPlan::ParamsN(_)));
    }

    #[test]
    fn shape1_axes_drop_to_flat() {
        let p = plan_of(&[1, 1, 1], &[0, 0, 0], RowMajor, 1);
        assert_eq!(p, OperandPlan::Flat);
    }

    #[test]
    fn negative_stride_stays_parameterized() {
        let p = plan_of(&[1000], &[-1], RowMajor, 1000);
        let (rank, strides, _) = pack_view(&p);
        assert_eq!(rank, 1);
        assert_eq!(strides, [-1]);
    }

    #[test]
    fn deep_rank_falls_back() {
        // 5 unsquashable axes exceed the 4-axis pack
        let shape = [3, 3, 3, 3, 3];
        let stride = [1, 100, 10000, 1_000_000, 100_000_000];
        let p = plan_of(&shape, &stride, RowMajor, 243);
        assert_eq!(p, OperandPlan::Generic);
        // 5 axes, 2 levels in col-major: the inner three form a run
        let p = plan_of(&shape, &[1, 3, 9, 1000, 3000], ColMajor, 243);
        let (rank, strides, divs) = pack_view(&p);
        assert_eq!(rank, 2);
        assert_eq!(divs, [27, 9]);
        assert_eq!(strides, [1, 1000]);
        // row-major contiguous 5-axis squashes flat
        let p = plan_of(&shape, &[81, 27, 9, 3, 1], RowMajor, 243);
        assert_eq!(p, OperandPlan::Flat);
    }
}
