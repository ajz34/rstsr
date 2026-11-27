//! cuBLAS `gemm` arguments, derived from the geometry — or `None` when the layout is not
//! expressible.
//!
//! cuBLAS is column-major only, and `C` has **no transpose flag** — only `ldc`. So the whole
//! mapping rests on the identity `C = A·B  <=>  Cᵀ = Bᵀ·Aᵀ`, which fixes the roles completely:
//!
//! | cuBLAS slot | our operand | must present as | how |
//! |---|---|---|---|
//! | `op(A_o)` | `geom.b` (`k × n`) | `n × k` | `n` contiguous → `OP_N`, ld = `sbk`; else `k` contiguous → `OP_T`, ld = `sbn` |
//! | `op(B_o)` | `geom.a` (`m × k`) | `k × m` | `k` contiguous → `OP_N`, ld = `sam`; else `m` contiguous → `OP_T`, ld = `sak` |
//! | `C_o` | `geom.c` (`m × n`) | `n × m` | requires `scn == 1`, ld = `scm` |
//!
//! with `m_o = n`, `n_o = m`, `k_o = k`. Note that the **operands are swapped relative to their
//! mathematical roles**: `geom.b` goes in cuBLAS's A slot. Getting either the swap or one `OP_*`
//! flag wrong produces a plausible transposed answer rather than an error, which is why [`args`]
//! is pure and is tested by re-deriving the same products through the declared leading dimensions
//! ([`tests::declared_lds_reproduce_the_product`]).
//!
//! `geom` is already order-normalised: [`crate::matmul::geom::resolve_geom`] reverses and swaps for
//! `ColMajor`, so this module only ever sees a row-major description.
//!
//! # What is not expressible
//!
//! - `scn != 1`: the output cannot be presented as `n × m` column-major. Fresh outputs never hit
//!   this; a stepped or transposed view does.
//! - neither stride of an operand is 1: cuBLAS needs one leading dimension, so a stepped 2-D block
//!   needs a pack.
//! - a negative stride with extent > 1: no negative leading dimension.
//! - a batch part that is not one linear run (cuBLAS takes one step per operand), or a dimension
//!   over `i32`.
//!
//! Every refusal hands the problem to the next driver.

use crate::matmul::geom::GemmGeom;
use crate::prelude_dev::*;

/// One `gemm` call, in cuBLAS's terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CublasArgs {
    pub mo: i32,
    pub no: i32,
    pub ko: i32,
    pub transa: bool,
    pub transb: bool,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    /// Batch element steps, named after the **geom** operands. `batch_b` belongs to `geom.b`, which
    /// the cuBLAS driver places in cuBLAS's A slot — the naming follows the data, not the slot.
    ///
    /// **These are element counts, not bytes.** cuBLAS documents `strideA` as "the offset in number
    /// of elements between `A[i]` and `A[i+1]`", which is the opposite of what `cublasSetVector`'s
    /// `elemSize` does two sections earlier, and getting it wrong is invisible for `batchCount ==
    /// 1`.
    pub batch_a: isize,
    pub batch_b: isize,
    pub batch_c: isize,
    pub batch: i32,
}

/// Largest leading dimension cuBLAS accepts for a declared matrix of `rows` rows.
///
/// An extent-1 axis may carry stride 0; then the leading dimension would be 0, which cuBLAS
/// rejects, and clamping to the declared row count is safe because that axis has only index 0.
/// The guard in `args` keeps every other zero stride out.
fn ld(stride: isize, rows: usize) -> i32 {
    let s = stride.max(0);
    let clamped = s.max(rows as isize);
    clamped.min(i32::MAX as isize) as i32
}

/// Derive the cuBLAS arguments, or `None` when this layout is not expressible.
///
/// Pure on purpose: this is the part where a mistake produces a silently transposed answer, so it
/// is tested against the declared leading dimensions rather than only on hardware.
pub(crate) fn args(geom: &GemmGeom) -> Option<CublasArgs> {
    let (m, n, k) = (geom.m, geom.n, geom.k);
    let (_, sam, sak) = geom.a;
    let (_, sbk, sbn) = geom.b;
    let (_, scm, scn) = geom.c;

    // An extent-1 axis may legitimately carry stride 0; anything else with a non-positive stride
    // (a broadcast along a matmul axis, or a reversed view) has no cuBLAS expression at all.
    for (stride, extent) in [(sam, m), (sak, k), (sbk, k), (sbn, n), (scm, m), (scn, n)] {
        if stride <= 0 && extent > 1 {
            return None;
        }
    }
    // `C` has no transpose flag: it must be presentable as `n x m` column-major.
    if scn != 1 {
        return None;
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
        return None;
    }
    if geom.n_batch > i32::MAX as usize {
        return None;
    }
    // one step per operand: a batch that is not one linear run has no expression
    let [sa_b, sb_b, sc_b] = geom.affine_batch()?;

    // Slot A = geom.b, which must appear as n x k.
    let (transa, lda_raw, rows_a) = if sbn == 1 {
        (false, sbk, n)
    } else if sbk == 1 {
        (true, sbn, k)
    } else {
        return None;
    };
    // Slot B = geom.a, which must appear as k x m.
    let (transb, ldb_raw, rows_b) = if sak == 1 {
        (false, sam, k)
    } else if sam == 1 {
        (true, sak, m)
    } else {
        return None;
    };

    let (lda, ldb, ldc) = (ld(lda_raw, rows_a), ld(ldb_raw, rows_b), ld(scm, n));
    // `ld(a/b)` went through `max(0)`, so a genuinely negative stride can hide there; re-check.
    if lda_raw < 0 || ldb_raw < 0 || ldc == 0 {
        return None;
    }

    Some(CublasArgs {
        mo: n as i32,
        no: m as i32,
        ko: k as i32,
        transa,
        transb,
        lda,
        ldb,
        ldc,
        batch_a: sa_b,
        batch_b: sb_b,
        batch_c: sc_b,
        batch: geom.n_batch as i32,
    })
}

/// The `strideA`, `strideB`, `strideC` triple cuBLAS receives, in **elements**, ordered by cuBLAS's
/// slots rather than by our operand names: `geom.b` sits in slot A, `geom.a` in slot B.
///
/// This is a separate function for one reason. cuBLAS's typed strided-batched API takes element
/// offsets — "the offset in number of elements between `A[i]` and `A[i+1]`" — while
/// `cublasSetVector` two sections earlier in the same manual takes bytes. The conversion therefore
/// must **not** happen anywhere, and a `* size_of::<T>()` inserted here is invisible to every
/// `batchCount == 1` test: it leaves the first slice exact, loses the rest, and writes the surplus
/// into whatever the allocator placed next. Naming the triple keeps the unit in one place, where
/// [`tests::batch_strides_are_element_counts`] can assert it.
pub(crate) fn batch_strides(a: &CublasArgs) -> (i64, i64, i64) {
    (a.batch_b as i64, a.batch_a as i64, a.batch_c as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::geom::resolve_geom;

    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), offset).unwrap()
    }

    /// Element index of the operand in cuBLAS's **A** slot (`geom.b`), addressed the way the
    /// *declared* matrix and `lda` say it should be — not the way `geom` says.
    fn slot_a(g: &GemmGeom, a: &CublasArgs, ni: usize, ki: usize) -> usize {
        let base = g.b.0;
        if a.transa {
            // declared k x n, read transposed
            base + ki + ni * a.lda as usize
        } else {
            // declared n x k
            base + ni + ki * a.lda as usize
        }
    }

    /// Element index of the operand in cuBLAS's **B** slot (`geom.a`).
    fn slot_b(g: &GemmGeom, a: &CublasArgs, ki: usize, mi: usize) -> usize {
        let base = g.a.0;
        if a.transb {
            base + mi + ki * a.ldb as usize
        } else {
            base + ki + mi * a.ldb as usize
        }
    }

    /// The output, as `n x m` column-major.
    fn slot_c(g: &GemmGeom, a: &CublasArgs, ni: usize, mi: usize) -> usize {
        g.c.0 + ni + mi * a.ldc as usize
    }

    fn data(n: usize, seed: usize) -> Vec<f64> {
        (0..n).map(|i| (((i * 7 + seed) % 9) as f64) - 4.0).collect()
    }

    /// Every orientation the derivation claims to support, plus a non-zero offset on all three
    /// layouts: the `gemm` result re-derived through the *declared* leading dimensions must equal
    /// the one computed through `geom`'s own strides. A wrong `OP_*` flag, a wrong `ld`, or a
    /// missing operand swap changes the answer here rather than in a rented-GPU session.
    #[test]
    fn declared_lds_reproduce_the_product() {
        let (m, k, n) = (3usize, 4usize, 5usize);
        // (a's strides, b's strides, label). One of each operand's two axes must be unit.
        let cases: &[([isize; 2], [isize; 2], &str)] = &[
            ([4, 1], [5, 1], "a k-contig, b n-contig"),
            ([1, 4], [5, 1], "a m-contig, b n-contig"),
            ([4, 1], [1, 4], "a k-contig, b k-contig"),
            ([1, 4], [1, 4], "a m-contig, b k-contig"),
        ];
        for (a_stride, b_stride, label) in cases {
            for offset in [0usize, 3] {
                let la = layout(&[m, k], a_stride, offset);
                let lb = layout(&[k, n], b_stride, offset + 1);
                let lc = layout(&[m, n], &[n as isize, 1], offset + 2);
                let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
                let args = args(&g).unwrap_or_else(|| panic!("{label}: should be expressible"));

                let av = data(64, 1);
                let bv = data(64, 2);

                for mi in 0..m {
                    for ni in 0..n {
                        // What the declared lds say this output element is. Note which buffer each
                        // slot reads: cuBLAS's A slot holds `geom.b` (so `bv`), its B slot holds
                        // `geom.a` (so `av`). Swapping these is the mistake this test exists to
                        // catch — and it caught it here first.
                        let declared: f64 =
                            (0..k).map(|ki| bv[slot_a(&g, &args, ni, ki)] * av[slot_b(&g, &args, ki, mi)]).sum();
                        // what the geometry says, straight from each operand's own strides
                        let reference: f64 = (0..k)
                            .map(|ki| {
                                let ai = g.a.0 + mi * g.a.1 as usize + ki * g.a.2 as usize;
                                let bi = g.b.0 + ki * g.b.1 as usize + ni * g.b.2 as usize;
                                av[ai] * bv[bi]
                            })
                            .sum();
                        assert_eq!(declared, reference, "{label}, offset {offset}, C[{mi},{ni}]");
                        // and the output slot addresses the same cell the layout does
                        assert_eq!(
                            slot_c(&g, &args, ni, mi),
                            g.c.0 + mi * g.c.1 as usize + ni * g.c.2 as usize,
                            "{label}, offset {offset}: output addressing"
                        );
                    }
                }
            }
        }
    }

    /// Refusals, each at the stage where it happens.
    ///
    /// The two are not interchangeable: the geometry admits a stepped **output** (other drivers
    /// address `c` by two strides, so they do not care), and it is `args` that has to refuse it,
    /// because cuBLAS's output side has only `ldc` and no transpose flag. A stepped or reversed
    /// **operand** has no cuBLAS expression either: one leading dimension per operand.
    #[test]
    fn refusals_happen_at_the_earliest_stage() {
        let lb = layout(&[4, 3], &[3, 1], 0);

        // stepped output: legal for our kernel, inexpressible for cuBLAS
        let la = layout(&[2, 4], &[4, 1], 0);
        let lc = layout(&[2, 3], &[6, 2], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        assert_eq!((g.c.1, g.c.2), (6, 2), "fixture: the output really is stepped");
        assert!(args(&g).is_none(), "a stepped output has no column-major expression");

        // stepped operand: no leading dimension to declare
        let la2 = layout(&[2, 4], &[16, 2], 0);
        let lc2 = layout(&[2, 3], &[3, 1], 0);
        let g2 = resolve_geom(RowMajor, &lc2, &la2, &lb).unwrap();
        assert!(args(&g2).is_none(), "a stepped operand has no cuBLAS expression");

        // reversed operand: accepted by the classifier (the kernel walks backwards), so this is the
        // one refusal `args` owns — cuBLAS has no negative leading dimension.
        let lc3 = layout(&[3, 3], &[3, 1], 0);
        let lb3 = layout(&[4, 3], &[3, 1], 0);
        let la3 = layout(&[3, 4], &[-4, 1], 8);
        // If the layout itself is rejected as out of bounds the case is unreachable and the guard
        // in `args` is defence in depth rather than a live branch, so only the Ok arm asserts.
        if let Ok(g) = resolve_geom(RowMajor, &lc3, &la3, &lb3) {
            assert!(args(&g).is_none(), "a reversed operand has no cuBLAS expression");
        }
    }

    /// The batch strides are **element** counts and exactly one matrix apart.
    ///
    /// The convention is cuBLAS's, not ours: the typed strided-batched API documents `strideA` as
    /// "the offset in number of elements between `A[i]` and `A[i+1]`", while `cublasSetVector`
    /// takes bytes. A `* size_of::<T>()` in [`batch_strides`] — the one place that produces these
    /// values — is invisible to every `batchCount == 1` fixture: it leaves the first slice exact,
    /// drops the rest, and writes the surplus into whatever the allocator placed next. That is not
    /// a hypothetical: it is what the first hardware run of this path did.
    #[test]
    fn batch_strides_are_element_counts() {
        let (p, m, k, n) = (5usize, 12usize, 16usize, 8usize);
        let la = layout(&[p, m, k], &[(m * k) as isize, k as isize, 1], 0);
        let lb = layout(&[p, k, n], &[(k * n) as isize, n as isize, 1], 0);
        let lc = layout(&[p, m, n], &[(m * n) as isize, n as isize, 1], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        let a = args(&g).unwrap();
        assert_eq!(a.batch, p as i32);

        let (stride_a, stride_b, stride_c) = batch_strides(&a);
        // slot A holds `geom.b` (k x n), slot B holds `geom.a` (m x k)
        assert_eq!(stride_a, (k * n) as i64, "slot A steps by one k x n matrix, in elements");
        assert_eq!(stride_b, (m * k) as i64, "slot B steps by one m x k matrix, in elements");
        assert_eq!(stride_c, (m * n) as i64, "the output steps by one m x n matrix, in elements");

        // The last slice's last element must still be inside each buffer. This is the assertion an
        // element-size scaling breaks, and only a batched fixture can fail it.
        let last = a.batch as i64 - 1;
        for (step, per_slice, whole, what) in [
            (stride_a, (k * n) as i64, (p * k * n) as i64, "slot A (geom.b)"),
            (stride_b, (m * k) as i64, (p * m * k) as i64, "slot B (geom.a)"),
            (stride_c, (m * n) as i64, (p * m * n) as i64, "output"),
        ] {
            assert!(last * step + per_slice <= whole, "{what}: the batch walk leaves its buffer");
        }
    }

    /// A broadcast batch axis is a zero stride, and a size-1 leading dimension is clamped up rather
    /// than passed as 0 (cuBLAS rejects ld = 0).
    #[test]
    fn degenerate_axes_are_clamped() {
        let (f, m, k, n) = (4usize, 5usize, 6usize, 7usize);
        let la = layout(&[m, k], &[k as isize, 1], 0);
        let lb = layout(&[f, k, n], &[(k * n) as isize, n as isize, 1], 0);
        let lc = layout(&[f, m, n], &[(m * n) as isize, n as isize, 1], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        let args = args(&g).unwrap();
        assert_eq!(args.batch, f as i32);
        assert_eq!(args.batch_a, 0, "a broadcast batch keeps stride 0");
        assert_eq!(args.batch_b, (k * n) as isize);
        assert_eq!(args.batch_c, (m * n) as isize);
        assert!(args.lda >= n as i32 && args.ldb >= k as i32 && args.ldc >= n as i32);
    }

    /// A broadcast along a **matmul** axis (stride 0, extent over 1) has no cuBLAS expression at
    /// all: declaring the clamped-up leading dimension would have cuBLAS read past the buffer.
    /// An extent-1 axis may still carry stride 0.
    #[test]
    fn broadcast_matmul_axis_is_refused() {
        let (m, k, n) = (3usize, 4usize, 5usize);
        let lc = layout(&[m, n], &[n as isize, 1], 0);
        // b broadcast along k: `sbk = 0`
        let la = layout(&[m, k], &[k as isize, 1], 0);
        let lb = layout(&[k, n], &[0, 1], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        assert!(args(&g).is_none(), "b broadcast along k is refused");
        // a broadcast along m: `sam = 0`
        let la = layout(&[m, k], &[0, 1], 0);
        let lb = layout(&[k, n], &[n as isize, 1], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        assert!(args(&g).is_none(), "a broadcast along m is refused");
        // an extent-1 axis may carry stride 0: the lifted `1 x k @ k x 1` dot
        let la = layout(&[1, k], &[0, 1], 0);
        let lb = layout(&[k, 1], &[1, k as isize], 0);
        let lc1 = layout(&[1, 1], &[1, 1], 0);
        let g = resolve_geom(RowMajor, &lc1, &la, &lb).unwrap();
        let args = args(&g).unwrap();
        assert_eq!(args.ldb, k as i32, "the extent-1 zero stride is clamped up");
    }
}
