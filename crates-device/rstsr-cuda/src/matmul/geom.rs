//! Geometry of one matmul problem, resolved from the layouts alone.
//!
//! [`resolve_geom`] is pure (no device, no allocation, no I/O), because this is the part that can
//! be wrong in a way a GPU run reports only as "wrong numbers": the col-major reversal, the batch
//! parts, which buffer plays which role. All of it is host-side arithmetic on strides, so it is
//! tested here rather than on hardware.
//!
//! Nothing about a *driver's* restrictions lives here. The geometry describes every layout the
//! rule table admits; each driver decides what it can express ([`super::drivers`]) and declines.

use crate::prelude_dev::*;

/// The batch part of one operand: per-axis extents and strides, in the row-major order of the
/// rule table. Empty when the operand has no batch part.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchPart {
    pub shape: Vec<usize>,
    pub stride: Vec<isize>,
}

impl BatchPart {
    fn of(l: Option<&Layout<IxD>>) -> Self {
        let Some(l) = l else { return Self::default() };
        Self { shape: l.shape().to_vec(), stride: l.stride().to_vec() }
    }

    /// Number of batch slices (1 when there is no batch part).
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Element step of one batch slice, if the batch part is a single affine run.
///
/// Size-1 axes are dropped first, because they contribute no offset however their stride is
/// written: `[4, 1]` with strides `[3, 0]` steps by 3, and `[4, 1]` with `[3, 7]` steps by 3 as
/// well. What survives must then be a row-major progression, so `offset = index * step` for every
/// index in `0..size`. Broadcast axes (`stride == 0`) are affine with step 0, which covers
/// `A[i,j] @ B[p,j,k]` where the batch axis belongs to `B` alone.
///
/// `None` means a genuinely strided multi-axis batch (`[4, 3]` with strides `[6, 1]`, whose offsets
/// are not one step); only a driver that walks the batch axes one by one can take those.
pub(crate) fn linear_step(part: &BatchPart) -> Option<isize> {
    let mut shape = Vec::with_capacity(part.shape.len());
    let mut stride = Vec::with_capacity(part.stride.len());
    for k in 0..part.shape.len() {
        if part.shape[k] != 1 {
            shape.push(part.shape[k]);
            stride.push(part.stride[k]);
        }
    }
    if shape.is_empty() {
        return Some(0);
    }
    let n = shape.len();
    let step = stride[n - 1];
    let mut expect = step;
    for k in (0..n).rev() {
        if stride[k] != expect {
            return None;
        }
        expect *= shape[k] as isize;
    }
    Some(step)
}

/// Everything a driver needs to address the three buffers, resolved from the layouts alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GemmGeom {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub n_batch: usize,
    /// `(element offset, stride of the m axis, stride of the k axis, batch part)` for the left
    /// operand of the *row-major* problem, which under `ColMajor` is the caller's `b`.
    pub a: (usize, isize, isize),
    /// The same for the right operand: `(offset, stride of the k axis, stride of the n axis)`.
    pub b: (usize, isize, isize),
    /// `(offset, stride of the m axis, stride of the n axis)` for the output.
    pub c: (usize, isize, isize),
    /// The batch parts of `a`, `b`, `c` in the *geom* roles.
    pub batch: [BatchPart; 3],
    /// True when the geometry came from the col-major reversal, i.e. `a` and `b` are the caller's
    /// second and first operands respectively.
    pub swapped: bool,
}

/// The geometry in the caller's roles: `c[i, j] = sum_k a[i, k] * b[k, j]`, whatever the order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CallerGeom<'a> {
    /// Extents `(m, k, n)` of the caller's problem.
    pub mnk: (usize, usize, usize),
    /// `(offset, stride of the m axis, stride of the k axis)` of the caller's `a`.
    pub a: (usize, isize, isize),
    /// `(offset, stride of the k axis, stride of the n axis)` of the caller's `b`.
    pub b: (usize, isize, isize),
    /// `(offset, stride of the m axis, stride of the n axis)` of the output.
    pub c: (usize, isize, isize),
    /// The batch parts of the caller's `a`, `b` and the output.
    pub batch: [&'a BatchPart; 3],
}

impl GemmGeom {
    /// The single element step of each batch part, or `None` if any is not one linear run.
    pub fn affine_batch(&self) -> Option<[isize; 3]> {
        Some([linear_step(&self.batch[0])?, linear_step(&self.batch[1])?, linear_step(&self.batch[2])?])
    }

    /// The same addresses, named by the caller's operands instead of the row-major roles.
    ///
    /// The col-major reversal only relabels axes: `geom`'s `(i, kk)` of its `a` is the caller's
    /// `b[kk, i]`, and so on, so no address changes here, only which stride belongs to which name.
    pub fn caller(&self) -> CallerGeom<'_> {
        if self.swapped {
            CallerGeom {
                mnk: (self.n, self.k, self.m),
                a: (self.b.0, self.b.2, self.b.1),
                b: (self.a.0, self.a.2, self.a.1),
                c: (self.c.0, self.c.2, self.c.1),
                batch: [&self.batch[1], &self.batch[0], &self.batch[2]],
            }
        } else {
            CallerGeom {
                mnk: (self.m, self.k, self.n),
                a: self.a,
                b: self.b,
                c: self.c,
                batch: [&self.batch[0], &self.batch[1], &self.batch[2]],
            }
        }
    }
}

/// Resolve the geometry of `c = a @ b` on the layout the tier already validated. Pure.
pub(crate) fn resolve_geom(order: FlagOrder, lc: &Layout<IxD>, la: &Layout<IxD>, lb: &Layout<IxD>) -> Result<GemmGeom> {
    // Col-major: reverse every layout **and swap the operands**, so the row-major decomposition of
    // the problem is the one the rule table knows. That is the identity
    // `rstsr_common::layout::matmul` documents, `C[n,m,t] = B[n,k,t] @ A[k,m,t]`, not a
    // device-specific convention.
    //
    // The swap has to happen here, when asking for the config: reversing alone would feed the rule
    // table `(A_rev, B_rev)`, whose contracted dimensions are `m_a` and `n_b` — a shape mismatch
    // for anything rectangular, and a silently transposed answer for anything square.
    let swapped = matches!(order, ColMajor);
    let (la_c, lb_c, lc_c) = if swapped {
        (lb.reverse_axes(), la.reverse_axes(), lc.reverse_axes())
    } else {
        (la.clone(), lb.clone(), lc.clone())
    };

    let cfg = layout_matmul_dyn_row_major_with_lc(&la_c, &lb_c, &lc_c)?;

    // The inner-dot rule leaves `la_matmul` rank 1. That rule is a reduction, not a GEMM: no
    // driver here is the reduce kernel, so it is refused by name.
    if cfg.la_matmul.ndim() != 2 || cfg.lb_matmul.ndim() != 2 {
        rstsr_raise!(
            UnImplemented,
            "cuda matmul: this is a single-axis contraction (inner dot), not a GEMM; it belongs on \
             the reduce kernel and is not wired to it yet"
        )?
    }

    let (lay_a, lay_b) = (&cfg.la_matmul, &cfg.lb_matmul);
    let (m, k) = (lay_a.shape()[0], lay_a.shape()[1]);
    let (k2, n) = (lay_b.shape()[0], lay_b.shape()[1]);
    rstsr_assert_eq!(k, k2, InvalidLayout, "the contracted dimensions of a and b should be the same")?;
    rstsr_assert_eq!(
        cfg.lc_matmul.shape(),
        &[m, n][..],
        InvalidLayout,
        "the matmul output should have shape (a.shape[0], b.shape[1])"
    )?;

    let batch =
        [BatchPart::of(cfg.la_rest.as_ref()), BatchPart::of(cfg.lb_rest.as_ref()), BatchPart::of(cfg.lc_rest.as_ref())];

    let sa = (lay_a.stride()[0], lay_a.stride()[1]);
    let sb = (lay_b.stride()[0], lay_b.stride()[1]);
    let sc = (cfg.lc_matmul.stride()[0], cfg.lc_matmul.stride()[1]);

    Ok(GemmGeom {
        m,
        n,
        k,
        n_batch: batch[2].size(),
        a: (lay_a.offset(), sa.0, sa.1),
        b: (lay_b.offset(), sb.0, sb.1),
        c: (cfg.lc_matmul.offset(), sc.0, sc.1),
        batch,
        swapped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], stride: &[isize]) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), 0).unwrap()
    }

    fn part(shape: &[usize], stride: &[isize]) -> BatchPart {
        BatchPart { shape: shape.to_vec(), stride: stride.to_vec() }
    }

    /// A contiguous batch, a broadcast batch and a rank-1 batch are all affine.
    #[test]
    fn affine_batches_are_linear() {
        fn lin(shape: &[usize], stride: &[isize]) -> Option<isize> {
            linear_step(&part(shape, stride))
        }
        assert_eq!(lin(&[4], &[64]), Some(64));
        assert_eq!(lin(&[4, 3], &[3, 1]), Some(1), "contiguous = step 1 over the flat index");
        assert_eq!(lin(&[4], &[0]), Some(0), "broadcast");
        assert_eq!(lin(&[4, 1], &[3, 0]), Some(3), "size-1 axes contribute nothing");
        assert_eq!(lin(&[4, 3], &[0, 0]), Some(0), "all-broadcast");
        assert_eq!(lin(&[], &[]), Some(0), "no batch at all");
    }

    /// A batch that walks two axes with a hole between them is not one step, and is refused.
    #[test]
    fn gapped_batches_are_refused() {
        assert_eq!(linear_step(&part(&[4, 3], &[6, 1])), None);
        assert_eq!(linear_step(&part(&[4, 3], &[9, 1])), None);
    }

    /// The offset a layout assigns to one multi-index. This is the tensor layer's own mapping, so
    /// comparing the geom against it checks *which layout plays which role*, not the arithmetic of
    /// `offset + i*stride` twice over.
    fn off(l: &Layout<IxD>, idx: &[usize]) -> isize {
        let mut o = l.offset() as isize;
        for (k, &i) in idx.iter().enumerate() {
            o += i as isize * l.stride()[k];
        }
        o
    }

    /// Row-major: `geom.a` walks `a[i, k]`, `geom.b` walks `b[k, j]`, `geom.c` walks `c[i, j]`.
    #[test]
    fn row_major_geom_matches_the_layouts() {
        let (m, k, n) = (2usize, 3usize, 4usize);
        let la = layout(&[m, k], &[k as isize, 1]);
        let lb = layout(&[k, n], &[n as isize, 1]);
        let lc = layout(&[m, n], &[n as isize, 1]);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();

        assert_eq!((g.m, g.n, g.k, g.n_batch), (m, n, k, 1));
        assert!(!g.swapped);

        for i in 0..m {
            for kk in 0..k {
                assert_eq!(g.a.0 as isize + i as isize * g.a.1 + kk as isize * g.a.2, off(&la, &[i, kk]));
            }
            for j in 0..n {
                assert_eq!(g.c.0 as isize + i as isize * g.c.1 + j as isize * g.c.2, off(&lc, &[i, j]));
            }
        }
        for kk in 0..k {
            for j in 0..n {
                assert_eq!(g.b.0 as isize + kk as isize * g.b.1 + j as isize * g.b.2, off(&lb, &[kk, j]));
            }
        }
        // without a batch part every step is 0 and there is one slice
        assert!(g.batch.iter().all(|p| p.shape.is_empty()));
        assert_eq!(g.affine_batch(), Some([0, 0, 0]));
    }

    /// Column-major: the roles swap and every index is read in the other order. This is the case a
    /// GPU run would report only as "wrong numbers", so it is pinned by walking each buffer and
    /// comparing against the tensor layout's own offset.
    #[test]
    fn col_major_geom_swaps_roles_and_index_order() {
        let (m, k, n) = (2usize, 3usize, 4usize);
        // column-major = first axis contiguous
        let la = layout(&[m, k], &[1, m as isize]);
        let lb = layout(&[k, n], &[1, k as isize]);
        let lc = layout(&[m, n], &[1, m as isize]);
        let g = resolve_geom(ColMajor, &lc, &la, &lb).unwrap();

        assert!(g.swapped, "col-major must then read the operands the other way round");
        assert_eq!((g.m, g.n, g.k), (n, m, k), "the problem is solved transposed");
        assert_eq!(g.m * g.n, m * n, "but it is still the same output matrix");

        // geom.a is the caller's b, walked as b[k, n]: geom's (i, kk) is b's (kk, i)
        for i in 0..g.m {
            for kk in 0..g.k {
                assert_eq!(g.a.0 as isize + i as isize * g.a.1 + kk as isize * g.a.2, off(&lb, &[kk, i]));
            }
        }
        // geom.b is the caller's a, walked as a[m, k]: geom's (kk, j) is a's (j, kk)
        for kk in 0..g.k {
            for j in 0..g.n {
                assert_eq!(g.b.0 as isize + kk as isize * g.b.1 + j as isize * g.b.2, off(&la, &[j, kk]));
            }
        }
        // geom.c is c[n, m]: geom's (i, j) is c's (j, i)
        for i in 0..g.m {
            for j in 0..g.n {
                assert_eq!(g.c.0 as isize + i as isize * g.c.1 + j as isize * g.c.2, off(&lc, &[j, i]));
            }
        }
    }

    /// A batched problem: the batch step must be the stride of one whole batched matrix, and a
    /// broadcast operand must have step 0.
    #[test]
    fn batch_step_is_one_matrix() {
        let (p, m, k, n) = (4usize, 5usize, 6usize, 7usize);
        let la = layout(&[p, m, k], &[(m * k) as isize, k as isize, 1]);
        let lb = layout(&[p, k, n], &[(k * n) as isize, n as isize, 1]);
        let lc = layout(&[p, m, n], &[(m * n) as isize, n as isize, 1]);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        assert_eq!(g.n_batch, p);
        assert_eq!(g.batch.iter().map(|b| b.shape.clone()).collect::<Vec<_>>(), vec![vec![p]; 3]);
        assert_eq!(g.affine_batch(), Some([(m * k) as isize, (k * n) as isize, (m * n) as isize]));

        // left operand broadcast over the batch: its batch part is absent, so its step is 0
        let la1 = layout(&[m, k], &[k as isize, 1]);
        let g1 = resolve_geom(RowMajor, &lc, &la1, &lb).unwrap();
        assert_eq!(g1.n_batch, p);
        assert_eq!(g1.batch[0].shape, vec![p], "a's batch part is broadcast to the output's");
        assert_eq!(g1.batch[0].stride, vec![0]);
        assert_eq!(g1.affine_batch(), Some([0, (k * n) as isize, (m * n) as isize]));
    }

    /// The inner-dot rule has no 2-D matmul part, and is refused rather than run as a 1x1 tile.
    #[test]
    fn inner_dot_has_no_geometry() {
        let la = layout(&[6], &[1]);
        let lc = layout(&[], &[]);
        let e = resolve_geom(RowMajor, &lc, &la, &la).unwrap_err();
        assert!(format!("{e}").contains("single-axis contraction"), "got: {e}");
    }

    /// The caller view undoes the col-major relabelling: every `(i, k)`, `(k, j)` and `(i, j)` of
    /// the caller names the same element the caller's own layout does, in both orders.
    #[test]
    fn caller_view_addresses_the_callers_elements() {
        let (p, m, k, n) = (3usize, 5usize, 6usize, 7usize);
        for order in [RowMajor, ColMajor] {
            // Contiguous in the order under test, with the batch axis where that order puts it:
            // trailing under row-major, leading under col-major (the matrix axes are the two the
            // order names first).
            // row-major strides: the running product of the extents to the right
            let contig = |shape: &[usize]| {
                let mut run: isize = shape.iter().map(|&d| d as isize).product();
                let stride: Vec<isize> = shape
                    .iter()
                    .map(|&d| {
                        run /= d as isize;
                        run
                    })
                    .collect();
                layout(shape, &stride)
            };
            let f = |d0: usize, d1: usize, d2: usize| layout(&[d0, d1, d2], &[1, d0 as isize, (d0 * d1) as isize]);
            let (la, lb, lc) = if matches!(order, RowMajor) {
                (contig(&[p, m, k]), contig(&[p, k, n]), contig(&[p, m, n]))
            } else {
                (f(m, k, p), f(k, n, p), f(m, n, p))
            };
            let g = resolve_geom(order, &lc, &la, &lb).unwrap();
            let v = g.caller();
            assert_eq!(g.swapped, matches!(order, ColMajor));
            assert_eq!(v.batch[0].shape, vec![p], "the batch parts follow the caller's names");
            assert_eq!(v.mnk, (m, k, n), "and so do the extents");
            // the caller's multi-index of `(x, y)` of an operand: batch axis last or first
            let idx = |x: usize, y: usize| {
                if matches!(order, RowMajor) {
                    vec![0, x, y]
                } else {
                    vec![x, y, 0]
                }
            };
            for i in 0..m {
                for kk in 0..k {
                    assert_eq!(
                        v.a.0 as isize + i as isize * v.a.1 + kk as isize * v.a.2,
                        off(&la, &idx(i, kk)),
                        "{order:?}: a[{i},{kk}]"
                    );
                }
                for j in 0..n {
                    assert_eq!(
                        v.c.0 as isize + i as isize * v.c.1 + j as isize * v.c.2,
                        off(&lc, &idx(i, j)),
                        "{order:?}: c[{i},{j}]"
                    );
                }
            }
            for kk in 0..k {
                for j in 0..n {
                    assert_eq!(
                        v.b.0 as isize + kk as isize * v.b.1 + j as isize * v.b.2,
                        off(&lb, &idx(kk, j)),
                        "{order:?}: b[{kk},{j}]"
                    );
                }
            }
        }
    }
}
