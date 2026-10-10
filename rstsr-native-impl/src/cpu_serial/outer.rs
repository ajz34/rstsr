use crate::prelude_dev::*;
use core::ops::Mul;

/// Outer-product kernel: writes `c[i, j] = a[i] * b[j]` for two one-dimensional
/// operands of arbitrary strides into the fresh `(N, M)` output. No conjugation.
pub fn outer_naive_cpu_serial<TA, TB, TC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix1>,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TA: Mul<TB, Output = TC>,
{
    let (n, m) = (la.shape()[0], lb.shape()[0]);
    rstsr_assert_eq!(lc.shape(), &[n, m], InvalidLayout, "the outer-product output should have shape (a.len, b.len)")?;
    // widen each 1-d operand to (N, M) with a zero stride on the broadcast axis
    let lam = Layout::new([n, m], [la.stride()[0], 0], la.offset())?;
    let lbm = Layout::new([n, m], [0, lb.stride()[0]], lb.offset())?;
    // translate to the col-major access order used by the dispatch helper
    let layouts = translate_to_col_major(&[lc, &lam, &lbm], TensorIterOrder::K)?;
    let (lc, lam, lbm) = (&layouts[0], &layouts[1], &layouts[2]);
    layout_col_major_dim_dispatch_3(lc, lam, lbm, |(idx_c, idx_a, idx_b)| {
        c[idx_c].write(a[idx_a].clone() * b[idx_b].clone());
    })
}
