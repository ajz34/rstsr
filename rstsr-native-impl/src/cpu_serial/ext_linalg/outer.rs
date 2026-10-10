use crate::prelude_dev::*;
use core::ops::Mul;
use rstsr_dtype_traits::DTypePromoteAPI;

/// Promoting twin of [`outer_naive_cpu_serial`]: `a` and `b` may have different
/// dtypes, each pair promoted to `TC` (= `TA::Res`) before the product.
pub fn outer_ext_naive_cpu_serial<TA, TB, TC>(
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
    TC: Clone + Mul<TC, Output = TC>,
    TA: DTypePromoteAPI<TB, Res = TC>,
{
    let (n, m) = (la.shape()[0], lb.shape()[0]);
    rstsr_assert_eq!(lc.shape(), &[n, m], InvalidLayout, "the outer-product output should have shape (a.len, b.len)")?;
    let lam = Layout::new([n, m], [la.stride()[0], 0], la.offset())?;
    let lbm = Layout::new([n, m], [0, lb.stride()[0]], lb.offset())?;
    let layouts = translate_to_col_major(&[lc, &lam, &lbm], TensorIterOrder::K)?;
    let (lc, lam, lbm) = (&layouts[0], &layouts[1], &layouts[2]);
    layout_col_major_dim_dispatch_3(lc, lam, lbm, |(idx_c, idx_a, idx_b)| {
        let (x, y) = a[idx_a].clone().promote_pair(b[idx_b].clone());
        c[idx_c].write(x * y);
    })
}
