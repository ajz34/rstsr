use crate::prelude_dev::*;
use core::ops::{Mul, Sub};

/// Cross-product kernel: for the (non-negative) `axis` of size 3, writes
///
/// ```text
/// c[.., 0, ..] = a[.., 1, ..] * b[.., 2, ..] - a[.., 2, ..] * b[.., 1, ..]
/// c[.., 1, ..] = a[.., 2, ..] * b[.., 0, ..] - a[.., 0, ..] * b[.., 2, ..]
/// c[.., 2, ..] = a[.., 0, ..] * b[.., 1, ..] - a[.., 1, ..] * b[.., 0, ..]
/// ```
///
/// with no conjugation, following the Array-API `linalg.cross` contract. The
/// three operands must already share one (broadcast) shape whose `axis` is 3.
pub fn cross_naive_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    axis: usize,
) -> Result<()>
where
    D: DimAPI,
    TA: Clone,
    TB: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Sub<TC, Output = TC>,
{
    let (sa, sb, sc) = (la.stride()[axis], lb.stride()[axis], lc.stride()[axis]);

    // Collapse the axis to a singleton so the dispatch walks each vector once;
    // the three components are then reached by the captured axis strides.
    let singleton = |layout: &Layout<D>| -> Result<Layout<D>> {
        let mut shape = layout.shape().clone();
        let mut stride = layout.stride().clone();
        shape[axis] = 1;
        stride[axis] = 0;
        Layout::new(shape, stride, layout.offset())
    };
    let lac = singleton(la)?;
    let lbc = singleton(lb)?;
    let lcc = singleton(lc)?;
    let layouts = translate_to_col_major(&[&lcc, &lac, &lbc], TensorIterOrder::K)?;
    let (lcc, lac, lbc) = (&layouts[0], &layouts[1], &layouts[2]);

    let at = |base: usize, s: isize, k: isize| (base as isize + k * s) as usize;
    layout_col_major_dim_dispatch_3(lcc, lac, lbc, |(ic, ia, ib)| {
        let (a0, a1, a2) = (at(ia, sa, 0), at(ia, sa, 1), at(ia, sa, 2));
        let (b0, b1, b2) = (at(ib, sb, 0), at(ib, sb, 1), at(ib, sb, 2));
        let (c0, c1, c2) = (at(ic, sc, 0), at(ic, sc, 1), at(ic, sc, 2));
        c[c0].write(a[a1].clone() * b[b2].clone() - a[a2].clone() * b[b1].clone());
        c[c1].write(a[a2].clone() * b[b0].clone() - a[a0].clone() * b[b2].clone());
        c[c2].write(a[a0].clone() * b[b1].clone() - a[a1].clone() * b[b0].clone());
    })
}
