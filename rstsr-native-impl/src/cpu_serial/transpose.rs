//! Naive implementation of matrix transpose

use crate::prelude_dev::*;

const BLOCK_SIZE: usize = 64;

/// Change order (row/col-major) a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (row-major) to `c` (col-major).
/// If shape or stride is not compatible, an error will be returned.
pub fn orderchange_out_r2c_ix2_cpu_serial<T>(c: &mut [T], lc: &Layout<Ix2>, a: &[T], la: &Layout<Ix2>) -> Result<()>
where
    T: Clone,
{
    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout, "This function requires shape identity")?;
    rstsr_assert_eq!(sc[1], sa[1], InvalidLayout, "This function requires shape identity")?;
    let [nrow, ncol] = *sa;

    // stride check
    rstsr_assert_eq!(lc.stride()[0], 1, InvalidLayout, "This function requires col-major output")?;
    rstsr_assert_eq!(la.stride()[1], 1, InvalidLayout, "This function requires row-major input")?;

    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    let lda = la.stride()[0];
    let ldc = lc.stride()[1];

    (0..ncol).step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        let (j_start, j_end) = (j_start as isize, j_end as isize);
        (0..nrow).step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            let (i_start, i_end) = (i_start as isize, i_end as isize);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i * lda + j) as usize;
                    let dst_idx = (offset_c + j * ldc + i) as usize;
                    c[dst_idx] = a[src_idx].clone();
                }
            }
        });
    });

    Ok(())
}

/// Promote-general variants of [`orderchange_out_r2c_ix2_cpu_serial`]: same
/// blocked traversal, but the inner write is `clone().into_cast()` (identity
/// inlined when `TC == TA`, so same-dtype callers pay nothing — the same
/// contract as the promote assign families). The uninit variant writes
/// `MaybeUninit<TC>` slots; this is sound **because the blocked loop covers
/// every output element (i, j) exactly once** (full-coverage by loop
/// construction: `i in 0..nrow`, `j in 0..ncol`, one write per pair).
///
/// All index arithmetic is `isize` (negative slow-axis strides are valid,
/// e.g. flip views); the `usize` cast happens only after the full offset sum,
/// and slice indexing keeps per-access bounds checks.
#[duplicate_item(
    func_name
        TypeC TypeA func_write
    ;
    [orderchange_out_r2c_ix2_promote_cpu_serial]
        [TC] [TA]
        [*ci = ai.clone().into_cast()]
    ;
    [orderchange_out_r2c_ix2_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA]
        [ci.write(ai.clone().into_cast())]
    ;
)]
pub fn func_name<TC, TA>(c: &mut [TypeC], lc: &Layout<Ix2>, a: &[TypeA], la: &Layout<Ix2>) -> Result<()>
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
{
    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout, "This function requires shape identity")?;
    rstsr_assert_eq!(sc[1], sa[1], InvalidLayout, "This function requires shape identity")?;
    let [nrow, ncol] = *sa;

    // stride check
    rstsr_assert_eq!(lc.stride()[0], 1, InvalidLayout, "This function requires col-major output")?;
    rstsr_assert_eq!(la.stride()[1], 1, InvalidLayout, "This function requires row-major input")?;

    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    let lda = la.stride()[0];
    let ldc = lc.stride()[1];

    (0..ncol).step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        let (j_start, j_end) = (j_start as isize, j_end as isize);
        (0..nrow).step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            let (i_start, i_end) = (i_start as isize, i_end as isize);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i * lda + j) as usize;
                    let dst_idx = (offset_c + j * ldc + i) as usize;
                    let ai = &a[src_idx];
                    let ci = &mut c[dst_idx];
                    func_write;
                }
            }
        });
    });

    Ok(())
}

/// Change order (row/col-major) a matrix out-place, promote/uninit variants.
///
/// Col-major-to-row-major wrappers of the `r2c` kernels above: reverse both
/// layouts' axes and delegate. See the r2c functions for the write-once /
/// isize-offset contract.
#[duplicate_item(
    func_name
        TypeC TypeA func_r2c
    ;
    [orderchange_out_c2r_ix2_promote_cpu_serial]
        [TC] [TA]
        [orderchange_out_r2c_ix2_promote_cpu_serial]
    ;
    [orderchange_out_c2r_ix2_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_serial]
    ;
)]
pub fn func_name<TC, TA>(c: &mut [TypeC], lc: &Layout<Ix2>, a: &[TypeA], la: &Layout<Ix2>) -> Result<()>
where
    TC: Clone,
    TA: Clone + DTypeCastAPI<TC>,
{
    let lc = lc.reverse_axes();
    let la = la.reverse_axes();
    func_r2c(c, &lc, a, &la)
}

/// Change order (row/col-major) a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (col-major) to `c` (row-major).
/// If shape or stride is not compatible, an error will be returned.
pub fn orderchange_out_c2r_ix2_cpu_serial<T>(c: &mut [T], lc: &Layout<Ix2>, a: &[T], la: &Layout<Ix2>) -> Result<()>
where
    T: Clone,
{
    let lc = lc.reverse_axes();
    let la = la.reverse_axes();
    orderchange_out_r2c_ix2_cpu_serial(c, &lc, a, &la)
}
