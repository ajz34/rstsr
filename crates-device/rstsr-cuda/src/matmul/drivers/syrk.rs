//! The symmetric form of a fresh product: `c = alpha * a @ a^T` as one `?syrk_v2` call.
//!
//! cuBLAS writes one triangle of the symmetric output. The other is a mirror image of the first,
//! copied across the diagonal by a small kernel; that copy is what makes the form a win, since a
//! second `?syrk` call costs as much as the gemm it replaces.
//!
//! The condition is the one the BLAS devices apply in their matmul: one type throughout, `beta`
//! zero, `b` the same buffer as `a` with reversed strides, unbatched ([`CallerGeom::is_syrk`]).
//! Which output types take the form is [`MatMulUseSyrk`][crate::matmul::MatMulUseSyrk]; there is no
//! size gate.

use cudarc::cublas::sys;

use super::cublas::{op, CublasScalar};
use super::{scale, AlignedScale, CublasPlan, Problem, Refusal, Verdict};
use crate::dtype::CudaDType;
use crate::kernel::{order_input, order_output, FlatLaunch};
use crate::kernels::elementwise::planned_function;
use crate::matmul::geom::CallerGeom;
use crate::prelude_dev::*;

/// One `?syrk_v2` call, in cuBLAS's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SyrkArgs {
    /// Whether `a` is stored as `k x n` and transposed on entry.
    pub trans: bool,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldc: i32,
    /// The triangle cuBLAS leaves written, in the caller's `(row, column)` indexing of `c`: the
    /// rows at or below the diagonal when the output is read transposed.
    pub lower: bool,
}

/// Derive the call, or `None` when a layout has no column-major expression.
///
/// Pure. The output may be presented to cuBLAS transposed (`ldc` = the row stride): the product
/// is symmetric, so that only swaps which triangle is written, which [`SyrkArgs::lower`] records.
pub(crate) fn args(v: &CallerGeom<'_>) -> Option<SyrkArgs> {
    let (n, k) = (v.mnk.0, v.mnk.1);
    let (_, sam, sak) = v.a;
    let (_, scm, scn) = v.c;
    if n > i32::MAX as usize || k > i32::MAX as usize || sam < 0 || sak < 0 || scm < 0 || scn < 0 {
        return None;
    }
    // a broadcast along a matmul axis (a zero stride with extent over 1) has no expression either
    for (stride, extent) in [(sam, n), (sak, k), (scm, n), (scn, n)] {
        if stride == 0 && extent > 1 {
            return None;
        }
    }
    let ld = |stride: isize, rows: usize| (stride.max(0).max(rows as isize)).min(i32::MAX as isize) as i32;
    let (trans, lda) = if sam == 1 {
        (false, ld(sak, n))
    } else if sak == 1 {
        (true, ld(sam, k))
    } else {
        return None;
    };
    let (ldc, lower) = if scm == 1 {
        (ld(scn, n), false)
    } else if scn == 1 {
        (ld(scm, n), true)
    } else {
        return None;
    };
    Some(SyrkArgs { trans, n: n as i32, k: k as i32, lda, ldc, lower })
}

/// One attempt of the symmetric form, for the four types the typed entry covers.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>, plan: CublasPlan) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    match plan {
        CublasPlan::F32(_) => typed::<f32, TA, TB, TC, TBuf>(p),
        CublasPlan::F64(_) => typed::<f64, TA, TB, TC, TBuf>(p),
        CublasPlan::C32(_) => typed::<num::Complex<f32>, TA, TB, TC, TBuf>(p),
        CublasPlan::C64(_) => typed::<num::Complex<f64>, TA, TB, TC, TBuf>(p),
        _ => Ok(Verdict::Declined(Refusal::Cannot("the typed syrk entries cover f32, f64 and complex only"))),
    }
}

fn typed<T: CublasScalar, TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    let view = p.geom.caller();
    let Some(a) = args(&view) else {
        return Ok(Verdict::Declined(Refusal::Cannot("no column-major expression of the symmetric product")));
    };
    let Some(u) = p.uniform::<T>() else {
        return Ok(Verdict::Declined(Refusal::Cannot("the operands and the output must share one dtype")));
    };
    // `beta` is zero here (the form requires it): passed for the signature, never applied.
    let (Some(alpha), Some(beta)) = (scale::<T, TC>(p.alpha), scale::<T, TC>(p.beta)) else {
        return Ok(Verdict::Declined(Refusal::Cannot("the output type is not of the plan's family")));
    };
    let (alpha, beta) = (AlignedScale(alpha), AlignedScale(beta));
    let exec = p.device.stream();
    let Some(blas) = exec.cublas() else {
        return Ok(Verdict::Declined(Refusal::NoCublas));
    };
    // the output is written in place (and never read: `beta` is zero): order it in before the
    // enqueues and out after them
    let orders = [order_input(u.a, exec)?, order_input(u.c, exec)?];
    let (pa, ra) = u.a.read_ptr_on(exec);
    let (pc, rc) = u.c.read_ptr_on(exec);
    let esz = core::mem::size_of::<T>();
    let (ap, cp) = (pa as usize + view.a.0 * esz, pc as usize + view.c.0 * esz);
    let status = {
        let _api = blas.lock();
        // SAFETY: the pointers are the buffers' bases folded with the layouts' own offsets, and
        // `args` was derived from the same view.
        unsafe {
            T::syrk(
                *blas.blas().handle(),
                sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                op(a.trans),
                a.n,
                a.k,
                &alpha.0,
                ap as u64,
                a.lda,
                &beta.0,
                cp as u64,
                a.ldc,
            )
        }
    };
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Ok(Verdict::Declined(Refusal::CublasStatus(status)));
    }
    let mirrored = mirror::<T>(p.device, &u, view, a.lower);
    // the syrk and the mirror are enqueued by now (a syrk refusal above enqueues nothing, a
    // mirror failure at most the syrk): order the output's allocation stream behind that
    let written = order_output(u.c, exec)?;
    drop((orders, ra, rc, written));
    mirrored?;
    Ok(Verdict::Ran)
}

const MIRROR_ENTRY: &str = r#"
extern "C" __global__ void rstsr_matmul_mirror(
    @T@* c, rstsr_isz scm, rstsr_isz scn, rstsr_usz total, rstsr_usz n, int lower)
{
    rstsr_usz t = (rstsr_usz)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    rstsr_usz r = t / n;
    rstsr_usz j = t - r * n;
    if (lower ? (r < j) : (r > j)) c[r * scm + j * scn] = c[j * scm + r * scn];
}
"#;

/// Copy the written triangle of `c` across the diagonal into the other one.
///
/// Runs on the executing stream right after the syrk it follows, so `c` needs no ordering of its
/// own here: the caller orders it before and after the enqueues.
fn mirror<T: CudaDType>(
    device: &DeviceCuda,
    u: &super::Buffers<'_, T, T, T>,
    view: CallerGeom<'_>,
    lower: bool,
) -> Result<()> {
    let (_, scm, scn) = view.c;
    let n = view.mnk.0;
    let total = n * n;
    if total == 0 {
        return Ok(());
    }
    let f = planned_function(device, "rstsr_matmul_mirror", T::INCLUDES, T::CNAME, || {
        MIRROR_ENTRY.replace("@T@", T::CNAME)
    })?;
    let exec = device.stream();
    FlatLaunch::new(exec, &f, total)
        .arg_ptr(u.c, view.c.0)
        .arg_bytes(&scm.to_ne_bytes())
        .arg_bytes(&scn.to_ne_bytes())
        .arg_usize(total)
        .arg_usize(n)
        .arg_i32(lower as i32)
        .launch()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::geom::{resolve_geom, GemmGeom};

    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), offset).unwrap()
    }

    fn geom(order: FlagOrder, sa: (isize, isize), sc: (isize, isize)) -> GemmGeom {
        let (n, k) = (5usize, 8usize);
        // an offset that keeps a reversed layout in range as well
        let la = layout(&[n, k], &[sa.0, sa.1], n * k - 1);
        let lc = layout(&[n, n], &[sc.0, sc.1], 7);
        resolve_geom(order, &lc, &la, &la.reverse_axes()).unwrap()
    }

    /// Row-major `a` is a transposed `k x n` matrix to cuBLAS, with `lda` the row stride; a
    /// column-major one is the `n x k` matrix itself. A row-major output is presented transposed,
    /// so cuBLAS's upper triangle is the output's lower one; a column-major output is read as is.
    #[test]
    fn args_follow_the_operand_orientations() {
        let row_major = args(&geom(RowMajor, (8, 1), (5, 1)).caller()).unwrap();
        assert_eq!(row_major, SyrkArgs { trans: true, n: 5, k: 8, lda: 8, ldc: 5, lower: true });
        let col_major = args(&geom(ColMajor, (1, 5), (1, 5)).caller()).unwrap();
        assert_eq!(col_major, SyrkArgs { trans: false, n: 5, k: 8, lda: 5, ldc: 5, lower: false });
    }

    /// A stepped operand or output has no column-major expression.
    #[test]
    fn stepped_layouts_are_refused() {
        assert!(args(&geom(RowMajor, (16, 2), (5, 1)).caller()).is_none());
        assert!(args(&geom(RowMajor, (8, 1), (10, 2)).caller()).is_none());
        assert!(args(&geom(RowMajor, (-1, -5), (5, 1)).caller()).is_none());
    }

    /// A broadcast along a matmul axis (a zero stride with extent over 1) has no syrk expression
    /// either; an extent-1 axis may still carry stride 0.
    #[test]
    fn broadcast_matmul_axis_is_refused() {
        // a column broadcast along k: `sak = 0` with k over 1
        let (n, k) = (5usize, 8usize);
        let la = layout(&[n, k], &[1, 0], n * k - 1);
        let lc = layout(&[n, n], &[n as isize, 1], 7);
        let g = resolve_geom(RowMajor, &lc, &la, &la.reverse_axes()).unwrap();
        assert!(args(&g.caller()).is_none(), "a broadcast along k is refused");
        // an extent-1 axis may carry stride 0 (k = 1)
        let la = layout(&[n, 1], &[1, 0], n - 1);
        let lc = layout(&[n, n], &[n as isize, 1], 7);
        let g = resolve_geom(RowMajor, &lc, &la, &la.reverse_axes()).unwrap();
        assert!(args(&g.caller()).is_some(), "a k = 1 zero stride is served");
    }
}
