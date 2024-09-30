use crate::device_cpu_rayon::device::DeviceCpuRayon;
use crate::device_cpu_serial::operators::op_with_func::*;
use crate::prelude_dev::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
const PARALLEL_SWITCH: usize = 256;
// Currently, we do not make contiguous parts to be parallel. Only outer
// iteration is parallelized.

#[allow(clippy::too_many_arguments)]
pub fn op_mutc_refa_refb_func_cpu_rayon<TA, TB, TC, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: impl Fn(&mut TC, &TA, &TB) + Sync + Send,
    nthreads: usize,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH * nthreads {
        return op_mutc_refa_refb_func_cpu_serial(c, lc, a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    let pool = DeviceCpuRayon::new(nthreads).get_pool(nthreads)?;
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous assign
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
        pool.install(|| {
            (iter_c, iter_a, iter_b).into_par_iter().for_each(|(idx_c, idx_a, idx_b)| unsafe {
                let c_ptr = c.as_ptr() as *mut TC;
                f(&mut *c_ptr.add(idx_c), &a[idx_a], &b[idx_b]);
            });
        });
    } else {
        // parallel for outer iteration
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
        let iter_b = IterLayoutColMajor::new(&layouts_contig[2])?;
        pool.install(|| {
            (iter_c, iter_a, iter_b).into_par_iter().for_each(|(idx_c, idx_a, idx_b)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b[idx_b + idx]);
                });
            });
        });
    }
    return Ok(());
}


