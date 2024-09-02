use crate::prelude_dev::*;

/// Compare two tensors with f64 data type.
///
/// This function assumes c-contiguous iteration, and will not check two
/// dimensions are broadcastable.
pub(crate) fn allclose_f64<RA, RB, DA, DB>(a: &TensorBase<RA, DA>, b: &TensorBase<RB, DB>) -> bool
where
    RA: DataAPI<Data = Storage<f64, CpuDevice>>,
    RB: DataAPI<Data = Storage<f64, CpuDevice>>,
    DA: DimAPI,
    DB: DimAPI,
{
    let la = a.layout().reverse_axes();
    let lb = b.layout().reverse_axes();
    let it_la = IterLayoutColMajor::new(&la).unwrap();
    let it_lb = IterLayoutColMajor::new(&lb).unwrap();
    let data_a = a.data().storage().rawvec();
    let data_b = b.data().storage().rawvec();
    let atol = 1e-8;
    let rtol = 1e-5;
    for (idx_a, idx_b) in izip!(it_la, it_lb) {
        let va = data_a[idx_a];
        let vb = data_b[idx_b];
        let comp = (va - vb).abs() <= atol + rtol * vb.abs();
        if !comp {
            return false;
        }
    }
    return true;
}
