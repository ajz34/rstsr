use super::*;
use rstsr_blas_traits::lapack_solve::*;

#[test]
fn test_dgetrf_dgetri() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default
    let driver = DGETRF::default().a(a.view()).build().unwrap();
    let (lu, piv) = driver.run().unwrap();
    let lu = lu.into_owned();
    let fpiv = piv.map(|&v| v as f64);
    assert!((fingerprint(&lu) - 5397.198541468395).abs() < 1e-8);
    assert!((fingerprint(&fpiv) - -14.694714160751573).abs() < 1e-8);

    let driver = DGETRI::default().a(lu.view()).ipiv(piv.view()).build().unwrap();
    let inv_a = driver.run().unwrap();
    let inv_a = inv_a.into_owned();
    assert!((fingerprint(&inv_a) - 143.3900557703788).abs() < 1e-8);
}
