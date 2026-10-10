use super::*;
use rstsr_blas_traits::lapack_solve::*;

#[test]
fn test_dsysv() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = &get_vec::<f64>('b')[..1024 * 512];
    let b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let driver = DSYSV::default().a(a.view()).b(b.view()).build().unwrap();
    let (udut, piv, x) = driver.run().unwrap();
    let udut = udut.into_owned();
    let x = x.into_owned();
    let fpiv = piv.map(|&v| v as f64);
    assert!((fingerprint(&udut) - -1201.6472395568974).abs() < 1e-8);
    assert!((fingerprint(&fpiv) - -16668.7094872639).abs() < 1e-8);
    assert!((fingerprint(&x) - -397.12032355166446).abs() < 1e-8);

    // upper
    let driver = DSYSV::default().a(a.view()).b(b.view()).uplo(Upper).build().unwrap();
    let (udut, piv, x) = driver.run().unwrap();
    let udut = udut.into_owned();
    let x = x.into_owned();
    let fpiv = piv.map(|&v| v as f64);
    assert!((fingerprint(&udut) - 1182.7836118324408).abs() < 1e-8);
    assert!((fingerprint(&fpiv) - 11905.503011559245).abs() < 1e-8);
    assert!((fingerprint(&x) - -314.4502289190444).abs() < 1e-8);
}
