use super::*;
use rstsr_blas_traits::lapack_solve::*;

#[test]
fn test_dpotrf() {
    let device = DeviceBLAS::default();
    let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default
    let driver = DPOTRF::default().a(b.view()).build().unwrap();
    let c = driver.run().unwrap();
    let c = c.into_owned();
    println!("fingerprint {:?}", fingerprint(&c));
    assert!((fingerprint(&c) - 35.17266259472725).abs() < 1e-8);

    // upper
    let driver = DPOTRF::default().a(b.view()).uplo(Upper).build().unwrap();
    let c = driver.run().unwrap();
    let c = c.into_owned();
    println!("fingerprint {:?}", fingerprint(&c));
    assert!((fingerprint(&c) - -53.53353704132017).abs() < 1e-8);
}
