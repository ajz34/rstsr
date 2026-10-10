use super::*;
use rstsr_blas_traits::lapack_eigh::*;

#[test]
fn test_dsyev() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default
    let driver = DSYEV::default().a(a.view()).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);

    // upper for c-contiguous
    let driver = DSYEV::default().a(a.view()).uplo(Upper).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 6.973792268793419).abs() < 1e-8);

    // transpose upper for c-contiguous
    let driver = DSYEV::default().a(a.t()).uplo(Upper).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);
}
