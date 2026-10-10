use super::*;
use rstsr_blas_traits::lapack_eigh::*;

#[test]
fn test_dsygvd() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default
    let driver = DSYGVD::default().a(a.view()).b(b.view()).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

    // upper for c-contiguous
    let driver = DSYGVD::default().a(a.view()).b(b.view()).uplo(Upper).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -65.27252612342873).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -7.0849504857534535).abs() < 1e-8);

    // transpose upper for c-contiguous
    let driver = DSYGVD::default().a(a.t()).b(b.t()).uplo(Upper).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

    // itype 2
    let driver = DSYGVD::default().a(a.view()).b(b.view()).itype(2).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -2437.094304861363).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -4.108281604767547).abs() < 1e-8);

    // itype 3
    let driver = DSYGVD::default().a(a.view()).b(b.view()).itype(3).build().unwrap();
    let (w, v) = driver.run().unwrap();
    let v = v.into_owned();
    assert!((fingerprint(&w) - -2437.094304861363).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 30.756098926747757).abs() < 1e-8);
}
