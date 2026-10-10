use super::*;

#[test]
fn test_cholesky() {
    let device = DeviceBLAS::default();
    let mut b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

    // default
    let c = rt::linalg::cholesky(b.view());
    assert!((fingerprint(&c) - c64!(62.89494065393874, -73.47055443374522)).norm() < 1e-8);

    // upper
    let c = rt::linalg::cholesky((b.view(), Upper));
    assert!((fingerprint(&c) - c64!(13.720509103165073, -1.8066465348490963)).norm() < 1e-8);

    // mutable changes itself
    rt::linalg::cholesky((b.view_mut(), Upper));
    assert!((fingerprint(&b) - c64!(13.720509103165073, -1.8066465348490963)).norm() < 1e-8);
}
