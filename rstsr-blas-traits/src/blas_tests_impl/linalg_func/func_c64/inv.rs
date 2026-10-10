use super::*;

#[test]
fn test_inv() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

    // immutable
    let a_inv = rt::linalg::inv(a.view());
    assert!((fingerprint(&a_inv) - c64!(-11.836382515156183, 8.250167298349842)).norm() < 1e-8);

    // mutable
    rt::linalg::inv(a.view_mut());
    assert!((fingerprint(&a) - c64!(-11.836382515156183, 8.250167298349842)).norm() < 1e-8);
}
