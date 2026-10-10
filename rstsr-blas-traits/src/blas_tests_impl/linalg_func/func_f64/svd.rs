use super::*;

#[test]
fn test_svd() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let (u, s, vt) = rt::linalg::svd(a.view()).into();
    assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -1.9368850983570982).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

    // full_matrices = false
    let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
    assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -9.144981428076894).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

    // m < n, full_matrices = false
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
    assert!((fingerprint(&s) - 32.27742168207757).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -3.716931052161584).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - -0.32301437281530243).abs() < 1e-8);
}
