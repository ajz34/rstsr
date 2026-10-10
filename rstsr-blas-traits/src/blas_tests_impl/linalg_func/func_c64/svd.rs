use super::*;

#[test]
fn test_svd() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let (u, s, vt) = rt::linalg::svd(a.view()).into();
    assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -15.44133470545584).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

    // full_matrices = false
    let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
    assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - -1.9516528722381659).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

    // m < n, full_matrices = false
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
    assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
    assert!((fingerprint(&u.abs()) - 4.636614351700778).abs() < 1e-8);
    assert!((fingerprint(&vt.abs()) - 1.4497879458575658).abs() < 1e-8);
}
