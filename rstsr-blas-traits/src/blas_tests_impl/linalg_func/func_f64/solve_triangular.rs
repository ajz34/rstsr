use super::*;

#[test]
fn test_solve_triangular() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
    let mut a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // default
    let x = rt::linalg::solve_triangular((b.view(), a.view()));
    assert!((fingerprint(&x) - -2.6133848012216587).abs() < 1e-8);

    // upper, mutable changes a
    rt::linalg::solve_triangular((b.view(), a.view_mut(), Upper));
    assert!((fingerprint(&a) - 5.112256818100785).abs() < 1e-8);
}
