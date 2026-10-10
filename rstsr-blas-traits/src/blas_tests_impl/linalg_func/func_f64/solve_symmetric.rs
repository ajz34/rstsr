use super::*;

#[test]
fn test_solve_symmetric() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<f64>('b')[..1024 * 512].to_vec();
    let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let x = rt::linalg::solve_symmetric((a.view(), b.view()));
    assert!((fingerprint(&x) - -397.1203235513806).abs() < 1e-8);

    // upper, mutable changes b
    rt::linalg::solve_symmetric((a.view(), b.view_mut(), Upper));
    assert!((fingerprint(&b) - -314.45022891879034).abs() < 1e-8);
}
