use super::*;

#[test]
fn test_det() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<f64>('a')[..5 * 5].to_vec();
    let mut a = rt::asarray((a_vec, [5, 5].c(), &device));

    let det = rt::linalg::det(a.view_mut());
    assert!((det - 3.9699917597338046).abs() < 1e-8);
}
