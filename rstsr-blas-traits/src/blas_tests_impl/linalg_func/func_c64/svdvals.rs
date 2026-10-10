use super::*;

#[test]
fn test_svdvals() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let s = rt::linalg::svdvals(a.view());
    assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);

    // m < n, full_matrices = false
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
    let s = rt::linalg::svdvals(a.view());
    assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
}
