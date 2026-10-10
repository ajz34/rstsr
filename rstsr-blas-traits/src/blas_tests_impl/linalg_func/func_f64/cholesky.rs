use super::*;

#[test]
fn test_cholesky() {
    let device = DeviceBLAS::default();
    let mut b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

    // default
    let c = rt::linalg::cholesky(b.view());
    assert!((fingerprint(&c) - 43.21904478556176).abs() < 1e-8);

    // upper
    let c = rt::linalg::cholesky((b.view(), Upper));
    assert!((fingerprint(&c) - -25.925655124816647).abs() < 1e-8);

    // mutable changes itself
    rt::linalg::cholesky((b.view_mut(), Upper));
    assert!((fingerprint(&b) - -25.925655124816647).abs() < 1e-8);
}

#[test]
fn test_cholesky_submatrix() {
    let device = DeviceBLAS::default();
    let vec_b: Vec<f64> = vec![0.0, 1.0, 2.0, 1.0, 5.0, 1.5, 2.0, 1.5, 8.0];
    let b = rt::asarray((vec_b, [3, 3].c(), &device));

    let b_view = b.i((1..3, 1..3));
    let c = rt::linalg::cholesky(b_view);
    assert!((fingerprint(&c) - -0.7633202592326889).abs() < 1e-8);
}
