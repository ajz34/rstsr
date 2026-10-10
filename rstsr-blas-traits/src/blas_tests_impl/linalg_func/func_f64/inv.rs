use super::*;

#[test]
fn test_inv() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

    // immutable
    let a_inv = rt::linalg::inv(a.view());
    assert!((fingerprint(&a_inv) - 143.39005577037764).abs() < 1e-8);

    // mutable
    rt::linalg::inv(a.view_mut());
    assert!((fingerprint(&a) - 143.39005577037764).abs() < 1e-8);
}
