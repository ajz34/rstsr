use super::*;

#[test]
fn test_solve_general() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<f64>('b')[..1024 * 512].to_vec();
    let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let x = rt::linalg::solve_general((a.view(), b.view()));
    assert!((fingerprint(&x) - -1951.253447757597).abs() < 1e-8);

    // mutable changes itself
    rt::linalg::solve_general((a.view_mut(), b.view_mut()));
    assert!((fingerprint(&b) - -1951.253447757597).abs() < 1e-8);
}

#[test]
fn test_solve_general_for_vec() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<f64>('b')[..1024].to_vec();
    let mut b = rt::asarray((b_vec, [1024].c(), &device)).into_dim::<Ix1>();

    // default
    let x = rt::linalg::solve_general((a.view(), b.view()));
    assert!((fingerprint(&x) - -9.120066438800688).abs() < 1e-8);

    // mutable changes itself
    rt::linalg::solve_general((a.view_mut(), b.view_mut()));
    assert!((fingerprint(&b) - -9.120066438800688).abs() < 1e-8);
}
