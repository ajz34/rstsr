use super::*;

#[test]
fn test_solve_general() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<c64>('b')[..1024 * 512].to_vec();
    let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default
    let x = rt::linalg::solve_general((a.view(), b.view()));
    assert!((fingerprint(&x) - c64!(404.1900761036138, -258.5602505551204)).norm() < 1e-8);

    // mutable changes itself
    rt::linalg::solve_general((a.view_mut(), b.view_mut()));
    assert!((fingerprint(&b) - c64!(404.1900761036138, -258.5602505551204)).norm() < 1e-8);
}

#[test]
fn test_solve_general_for_vec() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<c64>('b')[..1024].to_vec();
    let mut b = rt::asarray((b_vec, [1024].c(), &device)).into_dim::<Ix1>();

    // default
    let x = rt::linalg::solve_general((a.view(), b.view()));
    assert!((fingerprint(&x) - c64!(-15.070310793269726, -1.987917054716041)).norm() < 1e-8);

    // mutable changes itself
    rt::linalg::solve_general((a.view_mut(), b.view_mut()));
    assert!((fingerprint(&b) - c64!(-15.070310793269726, -1.987917054716041)).norm() < 1e-8);
}
