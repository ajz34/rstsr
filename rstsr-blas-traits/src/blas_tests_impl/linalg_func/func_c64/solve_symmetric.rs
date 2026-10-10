use super::*;

#[test]
fn test_solve_symmetric() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b_vec = get_vec::<c64>('b')[..1024 * 512].to_vec();
    let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    // default (hermi)
    let x = rt::linalg::solve_symmetric((a.view(), b.view()));
    assert!((fingerprint(&x) - c64!(-1053.7242100144504, -559.2846004618166)).norm() < 1e-8);

    // upper (hermi)
    let x = rt::linalg::solve_symmetric((a.view(), b.view(), Upper));
    assert!((fingerprint(&x) - c64!(674.2725854112028, -68.55236080351166)).norm() < 1e-8);

    // default (symm)
    let x = rt::linalg::solve_symmetric((a.view(), b.view(), false));
    assert!((fingerprint(&x) - c64!(401.05642312535775, -805.8028453625365)).norm() < 1e-8);

    // upper, mutable changes b (symm)
    rt::linalg::solve_symmetric((a.view(), b.view_mut(), false, Upper));
    assert!((fingerprint(&b) - c64!(141.70122084637046, -829.609691493499)).norm() < 1e-8);
}
