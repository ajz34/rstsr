use super::*;

#[test]
fn test_pinv() {
    let device = DeviceBLAS::default();

    // 1024 x 512
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

    let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
    assert!((fingerprint(&a_pinv) - c64!(-0.03454885412959018, -0.023651876085623254)).norm() < 1e-8);
    assert_eq!(rank, 240);

    // 512 x 1024
    let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
    let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

    let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
    assert!((fingerprint(&a_pinv) - c64!(-0.2814806469687325, -0.15198888300458474)).norm() < 1e-8);
    assert_eq!(rank, 240);
}
