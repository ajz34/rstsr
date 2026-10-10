use super::*;

#[test]
fn test_slogdet() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

    let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
    assert!((sign.to_scalar() - c64!(-0.44606842323663365, 0.8949988613351316)).norm() < 1e-8);
    assert!(logabsdet.to_scalar() - 3393.6720579594585 < 1e-8);
}
