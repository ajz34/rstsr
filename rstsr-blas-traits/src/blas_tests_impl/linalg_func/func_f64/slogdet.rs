use super::*;

#[test]
fn test_slogdet() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

    let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
    assert!(sign.to_scalar() - -1.0 < 1e-8);
    assert!(logabsdet.to_scalar() - 3031.1259211802403 < 1e-8);
}
