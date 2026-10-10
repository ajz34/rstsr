use super::*;

#[test]
fn test_generalized_eigh() {
    let device = DeviceBLAS::default();
    let a_vec = get_vec::<c64>('a')[..1024 * 1024].to_vec();
    let b_vec = get_vec::<c64>('b')[..1024 * 1024].to_vec();
    let a = rt::asarray((a_vec, [1024, 1024].c(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((b_vec, [1024, 1024].c(), &device)).into_dim::<Ix2>();

    // 1, lower
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Lower, 1)).into();
    println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
    assert!((fingerprint(&w) - -97.43376763322635).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -4.3181177983574255).abs() < 1e-8);

    // 1, upper
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 1)).into();
    println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
    assert!((fingerprint(&w) - -54.81859256480441).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -1.4841788446757156).abs() < 1e-8);

    // 2, lower
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Lower, 2)).into();
    println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
    assert!((fingerprint(&w) - -4967.627482507203).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 5.541034627252399).abs() < 1e-8);

    // 2, upper
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 2)).into();
    println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
    assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 1.0609263552377188).abs() < 1e-8);

    // 3, lower
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Lower, 3)).into();
    println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
    assert!((fingerprint(&w) - -4967.627482507203).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 118.76501084045631).abs() < 1e-8);

    // 3, upper
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 3)).into();
    println!("w: {:?}, v: {:?}", fingerprint(&w), fingerprint(&v.abs()));
    assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -0.15861903557045487).abs() < 1e-8);
}
