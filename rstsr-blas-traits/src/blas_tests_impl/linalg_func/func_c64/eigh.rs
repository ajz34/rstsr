use super::*;

#[test]
fn test_eigh() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));
    let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

    // default, a
    let (w, v) = rt::linalg::eigh(a.view()).into();
    assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -7.450761195788254).abs() < 1e-8);

    // upper, a
    let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
    assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -12.184946930165328).abs() < 1e-8);

    // default, a b
    let (w, v) = rt::linalg::eigh((a.view(), b.view())).into();
    assert!((fingerprint(&w) - -97.43376763322635).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -4.3181177983574255).abs() < 1e-8);

    // upper, a b, itype=3
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 3)).into();
    assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -0.15861903557045487).abs() < 1e-8);

    // mutable changes a
    let (w, _) = rt::linalg::eigh(a.view_mut()).into();
    assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
    assert!((fingerprint(&a.abs()) - -7.450761195788254).abs() < 1e-8);
}
