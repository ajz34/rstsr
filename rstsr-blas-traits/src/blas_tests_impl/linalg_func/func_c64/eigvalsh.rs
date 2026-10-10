use super::*;

#[test]
fn test_eigvalsh() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));
    let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

    // default, a
    let w = rt::linalg::eigvalsh(a.view());
    assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);

    // upper, a
    let w = rt::linalg::eigvalsh((a.view(), Upper));
    assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);

    // default, a b
    let w = rt::linalg::eigvalsh((a.view(), b.view()));
    assert!((fingerprint(&w) - -97.43376763322635).abs() < 1e-8);

    // upper, a b, itype=3
    let w = rt::linalg::eigvalsh((a.view(), b.view(), Upper, 3));
    assert!((fingerprint(&w) - -4656.824753078057).abs() < 1e-8);

    // mutable changes a
    let w = rt::linalg::eigvalsh(a.view_mut());
    assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
}
