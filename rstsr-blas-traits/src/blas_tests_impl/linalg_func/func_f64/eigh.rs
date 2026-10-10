use super::*;

#[test]
fn test_eigh() {
    let device = DeviceBLAS::default();
    let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));
    let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

    // default, a
    let (w, v) = rt::linalg::eigh(a.view()).into();
    assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);

    // upper, a
    let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
    assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 6.973792268793419).abs() < 1e-8);

    // default, a b
    let (w, v) = rt::linalg::eigh((a.view(), b.view())).into();
    assert!((fingerprint(&w) - -89.60433120129908).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - -5.243112559130817).abs() < 1e-8);

    // upper, a b, itype=3
    let (w, v) = rt::linalg::eigh((a.view(), b.view(), Upper, 3)).into();
    assert!((fingerprint(&w) - -2503.84161931662).abs() < 1e-8);
    assert!((fingerprint(&v.abs()) - 152.17700520642055).abs() < 1e-8);

    // mutable changes a
    let (w, _) = rt::linalg::eigh(a.view_mut()).into();
    assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
    assert!((fingerprint(&a.abs()) - -9.903934930318247).abs() < 1e-8);
}
