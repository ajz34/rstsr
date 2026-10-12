//! Layout-matrix and zero-size tests for the SVD driver family
//! (skill `lapacke-translation` §4; ADR-0009 retrofit Part 1).
//!
//! Self-validating (reconstruction + orthogonality) so every layout case is
//! checked against identical semantics without pinned fingerprints.

use super::*;
use duplicate::duplicate_item;
use num::complex::ComplexFloat;
use num::Complex;
use rstsr_blas_traits::lapack_svd::*;

const ATOL: f64 = 1e-10;

fn make_case_f64(
    device: &DeviceBLAS,
    case: &str,
    m: usize,
    n: usize,
) -> (Tensor<f64, DeviceBLAS, Ix2>, Tensor<f64, DeviceBLAS, Ix2>) {
    // returns (a_logical, a_buffer); the logical view is what the drivers see
    let v = &get_vec::<f64>('b')[..(m + 4) * (n + 4)];
    match case {
        // f-contiguous, ld == m
        "f-contig" => {
            let a = rt::asarray((v, [m, n].f(), device)).into_dim::<Ix2>().into_owned();
            (a.clone(), a)
        },
        // f-prefer with ld > m (padded columns)
        "f-padded" => {
            let buf = rt::asarray((v, [m + 4, n + 4].f(), device)).into_dim::<Ix2>().into_owned();
            let a = buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned();
            (a, buf)
        },
        // c-contiguous, ld == n
        "c-contig" => {
            let a = rt::asarray((v, [m, n].c(), device)).into_dim::<Ix2>().into_owned();
            (a.clone(), a)
        },
        // c-prefer with ld > n (padded rows)
        "c-padded" => {
            let buf = rt::asarray((v, [m + 4, n + 4].c(), device)).into_dim::<Ix2>().into_owned();
            let a = buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned();
            (a, buf)
        },
        // fully non-contiguous (strided both dims)
        "non-contig" => {
            let buf = rt::asarray((v, [m + 4, n + 4].c(), device)).into_dim::<Ix2>().into_owned();
            let a =
                buf.i((slice!(0, (m * 2) as isize, 2), slice!(0, (n * 2) as isize, 2))).into_dim::<Ix2>().into_owned();
            (a, buf)
        },
        _ => panic!("unknown layout case {case}"),
    }
}

fn assert_svd_correct_f64(device: &DeviceBLAS, a: &Tensor<f64, DeviceBLAS, Ix2>, full_matrices: bool) {
    let m = a.nrow();
    let n = a.ncol();
    let k = m.min(n);

    let (s, u, vt) = {
        let d = if full_matrices {
            DGESVD::default().a(a.view()).build().unwrap().run().unwrap()
        } else {
            DGESVD::default().a(a.view()).full_matrices(false).build().unwrap().run().unwrap()
        };
        match d {
            (s, Some(u), Some(vt), _) => (s, u, vt),
            _ => panic!("SVD did not return factors"),
        }
    };

    // s is 1-D of length min(m, n), non-negative descending
    assert_eq!(s.shape(), &[k]);
    let sv = s.to_vec();
    for i in 1..k {
        assert!(sv[i] <= sv[i - 1] + ATOL, "singular values must descend");
    }
    assert!(sv.iter().all(|x| x.is_finite() && *x >= 0.0));

    // expected shapes
    let (u0, u1) = if full_matrices { (m, m) } else { (m, k) };
    let (vt0, vt1) = if full_matrices { (n, n) } else { (k, n) };
    assert_eq!(u.shape(), &[u0, u1]);
    assert_eq!(vt.shape(), &[vt0, vt1]);

    // reconstruction: U diag(s) VT == A (work on the leading k columns/rows)
    let u_k = u.i((0..m, 0..k)).into_dim::<Ix2>().into_owned();
    let vt_k = vt.i((0..k, 0..n)).into_dim::<Ix2>().into_owned();
    let us = &u_k * s.i((None, ..)).into_dim::<Ix2>();
    let recon = rt::matmul(us.view(), vt_k.view());
    let diff = recon - a.view();
    let norm: f64 = (diff.view() * diff.view()).sum_all().sqrt();
    let scale = (m * n) as f64;
    assert!(norm < ATOL * scale, "reconstruction failed: ||A - U S Vt|| = {norm}");

    // orthogonality: U^T U = I_k, VT VT^T = I_k
    let u_k_t = u_k.view().into_reverse_axes().into_owned();
    let vt_k_t = vt_k.view().into_reverse_axes().into_owned();
    let g1 = rt::matmul(u_k_t.view(), u_k.view());
    let g2 = rt::matmul(vt_k.view(), vt_k_t.view());
    let eye: Tensor<f64, _, Ix2> =
        EyeAPI::<(f64, _)>::eye_f((k, k, 0isize, ColMajor, device)).unwrap().into_dim::<Ix2>();
    for (g, name) in [(&g1, "U"), (&g2, "VT")] {
        let d = g.view() - eye.view();
        let norm: f64 = (d.view() * d.view()).sum_all().sqrt();
        assert!(norm < ATOL * (k * k) as f64, "{name} not orthogonal: ||G - I|| = {norm}");
    }
}

#[duplicate_item(
    name      case          ;
    [test_dgesvd_layout_f_contig]    ["f-contig"];
    [test_dgesvd_layout_f_padded]    ["f-padded"];
    [test_dgesvd_layout_c_contig]    ["c-contig"];
    [test_dgesvd_layout_c_padded]    ["c-padded"];
    [test_dgesvd_layout_non_contig]  ["non-contig"];
)]
#[test]
fn name() {
    let device = DeviceBLAS::default();
    for (m, n) in [(5usize, 8usize), (8, 5), (6, 6), (1, 7), (7, 1)] {
        let (a, _buf) = make_case_f64(&device, case, m, n);
        assert_svd_correct_f64(&device, &a, true);
        assert_svd_correct_f64(&device, &a, false);
    }
}

#[test]
fn test_dgesvd_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 4usize), (4, 0), (0, 0)] {
        let a = rt::zeros(([m, n], &device)).into_dim::<Ix2>();

        // full_matrices (jobz = 'A'): identity sub-blocks
        let driver = DGESVD::default().a(a.view()).build().unwrap();
        let (s, u, vt, superb) = driver.run().unwrap();
        assert_eq!(s.shape(), &[0]);
        assert_eq!(superb.shape(), &[0]);
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert_eq!(u.shape(), &[m, m]);
        assert_eq!(vt.shape(), &[n, n]);
        if m > 0 {
            let eye: Tensor<f64, _, Ix2> =
                EyeAPI::<(f64, _)>::eye_f((m, m, 0isize, u.device().default_order(), &device))
                    .unwrap()
                    .into_dim::<Ix2>();
            let d = u - eye;
            let norm: f64 = (d.view() * d.view()).sum_all().sqrt();
            assert!(norm < 1e-12, "zero-size U must be identity, got ||U - I|| = {norm}");
        }
        if n > 0 {
            let eye: Tensor<f64, _, Ix2> =
                EyeAPI::<(f64, _)>::eye_f((n, n, 0isize, vt.device().default_order(), &device))
                    .unwrap()
                    .into_dim::<Ix2>();
            let d = vt - eye;
            let norm: f64 = (d.view() * d.view()).sum_all().sqrt();
            assert!(norm < 1e-12, "zero-size VT must be identity, got ||VT - I|| = {norm}");
        }

        // economy (jobz = 'S', k = 0): zero-width factors
        let driver = DGESVD::default().a(a.view()).full_matrices(false).build().unwrap();
        let (_, u, vt, _) = driver.run().unwrap();
        assert_eq!(u.unwrap().shape(), &[m, 0]);
        assert_eq!(vt.unwrap().shape(), &[0, n]);

        // compute_uv = false
        let driver = DGESVD::default().a(a.view()).compute_uv(false).build().unwrap();
        let (_, u, vt, _) = driver.run().unwrap();
        assert!(u.is_none() && vt.is_none());
    }
}

#[test]
fn test_dgesdd_zero_size() {
    let device = DeviceBLAS::default();
    for (m, n) in [(0usize, 4usize), (4, 0), (0, 0)] {
        let a = rt::zeros(([m, n], &device)).into_dim::<Ix2>();

        let driver = DGESDD::default().a(a.view()).build().unwrap();
        let (s, u, vt) = driver.run().unwrap();
        assert_eq!(s.shape(), &[0]);
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert_eq!(u.shape(), &[m, m]);
        assert_eq!(vt.shape(), &[n, n]);
        if m > 0 {
            let eye: Tensor<f64, _, Ix2> =
                EyeAPI::<(f64, _)>::eye_f((m, m, 0isize, u.device().default_order(), &device))
                    .unwrap()
                    .into_dim::<Ix2>();
            let d = u - eye;
            let norm: f64 = (d.view() * d.view()).sum_all().sqrt();
            assert!(norm < 1e-12, "zero-size U must be identity");
        }
        if n > 0 {
            let eye: Tensor<f64, _, Ix2> =
                EyeAPI::<(f64, _)>::eye_f((n, n, 0isize, vt.device().default_order(), &device))
                    .unwrap()
                    .into_dim::<Ix2>();
            let d = vt - eye;
            let norm: f64 = (d.view() * d.view()).sum_all().sqrt();
            assert!(norm < 1e-12, "zero-size VT must be identity");
        }
    }
}

#[test]
fn test_dgesvd_economy_matches() {
    // driver-tier consistency: gesvd and gesdd agree on singular values
    let device = DeviceBLAS::default();
    let v = &get_vec::<f64>('b')[..12 * 9];
    let a = rt::asarray((v, [12, 9].f(), &device)).into_dim::<Ix2>();
    let (s1, _, _, _) = DGESVD::default().a(a.view()).build().unwrap().run().unwrap();
    let (s2, _, _) = DGESDD::default().a(a.view()).build().unwrap().run().unwrap();
    let d: Vec<_> = s1.to_vec().iter().zip(s2.to_vec()).map(|(a, b)| a - b).collect();
    let n = d.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(n < 1e-12, "gesvd/gesdd singular values disagree by {n}");
}

// complex twins: exercise the c32/c64 rwork + superb smuggling paths
#[duplicate_item(
    name      dt              drv        tol    ;
    [test_cgesvd_layouts] [Complex<f32>] [CGESVD] [1e-3];
    [test_zgesvd_layouts] [Complex<f64>] [ZGESVD] [1e-10];
)]
#[test]
fn name() {
    let device = DeviceBLAS::default();
    let v = get_vec::<dt>('a')[..64 * 64].to_vec();
    for (m, n) in [(5usize, 8usize), (8, 5)] {
        let a = rt::asarray((v.clone(), [m, n].f(), &device)).into_dim::<Ix2>().into_owned();
        let k = m.min(n);

        // economy path with superb
        let driver = drv::default().a(a.view()).full_matrices(false).build().unwrap();
        let (s, u, vt, superb) = driver.run().unwrap();
        assert_eq!(s.shape(), &[k]);
        assert_eq!(superb.shape(), &[k - 1]);
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert_eq!(u.shape(), &[m, k]);
        assert_eq!(vt.shape(), &[k, n]);

        // reconstruction: U diag(s) VT == A
        let us = &u * s.i((None, ..)).into_dim::<Ix2>();
        let recon = rt::matmul(us.view(), vt.view());
        let diff = recon - a.view();
        let norm = ((diff.view() * diff.view()).sum_all()).re().abs().sqrt();
        let norm: f64 = norm as f64;
        assert!(norm < tol, "complex reconstruction failed: {norm}");

        // full path through padded c-prefer input (layout matrix)
        let vb = get_vec::<dt>('b')[(m + 4) * (n + 4)..((m + 4) * (n + 4)) * 2].to_vec();
        let buf = rt::asarray((vb, [m + 4, n + 4].c(), &device)).into_dim::<Ix2>().into_owned();
        let a2 = buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned();
        let driver = drv::default().a(a2.view()).build().unwrap();
        let (s2, u2, vt2, _) = driver.run().unwrap();
        assert_eq!(s2.shape(), &[k]);
        assert_eq!(u2.unwrap().shape(), &[m, m]);
        assert_eq!(vt2.unwrap().shape(), &[n, n]);
    }
}

#[duplicate_item(
    name      dt              drv        ;
    [test_cgesdd_layouts] [Complex<f32>] [CGESDD];
    [test_zgesdd_layouts] [Complex<f64>] [ZGESDD];
)]
#[test]
fn name() {
    let device = DeviceBLAS::default();
    let v = get_vec::<dt>('a')[..64 * 64].to_vec();
    for (m, n) in [(5usize, 8usize), (8, 5)] {
        let a = rt::asarray((v.clone(), [m, n].f(), &device)).into_dim::<Ix2>().into_owned();
        let k = m.min(n);

        let driver = drv::default().a(a.view()).full_matrices(false).build().unwrap();
        let (s, u, vt) = driver.run().unwrap();
        assert_eq!(s.shape(), &[k]);
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert_eq!(u.shape(), &[m, k]);
        assert_eq!(vt.shape(), &[k, n]);

        // f-padded input
        let vb = get_vec::<dt>('b')[(m + 4) * (n + 4)..((m + 4) * (n + 4)) * 2].to_vec();
        let buf = rt::asarray((vb, [m + 4, n + 4].f(), &device)).into_dim::<Ix2>().into_owned();
        let a2 = buf.i((0..m, 0..n)).into_dim::<Ix2>().into_owned();
        let driver = drv::default().a(a2.view()).build().unwrap();
        let (_, u2, vt2) = driver.run().unwrap();
        assert_eq!(u2.unwrap().shape(), &[m, m]);
        assert_eq!(vt2.unwrap().shape(), &[n, n]);
    }
}
