#![cfg(feature = "faer")]

use rstsr::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_test_manifest::get_vec;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

macro_rules! c64 {
    ($real:expr, $imag:expr) => {
        c64::new($real, $imag)
    };
    ($real:expr) => {
        c64::new($real, 0.0)
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cholesky() {
        let device = DeviceFaer::default();
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device));

        // default
        let c = rt::linalg::cholesky(b.view());
        assert!((fingerprint(&c) - c64!(62.89494065393874, -73.47055443374522)).norm() < 1e-8);

        // upper
        let c = rt::linalg::cholesky((b.view(), Upper));
        assert!((fingerprint(&c) - c64!(13.720509103165073, -1.8066465348490963)).norm() < 1e-8);
    }

    #[test]
    fn test_det() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..5 * 5].to_vec();
        let a = rt::asarray((a_vec, [5, 5].c(), &device));

        let det = rt::linalg::det(a.view());
        assert!((det.to_scalar() - c64!(-24.808965756481086, 11.800248863799464)).norm() < 1e-8);
    }

    #[test]
    fn test_eigh() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // faer does not ensure the diagonal is real, so we need to do it ourselves
        for i in 0..1024 {
            a[[i, i]] = c64!(a[[i, i]].re, 0.0);
        }

        // default, a
        let (w, v) = rt::linalg::eigh((a.view(), Lower)).into();
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -7.450761195788254).abs() < 1e-8);

        // upper, a
        let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
        assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -12.184946930165328).abs() < 1e-8);
    }

    #[test]
    fn test_eigvalsh() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // faer does not ensure the diagonal is real, so we need to do it ourselves
        for i in 0..1024 {
            a[[i, i]] = c64!(a[[i, i]].re, 0.0);
        }

        // default, a
        let w = rt::linalg::eigvalsh(a.view());
        assert!((fingerprint(&w) - -100.79793355894122).abs() < 1e-8);

        // upper, a
        let w = rt::linalg::eigvalsh((a.view(), Upper));
        assert!((fingerprint(&w) - -103.99103522434956).abs() < 1e-8);
    }

    #[test]
    fn test_inv() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        // immutable
        let a_inv = rt::linalg::inv(a.view());
        assert!((fingerprint(&a_inv) - c64!(-11.836382515156183, 8.250167298349842)).norm() < 1e-8);
    }

    #[test]
    fn test_pinv() {
        let device = DeviceFaer::default();

        // 1024 x 512
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        println!("rank: {rank}, a_pinv: {:?}", fingerprint(&a_pinv));
        assert!((fingerprint(&a_pinv) - c64!(-0.03454885412959018, -0.023651876085623254)).norm() < 1e-8);
        assert_eq!(rank, 240);

        // 512 x 1024
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - c64!(-0.2814806469687325, -0.15198888300458474)).norm() < 1e-8);
        assert_eq!(rank, 240);
    }

    #[test]
    fn test_solve_general() {
        let device = DeviceFaer::default();
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
        let device = DeviceFaer::default();
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

    #[test]
    fn test_solve_triangular() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let mut a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<c64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_triangular((b.view(), a.view()));
        assert!((fingerprint(&x) - c64!(-8.433708003916948, 20.578272827017052)).norm() < 1e-8);

        // upper, mutable changes a
        rt::linalg::solve_triangular((b.view(), a.view_mut(), Upper));
        assert!((fingerprint(&a) - c64!(0.1778922244846507, 11.42463765128442)).norm() < 1e-8);
    }

    #[test]
    fn test_svd() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        // note that full_matrices = true will give a different u compared to BLAS.
        let (_u, s, vt) = rt::linalg::svd(a.view()).into();
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
        // assert!((fingerprint(&u.abs()) - -15.44133470545584).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

        // full_matrices = false
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -1.9516528722381659).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 2.1605324161714172).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device));
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - 4.636614351700778).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 1.4497879458575658).abs() < 1e-8);
    }

    #[test]
    fn test_svdvals() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 46.60343405921802).abs() < 1e-8);

        // m < n
        let a_vec = get_vec::<c64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 47.599274835886646).abs() < 1e-8);
    }
}

#[cfg(test)]
mod test_generalized_eigh {
    use super::*;

    #[test]
    fn test_generalized_eigh() {
        let device = DeviceFaer::default();
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

    #[test]
    fn test_slogdet() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<c64>('a'), [1024, 1024].c(), &device));

        let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
        assert!((sign.to_scalar() - c64!(-0.44606842323663365, 0.8949988613351316)).norm() < 1e-8);
        assert!(logabsdet.to_scalar() - 3393.6720579594585 < 1e-8);
    }

    #[test]
    fn test_matrix_power() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<c64>('a')[..16].to_vec();
        let a = rt::asarray((a_vec, [4, 4].c(), &device));

        // n = 0 -> identity, in the input dtype
        let p = rt::linalg::matrix_power((a.view(), 0));
        assert!((fingerprint(&p) - c64!(-0.3150972564720475)).norm() < 1e-8);

        // n = 1 -> a itself
        let p = rt::linalg::matrix_power((a.view(), 1));
        assert!((fingerprint(&p) - c64!(-0.5537625095088878, -3.0610178896100395)).norm() < 1e-8);

        // shortcuts (2, 3) and the squaring loop (> 3)
        for (n, golden) in [
            (2isize, c64!(-3.3981320140425257, 0.16690991094120955)),
            (3, c64!(-14.635121297638918, -1.0839909116245399)),
            (5, c64!(-7.659626934528033, 29.380294417607516)),
        ] {
            let p = rt::linalg::matrix_power((a.view(), n));
            assert!((fingerprint(&p) - golden).norm() < 1e-8, "n = {n}");
        }

        // negative exponent: invert once, then raise the inverse to |n|
        for (n, golden) in [
            (-1isize, c64!(0.23944130706023403, 0.38201618793714387)),
            (-2, c64!(-0.40283871318247577, 0.22747768835164975)),
        ] {
            let p = rt::linalg::matrix_power((a.view(), n));
            assert!((fingerprint(&p) - golden).norm() < 1e-8, "n = {n}");
        }
    }

    #[test]
    fn test_matrix_power_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        let a_vec = get_vec::<c64>('a')[..16].to_vec();

        // stack: slice 0 is a, slice 1 is 2*a
        let mut stacked = a_vec.clone();
        stacked.extend(a_vec.iter().map(|x| 2.0 * x));
        let b = rt::asarray((stacked, [2, 4, 4].c(), &device));

        let p = rt::linalg::matrix_power((b.view(), 0));
        assert_eq!(p.shape(), &[2, 4, 4]);
        assert!((fingerprint(&p) - c64!(-0.2588243168865289)).norm() < 1e-8);

        let p = rt::linalg::matrix_power((b.view(), 2));
        assert!((fingerprint(&p) - c64!(5.8067955793386306, -11.549488782937473)).norm() < 1e-8);

        let p = rt::linalg::matrix_power((b.view(), -1));
        assert!((fingerprint(&p) - c64!(-0.06327227000728392, 0.22792209504132835)).norm() < 1e-8);
    }

    #[test]
    fn test_matrix_power_nd_colmajor() {
        let mut device = DeviceFaer::default();
        device.set_default_order(ColMajor);
        // Under ColMajor the matrix axes are the first two and the batch trails.
        // Passing a shape (not `.c()`) makes `asarray` follow the device order (F),
        // so each `[:, :, b]` slice equals the 2-D operand.
        let a_vec = get_vec::<c64>('a')[..16].to_vec();
        // F-order of the operand, so the 2-D and stacked inputs share one layout
        let mut va = Vec::with_capacity(16);
        for j in 0..4 {
            for i in 0..4 {
                va.push(a_vec[i * 4 + j]);
            }
        }
        let v2a: Vec<c64> = va.iter().map(|x| 2.0 * x).collect();
        let a2 = rt::asarray((va.clone(), [4, 4], &device));
        let b2 = rt::asarray((v2a.clone(), [4, 4], &device));

        // stack: slice 0 = a, slice 1 = 2*a (batch axis trails under ColMajor)
        let mut stacked = va;
        stacked.extend_from_slice(&v2a);
        let b_t = rt::asarray((stacked, [4, 4, 2], &device));

        for n in [0isize, 2, 5, -1] {
            let p_nd = rt::linalg::matrix_power((b_t.view(), n));
            assert_eq!(p_nd.shape(), &[4, 4, 2]);
            let p0 = rt::linalg::matrix_power((a2.view(), n));
            let p1 = rt::linalg::matrix_power((b2.view(), n));
            assert!(rt::allclose(p_nd.i((.., .., 0)), &p0, None), "n = {n}");
            assert!(rt::allclose(p_nd.i((.., .., 1)), &p1, None), "n = {n}");
        }
    }
}
