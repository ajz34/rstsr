#![cfg(feature = "faer")]

use rstsr::prelude::*;
use rstsr_core::prelude_dev::fingerprint;
use rstsr_test_manifest::get_vec;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cholesky() {
        let device = DeviceFaer::default();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device));

        // default
        let c = rt::linalg::cholesky(b.view());
        assert!((fingerprint(&c) - 43.21904478556176).abs() < 1e-8);

        // upper
        let c = rt::linalg::cholesky((b.view(), Upper));
        assert!((fingerprint(&c) - -25.925655124816647).abs() < 1e-8);
    }

    #[test]
    fn test_cholesky_submatrix() {
        let device = DeviceFaer::default();
        let vec_b: Vec<f64> = vec![0.0, 1.0, 2.0, 1.0, 5.0, 1.5, 2.0, 1.5, 8.0];
        let b = rt::asarray((vec_b, [3, 3].c(), &device));

        let b_view = b.i((1..3, 1..3));
        let c = rt::linalg::cholesky(b_view);
        assert!((fingerprint(&c) - -0.7633202592326889).abs() < 1e-8);
    }

    #[test]
    fn test_det() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<f64>('a')[..5 * 5].to_vec();
        let a = rt::asarray((a_vec, [5, 5].c(), &device));

        let det = rt::linalg::det(a.view());
        assert!((det - 3.9699917597338046).abs() < 1e-8);
    }

    #[test]
    fn test_eigh() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

        // default, a
        let (w, v) = rt::linalg::eigh(a.view()).into();
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - -9.903934930318247).abs() < 1e-8);

        // upper, a
        let (w, v) = rt::linalg::eigh((a.view(), Upper)).into();
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
        assert!((fingerprint(&v.abs()) - 6.973792268793419).abs() < 1e-8);
    }

    #[test]
    fn test_eigvalsh() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

        // default, a
        let w = rt::linalg::eigvalsh(a.view());
        assert!((fingerprint(&w) - -71.4747209499407).abs() < 1e-8);

        // upper, a
        let w = rt::linalg::eigvalsh((a.view(), Upper));
        assert!((fingerprint(&w) - -71.4902453763506).abs() < 1e-8);
    }

    #[test]
    fn test_inv() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

        // immutable
        let a_inv = rt::linalg::inv(a.view());
        assert!((fingerprint(&a_inv) - 143.39005577037764).abs() < 1e-8);
    }

    #[test]
    fn test_pinv() {
        let device = DeviceFaer::default();

        // 1024 x 512
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - 0.0878262837784408).abs() < 1e-8);
        assert_eq!(rank, 163);

        // 512 x 1024
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();

        let (a_pinv, rank) = rt::linalg::pinv((a.view(), 20.0, 0.3)).into();
        assert!((fingerprint(&a_pinv) - -0.3244041253699862).abs() < 1e-8);
        assert_eq!(rank, 161);
    }

    #[test]
    fn test_solve_general() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<f64>('b')[..1024 * 512].to_vec();
        let mut b = rt::asarray((b_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - -1951.253447757597).abs() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - -1951.253447757597).abs() < 1e-8);
    }

    #[test]
    fn test_solve_general_for_vec() {
        let device = DeviceFaer::default();
        let mut a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device)).into_dim::<Ix2>();
        let b_vec = get_vec::<f64>('b')[..1024].to_vec();
        let mut b = rt::asarray((b_vec, [1024].c(), &device)).into_dim::<Ix1>();

        // default
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert!((fingerprint(&x) - -9.120066438800688).abs() < 1e-8);

        // mutable changes itself
        rt::linalg::solve_general((a.view_mut(), b.view_mut()));
        assert!((fingerprint(&b) - -9.120066438800688).abs() < 1e-8);
    }

    #[test]
    fn test_solve_triangular() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let mut a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();
        let b = rt::asarray((get_vec::<f64>('b'), [1024, 1024].c(), &device)).into_dim::<Ix2>();

        // default
        let x = rt::linalg::solve_triangular((b.view(), a.view()));
        assert!((fingerprint(&x) - -2.6133848012216587).abs() < 1e-8);

        // upper, mutable changes a
        rt::linalg::solve_triangular((b.view(), a.view_mut(), Upper));
        assert!((fingerprint(&a) - 5.112256818100785).abs() < 1e-8);
    }

    #[test]
    fn test_svd() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let (u, s, vt) = rt::linalg::svd(a.view()).into();
        assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -1.9368850983570982).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

        // full_matrices = false
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -9.144981428076894).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - 13.465522484136157).abs() < 1e-8);

        // m < n, full_matrices = false
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let (u, s, vt) = rt::linalg::svd((a.view(), false)).into();
        assert!((fingerprint(&s) - 32.27742168207757).abs() < 1e-8);
        assert!((fingerprint(&u.abs()) - -3.716931052161584).abs() < 1e-8);
        assert!((fingerprint(&vt.abs()) - -0.32301437281530243).abs() < 1e-8);
    }

    #[test]
    fn test_svdvals() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [1024, 512].c(), &device)).into_dim::<Ix2>();

        // default
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 33.969339071043095).abs() < 1e-8);

        // m < n
        let a_vec = get_vec::<f64>('a')[..1024 * 512].to_vec();
        let a = rt::asarray((a_vec, [512, 1024].c(), &device)).into_dim::<Ix2>();
        let s = rt::linalg::svdvals(a.view());
        assert!((fingerprint(&s) - 32.27742168207757).abs() < 1e-8);
    }

    #[test]
    fn test_slogdet() {
        let device = DeviceFaer::default();
        let a = rt::asarray((get_vec::<f64>('a'), [1024, 1024].c(), &device));

        let (sign, logabsdet) = rt::linalg::slogdet(a.view()).into();
        assert!(sign.to_scalar() - -1.0 < 1e-8);
        assert!(logabsdet.to_scalar() - 3031.1259211802403 < 1e-8);
    }

    #[test]
    fn test_slogdet_nd() {
        let device = DeviceFaer::default();
        let vec_a: Vec<f64> = vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0];

        let a = rt::asarray((vec_a.clone(), [3, 3].c(), &device));
        let (sign0, log0) = rt::linalg::slogdet(a.view()).into();
        let (sign0, log0) = (sign0.to_scalar(), log0.to_scalar());

        // same matrix stacked twice; each slice must match the 2-D result
        let mut stacked = vec_a.clone();
        stacked.extend_from_slice(&vec_a);
        let b = rt::asarray((stacked, [2, 3, 3].c(), &device));
        let (sign, log) = rt::linalg::slogdet(b.view()).into();

        assert_eq!(sign.ndim(), 1);
        assert_eq!(sign.shape()[0], 2);
        for i in 0..2 {
            assert!((sign.i(i).to_scalar() - sign0).abs() < 1e-12);
            assert!((log.i(i).to_scalar() - log0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_solve_general_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        // well-conditioned 4x4 system, stacked twice along the batch axis
        let a_slice: Vec<f64> = vec![4.0, 1.0, 0.0, 0.5, 1.0, 3.0, 0.5, 0.0, 0.0, 0.5, 2.0, 1.0, 0.5, 0.0, 1.0, 2.5];
        let b_slice: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let a2 = rt::asarray((a_slice.clone(), [4, 4].c(), &device));
        let b2 = rt::asarray((b_slice.clone(), [4, 2].c(), &device));
        let x2 = rt::linalg::solve_general((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        let mut a_st = a_slice.clone();
        a_st.extend_from_slice(&a_slice);
        let mut b_st = b_slice.clone();
        b_st.extend_from_slice(&b_slice);
        let a_nd = rt::asarray((a_st, [2, 4, 4].c(), &device));

        // allocating path (b as an immutable view)
        let b_nd = rt::asarray((b_st.clone(), [2, 4, 2].c(), &device));
        let x_nd = rt::linalg::solve_general((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.ndim(), 3);
        assert_eq!(x_nd.shape(), &[2, 4, 2]);
        for i in 0..2 {
            assert!((fingerprint(&x_nd.i(i).to_owned()) - fp2).abs() < 1e-8);
        }

        // in-place path (b owned): b's own buffer holds the solution
        let mut b_mut = rt::asarray((b_st, [2, 4, 2].c(), &device));
        rt::linalg::solve_general((a_nd.view(), b_mut.view_mut()));
        assert!((fingerprint(&b_mut) - fingerprint(&x_nd)).abs() < 1e-8);
    }

    #[test]
    fn test_solve_general_vec_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        let a_slice: Vec<f64> = vec![4.0, 1.0, 0.0, 0.5, 1.0, 3.0, 0.5, 0.0, 0.0, 0.5, 2.0, 1.0, 0.5, 0.0, 1.0, 2.5];
        let b_slice: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let a2 = rt::asarray((a_slice.clone(), [4, 4].c(), &device));
        let b2 = rt::asarray((b_slice.clone(), [4].c(), &device));
        let x2 = rt::linalg::solve_general((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        let mut a_st = a_slice.clone();
        a_st.extend_from_slice(&a_slice);
        let mut b_st = b_slice.clone();
        b_st.extend_from_slice(&b_slice);
        let a_nd = rt::asarray((a_st, [2, 4, 4].c(), &device));
        let b_nd = rt::asarray((b_st.clone(), [2, 4].c(), &device));

        let x_nd = rt::linalg::solve_general((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.shape(), &[2, 4]);
        for i in 0..2 {
            assert!((fingerprint(&x_nd.i(i).to_owned()) - fp2).abs() < 1e-8);
        }

        // in-place path
        let mut b_mut = rt::asarray((b_st, [2, 4].c(), &device));
        rt::linalg::solve_general((a_nd.view(), b_mut.view_mut()));
        assert!((fingerprint(&b_mut) - fingerprint(&x_nd)).abs() < 1e-8);
    }

    #[test]
    fn test_solve_triangular_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        // lower-triangular, stacked twice
        let a_slice: Vec<f64> = vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.5, 0.5, 4.0];
        let b_slice: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a2 = rt::asarray((a_slice.clone(), [3, 3].c(), &device));
        let b2 = rt::asarray((b_slice.clone(), [3, 2].c(), &device));
        let x2 = rt::linalg::solve_triangular((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        let mut a_st = a_slice.clone();
        a_st.extend_from_slice(&a_slice);
        let mut b_st = b_slice.clone();
        b_st.extend_from_slice(&b_slice);
        let a_nd = rt::asarray((a_st, [2, 3, 3].c(), &device));
        let b_nd = rt::asarray((b_st.clone(), [2, 3, 2].c(), &device));

        let x_nd = rt::linalg::solve_triangular((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.shape(), &[2, 3, 2]);
        for i in 0..2 {
            assert!((fingerprint(&x_nd.i(i).to_owned()) - fp2).abs() < 1e-8);
        }

        // in-place path (b owned): b's own buffer holds the solution
        let mut b_mut = rt::asarray((b_st, [2, 3, 2].c(), &device));
        rt::linalg::solve_triangular((a_nd.view(), b_mut.view_mut()));
        assert!((fingerprint(&b_mut) - fingerprint(&x_nd)).abs() < 1e-8);
    }

    #[test]
    fn test_solve_general_nd_colmajor() {
        let mut device = DeviceFaer::default();
        device.set_default_order(ColMajor);
        // Under ColMajor the matrix axes are the first two and the batch trails.
        // Passing a shape (not an explicit `.c()` layout) makes `asarray` follow the
        // device order (F), so each `[:, :, b]` slice equals the 2-D operand.
        let a_slice: Vec<f64> = vec![4.0, 1.0, 0.0, 0.5, 1.0, 3.0, 0.5, 0.0, 0.0, 0.5, 2.0, 1.0, 0.5, 0.0, 1.0, 2.5];
        let b_slice: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let a2 = rt::asarray((a_slice.clone(), [4, 4], &device));
        let b2 = rt::asarray((b_slice.clone(), [4, 2], &device));
        let x2 = rt::linalg::solve_general((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        let mut a_st = a_slice.clone();
        a_st.extend_from_slice(&a_slice);
        let mut b_st = b_slice.clone();
        b_st.extend_from_slice(&b_slice);
        let a_nd = rt::asarray((a_st, [4, 4, 2], &device));
        let b_nd = rt::asarray((b_st, [4, 2, 2], &device));

        let x_nd = rt::linalg::solve_general((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.shape(), &[4, 2, 2]);
        for i in 0..2 {
            let slice = x_nd.i((.., .., i)).to_owned();
            assert!((fingerprint(&slice) - fp2).abs() < 1e-8);
        }
    }
}
