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
        assert!((det.to_scalar() - 3.9699917597338046).abs() < 1e-8);
    }

    #[test]
    fn test_det_nd() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<f64>('a')[..5 * 5].to_vec();
        let a = rt::asarray((a_vec.clone(), [5, 5].c(), &device));
        let det0 = rt::linalg::det(a.view()).to_scalar();

        // distinct matrices per slot: slice 1 is 2*a, so its det scales by 2**5
        let mut stacked = a_vec.clone();
        stacked.extend(a_vec.iter().map(|x| 2.0 * x));
        let b = rt::asarray((stacked, [2, 5, 5].c(), &device));
        let det = rt::linalg::det(b.view());
        assert_eq!(det.ndim(), 1);
        assert_eq!(det.shape()[0], 2);
        assert!((det.i(0).to_scalar() - det0).abs() < 1e-8);
        assert!((det.i(1).to_scalar() - 32.0 * det0).abs() < 1e-8);
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

        // distinct matrices per slot: slice 1 is 2*a, so log|det| gains 3*ln(2)
        let mut stacked = vec_a.clone();
        stacked.extend(vec_a.iter().map(|x| 2.0 * x));
        let b = rt::asarray((stacked, [2, 3, 3].c(), &device));
        let (sign, log) = rt::linalg::slogdet(b.view()).into();

        assert_eq!(sign.ndim(), 1);
        assert_eq!(sign.shape()[0], 2);
        assert!((sign.i(0).to_scalar() - sign0).abs() < 1e-12);
        assert!((log.i(0).to_scalar() - log0).abs() < 1e-12);
        assert!((sign.i(1).to_scalar() - sign0).abs() < 1e-12);
        assert!((log.i(1).to_scalar() - log0 - 3.0 * 2f64.ln()).abs() < 1e-12);
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

        // distinct systems per slot: slice 1 is (2*a, 3*b), so x1 = 1.5 * x0
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let mut b_st = b_slice.clone();
        b_st.extend(b_slice.iter().map(|x| 3.0 * x));
        let a_nd = rt::asarray((a_st, [2, 4, 4].c(), &device));

        // allocating path (b as an immutable view)
        let b_nd = rt::asarray((b_st.clone(), [2, 4, 2].c(), &device));
        let x_nd = rt::linalg::solve_general((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.ndim(), 3);
        assert_eq!(x_nd.shape(), &[2, 4, 2]);
        assert!((fingerprint(&x_nd.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x_nd.i(1).to_owned()) - 1.5 * fp2).abs() < 1e-8);

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
        let b_vec: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let a2 = rt::asarray((a_slice.clone(), [4, 4].c(), &device));
        let b2 = rt::asarray((b_vec.clone(), [4].c(), &device));
        let x2 = rt::linalg::solve_general((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        // 1-D b broadcasts into a's batch -> result (2, 4);
        // slice 1 is 2*a with the same b, so x1 = 0.5 * x0
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let a_nd = rt::asarray((a_st, [2, 4, 4].c(), &device));
        let x_nd = rt::linalg::solve_general((a_nd.view(), b2.view()));
        assert_eq!(x_nd.shape(), &[2, 4]);
        assert!((fingerprint(&x_nd.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x_nd.i(1).to_owned()) - 0.5 * fp2).abs() < 1e-8);
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

        // distinct systems per slot: slice 1 is (2*a, 3*b), so x1 = 1.5 * x0
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let mut b_st = b_slice.clone();
        b_st.extend(b_slice.iter().map(|x| 3.0 * x));
        let a_nd = rt::asarray((a_st, [2, 3, 3].c(), &device));
        let b_nd = rt::asarray((b_st.clone(), [2, 3, 2].c(), &device));

        let x_nd = rt::linalg::solve_triangular((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.shape(), &[2, 3, 2]);
        assert!((fingerprint(&x_nd.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x_nd.i(1).to_owned()) - 1.5 * fp2).abs() < 1e-8);

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

        // distinct systems per slot: slice 1 is (2*a, 3*b), so x1 = 1.5 * x0
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let mut b_st = b_slice.clone();
        b_st.extend(b_slice.iter().map(|x| 3.0 * x));
        let a_nd = rt::asarray((a_st, [4, 4, 2], &device));
        let b_nd = rt::asarray((b_st, [4, 2, 2], &device));

        let x_nd = rt::linalg::solve_general((a_nd.view(), b_nd.view()));
        assert_eq!(x_nd.shape(), &[4, 2, 2]);
        assert!((fingerprint(&x_nd.i((.., .., 0)).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x_nd.i((.., .., 1)).to_owned()) - 1.5 * fp2).abs() < 1e-8);
    }

    #[test]
    fn test_solve_general_broadcast() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        let a_slice: Vec<f64> = vec![4.0, 1.0, 0.0, 0.5, 1.0, 3.0, 0.5, 0.0, 0.0, 0.5, 2.0, 1.0, 0.5, 0.0, 1.0, 2.5];
        let b_slice: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let a2 = rt::asarray((a_slice.clone(), [4, 4].c(), &device));
        let b2 = rt::asarray((b_slice.clone(), [4, 2].c(), &device));
        let x2 = rt::linalg::solve_general((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        // a's slot 1 is 2*a; b's slot 1 is 3*b, so x1 = 1.5 * x0 throughout
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let mut b_st = b_slice.clone();
        b_st.extend(b_slice.iter().map(|x| 3.0 * x));
        let a_nd = rt::asarray((a_st, [2, 4, 4].c(), &device));

        // b's batch is size 1 -> broadcast up to a's batch; result outgrows b
        let b_1 = rt::asarray((b_slice.clone(), [1, 4, 2].c(), &device));
        let x = rt::linalg::solve_general((a_nd.view(), b_1.view()));
        assert_eq!(x.shape(), &[2, 4, 2]);
        assert!((fingerprint(&x.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x.i(1).to_owned()) - 0.5 * fp2).abs() < 1e-8);

        // a unbatched: it broadcasts into b's batch
        let b_nd = rt::asarray((b_st.clone(), [2, 4, 2].c(), &device));
        let x = rt::linalg::solve_general((a2.view(), b_nd.view()));
        assert_eq!(x.shape(), &[2, 4, 2]);
        assert!((fingerprint(&x.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x.i(1).to_owned()) - 3.0 * fp2).abs() < 1e-8);

        // in-place: a unbatched broadcasting into an owned batched b
        let mut b_mut = rt::asarray((b_st, [2, 4, 2].c(), &device));
        rt::linalg::solve_general((a2.view(), b_mut.view_mut()));
        assert!((fingerprint(&b_mut.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&b_mut.i(1).to_owned()) - 3.0 * fp2).abs() < 1e-8);
    }

    #[test]
    fn test_solve_triangular_broadcast() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        let a_slice: Vec<f64> = vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.5, 0.5, 4.0];
        let b_slice: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a2 = rt::asarray((a_slice.clone(), [3, 3].c(), &device));
        let b2 = rt::asarray((b_slice.clone(), [3, 2].c(), &device));
        let x2 = rt::linalg::solve_triangular((a2.view(), b2.view()));
        let fp2 = fingerprint(&x2);

        // a's slot 1 is 2*a; b's slot 1 is 3*b, so x1 = 1.5 * x0 throughout
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let a_nd = rt::asarray((a_st, [2, 3, 3].c(), &device));

        // b's batch is size 1 -> broadcast up to a's batch
        let b_1 = rt::asarray((b_slice.clone(), [1, 3, 2].c(), &device));
        let x = rt::linalg::solve_triangular((a_nd.view(), b_1.view()));
        assert_eq!(x.shape(), &[2, 3, 2]);
        assert!((fingerprint(&x.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x.i(1).to_owned()) - 0.5 * fp2).abs() < 1e-8);

        // a unbatched: it broadcasts into b's batch
        let mut b_st = b_slice.clone();
        b_st.extend(b_slice.iter().map(|x| 3.0 * x));
        let b_nd = rt::asarray((b_st.clone(), [2, 3, 2].c(), &device));
        let x = rt::linalg::solve_triangular((a2.view(), b_nd.view()));
        assert_eq!(x.shape(), &[2, 3, 2]);
        assert!((fingerprint(&x.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&x.i(1).to_owned()) - 3.0 * fp2).abs() < 1e-8);

        // in-place: a unbatched broadcasting into an owned batched b
        let mut b_mut = rt::asarray((b_st, [2, 3, 2].c(), &device));
        rt::linalg::solve_triangular((a2.view(), b_mut.view_mut()));
        assert!((fingerprint(&b_mut.i(0).to_owned()) - fp2).abs() < 1e-8);
        assert!((fingerprint(&b_mut.i(1).to_owned()) - 3.0 * fp2).abs() < 1e-8);
    }

    #[test]
    fn test_generalized_eigh_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        // symmetric a, positive-definite b
        let a_slice: Vec<f64> = vec![4.0, 1.0, 1.0, 3.0];
        let b_slice: Vec<f64> = vec![2.0, 0.5, 0.5, 1.5];

        let a2 = rt::asarray((a_slice.clone(), [2, 2].c(), &device));
        let b2 = rt::asarray((b_slice.clone(), [2, 2].c(), &device));
        let (w2, v2) = rt::linalg::eigh((a2.view(), b2.view(), Lower, 1)).into();
        let fp_w2 = fingerprint(&w2);
        let fp_v2 = fingerprint(&v2.abs());

        // a's slot 1 is 2*a, b's batch size 1 -> broadcast to (2, 2, 2) / (2, 2);
        // the scaled slot's eigenvalues double, its eigenvectors do not
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let a_nd = rt::asarray((a_st, [2, 2, 2].c(), &device));
        let b_1 = rt::asarray((b_slice.clone(), [1, 2, 2].c(), &device));
        let (w, v) = rt::linalg::eigh((a_nd.view(), b_1.view(), Lower, 1)).into();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(v.shape(), &[2, 2, 2]);
        assert!((fingerprint(&w.i(0).to_owned()) - fp_w2).abs() < 1e-8);
        assert!((fingerprint(&w.i(1).to_owned()) - 2.0 * fp_w2).abs() < 1e-8);
        assert!((fingerprint(&v.i(0).to_owned().abs()) - fp_v2).abs() < 1e-6);
        assert!((fingerprint(&v.i(1).to_owned().abs()) - fp_v2).abs() < 1e-6);
    }

    #[test]
    fn test_generalized_eigh_nd_colmajor() {
        let mut device = DeviceFaer::default();
        device.set_default_order(ColMajor);
        // Under ColMajor the matrix axes are the first two and the batch trails.
        // Passing a shape (not an explicit `.c()` layout) makes `asarray` follow
        // the device order (F), so each `[:, :, b]` slice equals the 2-D operand.
        let a_slice: Vec<f64> = vec![4.0, 1.0, 1.0, 3.0];
        let b_slice: Vec<f64> = vec![2.0, 0.5, 0.5, 1.5];

        let a2 = rt::asarray((a_slice.clone(), [2, 2], &device));
        let b2 = rt::asarray((b_slice.clone(), [2, 2], &device));
        let (w2, v2) = rt::linalg::eigh((a2.view(), b2.view(), Lower, 1)).into();
        let fp_w2 = fingerprint(&w2);
        let fp_v2 = fingerprint(&v2.abs());

        // a's slot 1 is 2*a, b's batch size 1 -> broadcast up to a's batch
        let mut a_st = a_slice.clone();
        a_st.extend(a_slice.iter().map(|x| 2.0 * x));
        let a_nd = rt::asarray((a_st, [2, 2, 2], &device));
        let b_1 = rt::asarray((b_slice.clone(), [2, 2, 1], &device));
        let (w, v) = rt::linalg::eigh((a_nd.view(), b_1.view(), Lower, 1)).into();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(v.shape(), &[2, 2, 2]);
        assert!((fingerprint(&w.i((.., 0)).to_owned()) - fp_w2).abs() < 1e-8);
        assert!((fingerprint(&w.i((.., 1)).to_owned()) - 2.0 * fp_w2).abs() < 1e-8);
        assert!((fingerprint(&v.i((.., .., 0)).to_owned().abs()) - fp_v2).abs() < 1e-6);
        assert!((fingerprint(&v.i((.., .., 1)).to_owned().abs()) - fp_v2).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_power() {
        let device = DeviceFaer::default();
        let a_vec = get_vec::<f64>('a')[..16].to_vec();
        let a = rt::asarray((a_vec, [4, 4].c(), &device));

        // n = 0 -> identity, in the input dtype
        let p = rt::linalg::matrix_power((a.view(), 0));
        assert!((fingerprint(&p) - -0.3150972564720475).abs() < 1e-8);

        // n = 1 -> a itself
        let p = rt::linalg::matrix_power((a.view(), 1));
        assert!((fingerprint(&p) - -0.5537625095088874).abs() < 1e-8);

        // shortcuts (2, 3) and the squaring loop (> 3)
        for (n, golden) in [(2isize, -3.3066512682104), (3, 4.312505984257974), (5, -2.904867370507913)] {
            let p = rt::linalg::matrix_power((a.view(), n));
            assert!((fingerprint(&p) - golden).abs() < 1e-8, "n = {n}");
        }

        // negative exponent: invert once, then raise the inverse to |n|
        for (n, golden) in [(-1isize, 1.5318824349546132), (-2, -4.11401929372348)] {
            let p = rt::linalg::matrix_power((a.view(), n));
            assert!((fingerprint(&p) - golden).abs() < 1e-8, "n = {n}");
        }
    }

    #[test]
    fn test_matrix_power_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        let a_vec = get_vec::<f64>('a')[..16].to_vec();

        // stack: slice 0 is a, slice 1 is 2*a
        let mut stacked = a_vec.clone();
        stacked.extend(a_vec.iter().map(|x| 2.0 * x));
        let b = rt::asarray((stacked, [2, 4, 4].c(), &device));

        let p = rt::linalg::matrix_power((b.view(), 0));
        assert_eq!(p.shape(), &[2, 4, 4]);
        assert!((fingerprint(&p) - -0.2588243168865289).abs() < 1e-8);

        let p = rt::linalg::matrix_power((b.view(), 2));
        assert!((fingerprint(&p) - 1.5180278106564264).abs() < 1e-8);

        let p = rt::linalg::matrix_power((b.view(), -1));
        assert!((fingerprint(&p) - 0.5342839334664118).abs() < 1e-8);
    }

    #[test]
    fn test_matrix_power_nd_colmajor() {
        let mut device = DeviceFaer::default();
        device.set_default_order(ColMajor);
        // Under ColMajor the matrix axes are the first two and the batch trails.
        // Passing a shape (not `.c()`) makes `asarray` follow the device order (F),
        // so each `[:, :, b]` slice equals the 2-D operand.
        let a_vec = get_vec::<f64>('a')[..16].to_vec();
        // F-order of the operand, so the 2-D and stacked inputs share one layout
        let mut va = Vec::with_capacity(16);
        for j in 0..4 {
            for i in 0..4 {
                va.push(a_vec[i * 4 + j]);
            }
        }
        let v2a: Vec<f64> = va.iter().map(|x| 2.0 * x).collect();
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

    #[test]
    fn test_matrix_norm() {
        let device = DeviceFaer::default();
        // NumPy's 2x2 case (numpy/linalg/tests/test_linalg.py: test_matrix_2x2)
        let a = rt::asarray((vec![1.0f64, 3.0, 5.0, 7.0], [2, 2].c(), &device));

        // the default order is 'fro'
        assert!((rt::linalg::matrix_norm(a.view()).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), "fro")).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), "nuc")).to_scalar() - 10.0).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), -1)).to_scalar() - 6.0).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), 2)).to_scalar() - 9.123105625617661).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), -2)).to_scalar() - 0.8768943743823404).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar() - 12.0).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), f64::NEG_INFINITY)).to_scalar() - 4.0).abs() < 1e-12);

        // keepdims
        let k = rt::linalg::matrix_norm((a.view(), ("fro", true)));
        assert_eq!(k.shape(), &[1, 1]);

        // orders a matrix has no norm for (NumPy raises for 0, 3 and an unknown string)
        assert!(rt::linalg::matrix_norm_f((a.view(), 0)).is_err());
        assert!(rt::linalg::matrix_norm_f((a.view(), 3)).is_err());
        assert!(rt::linalg::matrix_norm_f((a.view(), "nofro")).is_err());
    }

    #[test]
    fn test_matrix_norm_nd() {
        let mut device = DeviceFaer::default();
        device.set_default_order(RowMajor);
        // stack [a; 2a] of the 2x2 case
        let stacked = rt::asarray((vec![1.0f64, 3.0, 5.0, 7.0, 2.0, 6.0, 10.0, 14.0], [2, 2, 2].c(), &device));

        let n = rt::linalg::matrix_norm((stacked.view(), "nuc"));
        assert_eq!(n.shape(), &[2]);
        assert!((n[[0]] - 10.0).abs() < 1e-12);
        assert!((n[[1]] - 20.0).abs() < 1e-12);

        let f = rt::linalg::matrix_norm((stacked.view(), "fro"));
        assert!((f[[0]] - 84f64.sqrt()).abs() < 1e-12);
        assert!((f[[1]] - 2.0 * 84f64.sqrt()).abs() < 1e-12);

        let k = rt::linalg::matrix_norm((stacked.view(), ("nuc", true)));
        assert_eq!(k.shape(), &[2, 1, 1]);
    }

    #[test]
    fn test_matrix_norm_nd_colmajor() {
        let mut device = DeviceFaer::default();
        device.set_default_order(ColMajor);
        // logical A = [[1, 3], [5, 7]]; under ColMajor `asarray` fills in F order
        let a = rt::asarray((vec![1.0f64, 5.0, 3.0, 7.0], [2, 2], &device));
        assert!((rt::linalg::matrix_norm((a.view(), "fro")).to_scalar() - 84f64.sqrt()).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), "nuc")).to_scalar() - 10.0).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
        assert!((rt::linalg::matrix_norm((a.view(), f64::INFINITY)).to_scalar() - 12.0).abs() < 1e-12);

        // the matrix axes are the first two under ColMajor, the batch trails
        let stacked = rt::asarray((vec![1.0f64, 5.0, 3.0, 7.0, 2.0, 10.0, 6.0, 14.0], [2, 2, 2], &device));
        let n = rt::linalg::matrix_norm((stacked.view(), "nuc"));
        assert_eq!(n.shape(), &[2]);
        assert!((n[[0]] - 10.0).abs() < 1e-12);
        assert!((n[[1]] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_vector_norm() {
        let device = DeviceFaer::default();
        // NumPy's vector case (numpy/linalg/tests/test_linalg.py: test_vector)
        let v = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], [4].c(), &device));

        assert!((rt::linalg::vector_norm(v.view()).to_scalar() - 30f64.sqrt()).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), 2)).to_scalar() - 30f64.sqrt()).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), 1)).to_scalar() - 10.0).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), 0)).to_scalar() - 4.0).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), -1)).to_scalar() - 12.0 / 25.0).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), -2)).to_scalar() - (205f64 / 144.0).powf(-0.5)).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), 3)).to_scalar() - 100f64.powf(1.0 / 3.0)).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), f64::INFINITY)).to_scalar() - 4.0).abs() < 1e-12);
        assert!((rt::linalg::vector_norm((v.view(), f64::NEG_INFINITY)).to_scalar() - 1.0).abs() < 1e-12);

        // 'fro'/'nuc' are matrix orders, rejected for vectors
        assert!(rt::linalg::vector_norm_f((v.view(), "fro")).is_err());
        assert!(rt::linalg::vector_norm_f((v.view(), "nuc")).is_err());
    }

    #[test]
    fn test_vector_norm_axis() {
        let device = DeviceFaer::default();
        // NumPy: x = arange(9).reshape(3, 3)
        let x = rt::asarray((vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [3, 3].c(), &device));

        // axis = 0 -> column-wise 2-norms
        let n = rt::linalg::vector_norm((x.view(), (2, AxesIndex::Val(0))));
        assert_eq!(n.shape(), &[3]);
        assert!((n[[0]] - 45f64.sqrt()).abs() < 1e-12);
        assert!((n[[1]] - 66f64.sqrt()).abs() < 1e-12);
        assert!((n[[2]] - 93f64.sqrt()).abs() < 1e-12);

        // axis = None (the default) flattens; keepdims keeps both axes
        assert!((rt::linalg::vector_norm(x.view()).to_scalar() - 204f64.sqrt()).abs() < 1e-12);
        let k = rt::linalg::vector_norm((x.view(), (rt::linalg::NormOrd::None, AxesIndex::None, true)));
        assert_eq!(k.shape(), &[1, 1]);
        assert!((k.to_scalar() - 204f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_norm_empty() {
        // NumPy: every order of an empty matrix / vector is 0
        // (numpy/linalg/tests/test_linalg.py: test_matrix_norm_empty / test_vector_norm_empty)
        let device = DeviceFaer::default();
        let z = rt::asarray((vec![0.0f64; 0], [0, 2].c(), &device));
        for ord in ["fro", "nuc"] {
            assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
        }
        for ord in [1.0f64, 2.0, f64::INFINITY] {
            assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
        }
        // the min-based orders too: NumPy itself is inconsistent on an empty
        // input (it raises for the matrix -2 and for -inf, and gives `inf` for
        // the vector -1); rstsr returns 0 under every order
        for ord in [-1.0f64, -2.0, f64::NEG_INFINITY] {
            assert_eq!(rt::linalg::matrix_norm((z.view(), ord)).to_scalar(), 0.0);
        }

        let zv = rt::asarray((vec![0.0f64; 0], [0].c(), &device));
        for ord in [1.0f64, 2.0, f64::INFINITY] {
            assert_eq!(rt::linalg::vector_norm((zv.view(), ord)).to_scalar(), 0.0);
        }
        for ord in [-1.0f64, f64::NEG_INFINITY] {
            assert_eq!(rt::linalg::vector_norm((zv.view(), ord)).to_scalar(), 0.0);
        }

        // an empty batch is empty (NumPy: shape (0,)), not an error
        let eb = rt::asarray((vec![0.0f64; 0], [0, 1, 1].c(), &device));
        assert_eq!(rt::linalg::matrix_norm((eb.view(), f64::NEG_INFINITY)).shape(), &[0]);
        let eb2 = rt::asarray((vec![0.0f64; 0], [0, 3].c(), &device));
        assert_eq!(rt::linalg::vector_norm((eb2.view(), (2, AxesIndex::Val(1)))).shape(), &[0]);
    }
}
