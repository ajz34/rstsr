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

    /// Two SPD 3x3 matrices and their row-major stack (shape `[2, 3, 3]`).
    fn spd_stack(
        device: &DeviceFaer,
    ) -> (Tensor<f64, DeviceFaer, IxD>, Tensor<f64, DeviceFaer, IxD>, Tensor<f64, DeviceFaer, IxD>) {
        let m0 = rt::asarray((vec![4.0f64, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0], [3, 3].c(), device));
        let m1 = rt::asarray((vec![2.0f64, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 3.0], [3, 3].c(), device));
        let a = rt::stack((vec![m0.clone(), m1.clone()], 0));
        (m0, m1, a)
    }

    #[test]
    fn test_batched_linalg() {
        let device = DeviceFaer::default();
        let (m0, m1, a) = spd_stack(&device);
        // det: batch shape (2,)
        let d = rt::linalg::det(a.view());
        assert_eq!(d.shape(), &[2]);
        assert!((d.i(0).to_scalar() - rt::linalg::det(m0.view()).to_scalar()).abs() < 1e-10);
        assert!((d.i(1).to_scalar() - rt::linalg::det(m1.view()).to_scalar()).abs() < 1e-10);

        // cholesky / inv: stack of matrices, each slice matches the 2-D result
        let c = rt::linalg::cholesky(a.view());
        assert_eq!(c.shape(), &[2, 3, 3]);
        assert!((fingerprint(&c.i(0).into_owned()) - fingerprint(&rt::linalg::cholesky(m0.view()))).abs() < 1e-10);
        let ai = rt::linalg::inv(a.view());
        assert_eq!(ai.shape(), &[2, 3, 3]);
        assert!((fingerprint(&ai.i(1).into_owned()) - fingerprint(&rt::linalg::inv(m1.view()))).abs() < 1e-8);

        // eigvalsh / svdvals: batch of vectors
        let w = rt::linalg::eigvalsh(a.view());
        assert_eq!(w.shape(), &[2, 3]);
        assert!((fingerprint(&w.i(0).into_owned()) - fingerprint(&rt::linalg::eigvalsh(m0.view()))).abs() < 1e-8);
        let s = rt::linalg::svdvals(a.view());
        assert_eq!(s.shape(), &[2, 3]);

        // eigh / svd: multiple stacked outputs
        let e = rt::linalg::eigh(a.view());
        assert_eq!(e.eigenvalues.shape(), &[2, 3]);
        assert_eq!(e.eigenvectors.shape(), &[2, 3, 3]);
        let (u, sv, vt) = rt::linalg::svd((a.view(), true)).into();
        assert_eq!(u.shape(), &[2, 3, 3]);
        assert_eq!(sv.shape(), &[2, 3]);
        assert_eq!(vt.shape(), &[2, 3, 3]);

        // pinv: same shape as input for square stacks
        let p = rt::linalg::pinv(a.view()).pinv;
        assert_eq!(p.shape(), &[2, 3, 3]);

        // solve: a (..., M, M) with b (..., M, K) -> (..., M, K)
        let b =
            rt::asarray((vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [2, 3, 2].c(), &device));
        let x = rt::linalg::solve_general((a.view(), b.view()));
        assert_eq!(x.shape(), &[2, 3, 2]);
        // first slice solves m0 x0 = b0
        let b0 = b.i(0);
        let x0 = rt::linalg::solve_general((m0.view(), b0.view()));
        assert!((fingerprint(&x.i(0).into_owned()) - fingerprint(&x0)).abs() < 1e-8);

        // solve broadcast: a batch (2,), b batch (1,) -> (2, 3, 2)
        let b1 = rt::asarray((vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0], [1, 3, 2].c(), &device));
        let xb = rt::linalg::solve_general((a.view(), b1.view()));
        assert_eq!(xb.shape(), &[2, 3, 2]);

        // in-place: a mutable / owned b is solved into its own buffer (no copy)
        let bd = vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut bv = rt::asarray((bd.clone(), [2, 3, 2].c(), &device));
        rt::linalg::solve_general((a.view(), bv.view_mut()));
        assert!((fingerprint(&bv) - fingerprint(&x)).abs() < 1e-8);
        let bo = rt::asarray((bd, [2, 3, 2].c(), &device));
        let ret = rt::linalg::solve_general((a.view(), bo));
        assert_eq!(ret.shape(), &[2, 3, 2]);
        assert!((fingerprint(&ret) - fingerprint(&x)).abs() < 1e-8);

        // solve_triangular: allocating and in-place, same batch walk as solve
        let x_tri = rt::linalg::solve_triangular((a.view(), b.view()));
        assert_eq!(x_tri.shape(), &[2, 3, 2]);
        let x0_tri = rt::linalg::solve_triangular((m0.view(), b.i(0).view()));
        assert!((fingerprint(&x_tri.i(0).into_owned()) - fingerprint(&x0_tri)).abs() < 1e-8);
        let mut bv_tri =
            rt::asarray((vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [2, 3, 2].c(), &device));
        rt::linalg::solve_triangular((a.view(), bv_tri.view_mut()));
        assert!((fingerprint(&bv_tri) - fingerprint(&x_tri)).abs() < 1e-8);

        // generalized eigh over a pair of stacks
        let ge = rt::linalg::eigh((a.view(), a.view()));
        assert_eq!(ge.eigenvalues.shape(), &[2, 3]);
        assert_eq!(ge.eigenvectors.shape(), &[2, 3, 3]);
        let ge0 = rt::linalg::eigh((m0.view(), m0.view()));
        assert!((fingerprint(&ge.eigenvalues.i(0).into_owned()) - fingerprint(&ge0.eigenvalues)).abs() < 1e-8);
    }

    #[test]
    fn test_batched_linalg_col_major() {
        let mut device = DeviceFaer::default();
        device.set_default_order(ColMajor);
        // col-major: matrix axes lead, batch trails -> (3, 3, 2)
        let m0 = rt::asarray((vec![4.0f64, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0], [3, 3].c(), &device));
        let m1 = rt::asarray((vec![2.0f64, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 3.0], [3, 3].c(), &device));
        let a: Tensor<f64, DeviceFaer, IxD> = rt::stack((vec![m0.clone(), m1.clone()], -1isize));
        assert_eq!(a.shape(), &[3, 3, 2]);

        let d = rt::linalg::det(a.view());
        assert_eq!(d.shape(), &[2]);
        assert!((d.i(0).to_scalar() - rt::linalg::det(m0.view()).to_scalar()).abs() < 1e-10);

        let c = rt::linalg::cholesky(a.view());
        assert_eq!(c.shape(), &[3, 3, 2]);
        assert!(
            (fingerprint(&c.i((.., .., 0)).into_owned()) - fingerprint(&rt::linalg::cholesky(m0.view()))).abs() < 1e-10
        );

        let s = rt::linalg::svdvals(a.view());
        assert_eq!(s.shape(), &[3, 2]);
    }
}
