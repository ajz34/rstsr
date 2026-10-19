use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

fn device() -> DeviceOpenBLAS {
    DeviceOpenBLAS::default()
}

/* #region TRSM / solve_triangular (issue #161) */

#[test]
fn trsm_issue_161_static_f_reversed() {
    let device = device();
    let naux = 18usize;
    let buf: Vec<f64> = Vec::new();

    // shape [18, 0], degenerate strides [1, 1] before any fix
    let b = rt::asarray((&buf[..], [0, naux].f(), &device)).into_reverse_axes();
    assert_eq!(b.shape()[0], naux);
    assert_eq!(b.shape()[1], 0);

    let out = rt::linalg::solve_triangular((rt::eye((naux, &device)).t(), b, Lower));
    assert_eq!(out.shape()[0], naux);
    assert_eq!(out.shape()[1], 0);
}

#[test]
fn trsm_issue_161_dynamic_row_major_empty() {
    let device = device();
    let b: Tensor<f64, DeviceOpenBLAS, IxD> = rt::zeros((vec![18usize, 0], RowMajor, &device));

    let out = rt::linalg::solve_triangular((rt::eye((18usize, &device)).t(), b, Lower));
    assert_eq!(out.shape()[0], 18);
    assert_eq!(out.shape()[1], 0);
}

#[test]
fn trsm_issue_161_empty_view() {
    let device = device();
    let base: Tensor<f64, DeviceOpenBLAS, IxD> = rt::zeros((vec![5usize, 18], ColMajor, &device));
    let b = base.i((0..0, ..)).into_dim::<IxD>().into_reverse_axes();

    let out = rt::linalg::solve_triangular((rt::eye((18usize, &device)).t(), b, Lower));
    assert_eq!(out.shape()[0], 18);
    assert_eq!(out.shape()[1], 0);
}

#[test]
fn trsm_zero_rows() {
    let device = device();
    let b: Tensor<f64, DeviceOpenBLAS, IxD> = rt::asarray((Vec::<f64>::new(), [0usize, 18].f(), &device));

    let out = rt::linalg::solve_triangular((rt::eye((0usize, &device)).t(), b, Lower));
    assert_eq!(out.shape()[0], 0);
    assert_eq!(out.shape()[1], 18);
}

/* #endregion */

/* #region GEMM */

#[test]
fn gemm_empty_m() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((Vec::<f64>::new(), [0usize, 4].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((vec![1.0f64; 12], [4usize, 3].f(), &device)).into_dim::<Ix2>();

    let c = GEMM::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap().into_owned();
    assert_eq!(c.shape()[0], 0);
    assert_eq!(c.shape()[1], 3);
}

#[test]
fn gemm_empty_n() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((vec![1.0f64; 8], [4usize, 2].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((Vec::<f64>::new(), [2usize, 0].f(), &device)).into_dim::<Ix2>();

    let c = GEMM::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap().into_owned();
    assert_eq!(c.shape()[0], 4);
    assert_eq!(c.shape()[1], 0);
}

#[test]
fn gemm_k_zero_scales_c_by_beta() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((Vec::<f64>::new(), [2usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::asarray((Vec::<f64>::new(), [0usize, 3].f(), &device)).into_dim::<Ix2>();
    let mut c: Tensor<f64, DeviceOpenBLAS, Ix2> =
        rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).into_dim::<Ix2>();

    let c =
        GEMM::default().a(a.view()).b(b.view()).c(c.view_mut()).beta(2.0).build().unwrap().run().unwrap().into_owned();

    let expect = rt::tensor_from_nested!([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], &device).into_dim::<Ix2>();
    assert!(rt::allclose(&c, &expect, None), "c = {c:?}");
}

/* #endregion */

/* #region SYHEMM */

#[test]
fn syhemm_empty_b_side_left() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([3usize, 3].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([3usize, 0].f(), &device)).into_dim::<Ix2>();

    let c = DSYMM::default().a(a.view()).b(b.view()).side(FlagSide::L).build().unwrap().run().unwrap().into_owned();
    assert_eq!(c.shape()[0], 3);
    assert_eq!(c.shape()[1], 0);
}

#[test]
fn syhemm_empty_b_side_right() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([3usize, 0].f(), &device)).into_dim::<Ix2>();

    let c = DSYMM::default().a(a.view()).b(b.view()).side(FlagSide::R).build().unwrap().run().unwrap().into_owned();
    assert_eq!(c.shape()[0], 3);
    assert_eq!(c.shape()[1], 0);
}

/* #endregion */
