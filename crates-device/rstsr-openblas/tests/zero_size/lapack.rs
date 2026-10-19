use rstsr::prelude::*;
use rstsr_blas_traits::prelude::*;

fn device() -> DeviceOpenBLAS {
    DeviceOpenBLAS::default()
}

/* #region GESV / SYSV */

fn assert_gesv_zero_rhs_is_noop(a_in: &Tensor<f64, DeviceOpenBLAS, Ix2>) {
    let device = device();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([a_in.nrow(), 0usize].f(), &device)).into_dim::<Ix2>();
    let (a, ipiv, b) = DGESV::default().a(a_in.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert!(rt::allclose(&a, a_in, None), "A must be returned unchanged: {a:?}");
    assert_eq!(ipiv.size(), 0);
    assert_eq!(b.shape(), &[a_in.nrow(), 0]);
}

fn assert_sysv_zero_rhs_is_noop(a_in: &Tensor<f64, DeviceOpenBLAS, Ix2>) {
    let device = device();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([a_in.nrow(), 0usize].f(), &device)).into_dim::<Ix2>();
    let (a, ipiv, b) = DSYSV::default().a(a_in.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert!(rt::allclose(&a, a_in, None), "A must be returned unchanged: {a:?}");
    assert_eq!(ipiv.size(), 0);
    assert_eq!(b.shape(), &[a_in.nrow(), 0]);
}

#[test]
fn gesv_empty_n() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();

    let (a, ipiv, b) = DGESV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert_eq!(a.shape(), &[0, 0]);
    assert_eq!(b.shape(), &[0, 0]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn gesv_empty_n_nonempty_rhs() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 3].f(), &device())).into_dim::<Ix2>();

    let (a, ipiv, b) = DGESV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert_eq!(a.shape(), &[0, 0]);
    assert_eq!(b.shape(), &[0, 3]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn gesv_zero_rhs_is_noop() {
    let device = device();
    let a = rt::tensor_from_nested!([[4.0f64, 3.0], [6.0, 3.0]], &device).into_dim::<Ix2>();
    assert_gesv_zero_rhs_is_noop(&a);
}

#[test]
fn gesv_zero_rhs_singular_is_noop() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0f64, 2.0], [2.0, 4.0]], &device).into_dim::<Ix2>();
    assert_gesv_zero_rhs_is_noop(&a);
}

#[test]
fn sysv_empty_n() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();

    let (a, ipiv, b) = DSYSV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let (a, b) = (a.into_owned(), b.into_owned());
    assert_eq!(a.shape(), &[0, 0]);
    assert_eq!(b.shape(), &[0, 0]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn sysv_zero_rhs_is_noop() {
    let device = device();
    let a = rt::tensor_from_nested!([[4.0f64, 3.0], [3.0, 3.0]], &device).into_dim::<Ix2>();
    assert_sysv_zero_rhs_is_noop(&a);
}

#[test]
fn sysv_zero_rhs_singular_is_noop() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0f64, 2.0], [2.0, 4.0]], &device).into_dim::<Ix2>();
    assert_sysv_zero_rhs_is_noop(&a);
}

/* #endregion */

/* #region GETRF / GETRI / POTRF */

#[test]
fn getrf_empty_rows() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 3].f(), &device())).into_dim::<Ix2>();
    let (a, ipiv) = DGETRF::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[0, 3]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn getrf_empty_cols() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([3usize, 0].f(), &device())).into_dim::<Ix2>();
    let (a, ipiv) = DGETRF::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[3, 0]);
    assert_eq!(ipiv.size(), 0);
}

#[test]
fn getri_empty() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let ipiv: Tensor<blas_int, DeviceOpenBLAS, Ix1> = rt::zeros(([0usize].f(), &device)).into_dim::<Ix1>();

    let a = DGETRI::default().a(a.view()).ipiv(ipiv.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn potrf_empty() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let a = DPOTRF::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(a.shape(), &[0, 0]);
}

/* #endregion */

/* #region EIGH */

#[test]
fn syev_empty() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let (w, a) = DSYEV::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn syevd_empty() {
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device())).into_dim::<Ix2>();
    let (w, a) = DSYEVD::default().a(a.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn sygv_empty() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let (w, a) = DSYGV::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

#[test]
fn sygvd_empty() {
    let device = device();
    let a: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let b: Tensor<f64, DeviceOpenBLAS, Ix2> = rt::zeros(([0usize, 0].f(), &device)).into_dim::<Ix2>();
    let (w, a) = DSYGVD::default().a(a.view()).b(b.view()).build().unwrap().run().unwrap();
    let a = a.into_owned();
    assert_eq!(w.shape(), &[0]);
    assert_eq!(a.shape(), &[0, 0]);
}

/* #endregion */

/* #region SVD */

/// Empty operand with degenerate strides `[1, 1]`: the leading dimension must
/// not be derived from the layout before the empty shortcut.
fn degenerate_empty(device: &DeviceOpenBLAS) -> Tensor<f64, DeviceOpenBLAS, Ix2> {
    let buf: Vec<f64> = Vec::new();
    rt::asarray((buf, [5usize, 0].f(), device)).into_reverse_axes().into_dim::<Ix2>()
}

#[test]
fn gesvd_empty_degenerate_strides() {
    let device = device();
    let a = degenerate_empty(&device);
    assert_eq!(a.shape(), &[0, 5]);

    let (s, u, vt, superb) = DGESVD::default().a(a.view()).build().unwrap().run().unwrap();
    assert_eq!(s.shape(), &[0]);
    assert_eq!(u.unwrap().shape(), &[0, 0]);
    assert_eq!(vt.unwrap().shape(), &[5, 5]);
    assert_eq!(superb.shape(), &[0]);
}

#[test]
fn gesdd_empty_degenerate_strides() {
    let device = device();
    let a = degenerate_empty(&device);
    assert_eq!(a.shape(), &[0, 5]);

    let (s, u, vt) = DGESDD::default().a(a.view()).build().unwrap().run().unwrap();
    assert_eq!(s.shape(), &[0]);
    assert_eq!(u.unwrap().shape(), &[0, 0]);
    assert_eq!(vt.unwrap().shape(), &[5, 5]);
}

/* #endregion */
