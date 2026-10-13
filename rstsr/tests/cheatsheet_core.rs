//! Runnable counterpart of the core-operations cheat sheet
//! (`rstsr/src/docs/cheatsheet.md`). Every row of that page has a matching
//! example id here; keep the two in sync (skill `api-cheatsheet-conventions`).

use rstsr::prelude::*;

fn device() -> DeviceFaer {
    let mut device = DeviceFaer::default();
    device.set_default_order(RowMajor);
    device
}

#[test]
fn creation_special() {
    let device = device();

    let z: Tensor<f64, DeviceFaer, _> = rt::zeros(([2, 3], &device));
    assert_eq!(z.sum_all(), 0.0);
    let o: Tensor<f64, DeviceFaer, _> = rt::ones(([2, 3], &device));
    assert_eq!(o.sum_all(), 6.0);
    let f = rt::full(([2, 2], 7, &device));
    assert_eq!(f.reshape(-1).to_vec(), vec![7, 7, 7, 7]);
    let e: Tensor<i32, DeviceFaer, _> = rt::eye((3, &device));
    assert_eq!(rt::trace(&e, ()).to_scalar(), 3);

    let a = rt::arange((6.0, &device));
    assert_eq!(a.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let l = rt::linspace((0.0, 1.0, 5, &device));
    assert_eq!(l.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn creation_from_lists() {
    let device = device();

    let owned = rt::asarray((vec![0, 1, 2, 3, 4, 5], &device));
    assert_eq!(owned.shape(), &[6]);

    let v2 = vec![0, 1, 2, 3, 4, 5];
    let view = rt::asarray((&v2, &device));
    assert_eq!(view.to_vec(), vec![0, 1, 2, 3, 4, 5]);

    let m = rt::asarray((vec![0, 1, 2, 3, 4, 5], [2, 3], &device));
    assert_eq!(m.shape(), &[2, 3]);

    let n = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
    assert_eq!(n.shape(), &[2, 2]);
}

#[test]
fn indexing_basic() {
    let device = device();
    let m = rt::arange((12, &device)).into_shape([3, 4]);

    assert_eq!(m.i((.., 3)).to_vec(), vec![3, 7, 11]);
    assert_eq!(m.i((0..2, 2..4)).reshape(-1).to_vec(), vec![2, 3, 6, 7]);
    assert_eq!(m.i((Ellipsis, -1)).to_vec(), vec![3, 7, 11]);
    assert_eq!(m.i((.., None)).shape(), &[3, 1, 4]);
    let v = rt::arange((6, &device));
    assert_eq!(v.i(slice!(1, 6, 2)).to_vec(), vec![1, 3, 5]);
    assert_eq!(v.i(slice!(None, None, -1)).to_vec(), vec![5, 4, 3, 2, 1, 0]);
}

#[test]
fn indexing_array() {
    let device = device();
    let m = rt::arange((12, &device)).into_shape([3, 4]);

    // integer-array indexing
    assert_eq!(rt::array_index(&m, [2, 0]).i(0).to_vec(), vec![8, 9, 10, 11]);
    assert_eq!(rt::array_index(&m, ([2, 0], [1, 3])).to_vec(), vec![9, 3]);
    assert_eq!(rt::array_index(&m, ([2, 0], 1)).to_vec(), vec![9, 1]);
    assert_eq!(rt::index_select(&m, 1, [3, 1]).i(0).to_vec(), vec![3, 1]);
    let idx = rt::tensor_from_nested!([[3], [1], [0]], &device);
    assert_eq!(rt::take_along_axis(&m, &idx, -1).shape(), &[3, 1]);

    // boolean-array indexing
    let mask = rt::tensor_from_nested!(
        [[true, false, true, false], [false, false, false, true], [false, false, true, false]],
        &device
    );
    assert_eq!(rt::mask_select(&m, &mask).to_vec(), vec![0, 2, 7, 10]);
    assert_eq!(rt::bool_select(&m, 1, [true, false, true, false]).shape(), &[3, 2]);
}

#[test]
fn manipulation_reshape_and_layout() {
    let device = device();
    let a = rt::arange((24, &device));

    assert_eq!(a.reshape((2, 3, 4)).shape(), &[2, 3, 4]);
    assert_eq!(a.reshape(-1).shape(), &[24]);
    let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
    assert_eq!(b.transpose((2, 0, 1)).shape(), &[4, 2, 3]);
    assert_eq!(b.t().shape(), &[4, 3, 2]);
    assert_eq!(b.swapaxes(0, 2).shape(), &[4, 3, 2]);
    assert_eq!(b.moveaxis(0, 2).shape(), &[3, 4, 2]);
    assert_eq!(b.expand_dims(0).shape(), &[1, 2, 3, 4]);
    assert_eq!(rt::tensor_from_nested!([[1, 2, 3]], &device).squeeze(0).shape(), &[3]);
    assert_eq!(b.flip(0).i(0).reshape(-1).to_vec(), b.i(1).reshape(-1).to_vec());
}

#[test]
fn manipulation_joining() {
    let device = device();
    let a: Tensor<i32, DeviceFaer, _> = rt::arange((3, &device));
    let b: Tensor<i32, DeviceFaer, _> = rt::arange((3, 6, &device));

    assert_eq!(rt::concat([&a, &b]).to_vec(), vec![0, 1, 2, 3, 4, 5]);
    let m: Tensor<i32, DeviceFaer, _> = rt::arange((6, &device)).into_shape([2, 3]);
    let n: Tensor<i32, DeviceFaer, _> = rt::full(([2, 2], 9, &device));
    let cc: Tensor<i32, DeviceFaer, IxD> = rt::concat(([&m, &n], 1));
    assert_eq!(cc.shape(), &[2, 5]);
    assert_eq!(rt::stack([&a, &a]).shape(), &[2, 3]);

    let t = rt::tensor_from_nested!([1, 2], &device);
    assert_eq!(rt::tile(&t, [2, 2]).reshape(-1).to_vec(), vec![1, 2, 1, 2, 1, 2, 1, 2]);
    assert_eq!(rt::repeat(&t, (2, None)).to_vec(), vec![1, 1, 2, 2]);
    assert_eq!(rt::roll(&a, (2, None)).to_vec(), vec![1, 2, 0]);

    let sq = rt::arange((9, &device)).into_shape([3, 3]);
    assert_eq!(rt::trace(&sq, ()).to_scalar(), 12);
}

#[test]
fn operations_operators() {
    let device = device();
    let a = rt::tensor_from_nested!([1, 2, 3], &device);
    let b = rt::tensor_from_nested!([4, 5, 6], &device);

    assert_eq!((&a + &b).to_vec(), vec![5, 7, 9]);
    assert_eq!((&a * 2).to_vec(), vec![2, 4, 6]);
    assert_eq!(rt::rem(&a, &b).to_vec(), vec![1, 2, 3]);

    // `%` is matrix multiplication
    let m: Tensor<i32, DeviceFaer, _> = rt::arange((6, &device)).into_shape([2, 3]);
    let p: Tensor<i32, DeviceFaer, _> = rt::arange((6, &device)).into_shape([3, 2]);
    let c = &m % &p;
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.reshape(-1).to_vec(), vec![10, 13, 28, 40]);
}

#[test]
fn operations_elementwise() {
    let device = device();
    let a = rt::tensor_from_nested!([1.0, 4.0, 9.0], &device);

    assert_eq!(rt::sqrt(&a).to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(rt::clip(&a, (2, 5)).to_vec(), vec![2.0, 4.0, 5.0]);
    assert_eq!(rt::greater(&a, 3.0).to_vec(), vec![false, true, true]);
    let cond = rt::greater(&a, 3.0);
    let y = rt::tensor_from_nested!([10.0, 20.0, 30.0], &device);
    assert_eq!(rt::r#where(&cond, &y, &a).to_vec(), vec![1.0, 20.0, 30.0]);
    assert_eq!(a.mapv(|x| x + 1.0).to_vec(), vec![2.0, 5.0, 10.0]);
}

#[test]
fn reduction_basic() {
    let device = device();
    let a = rt::arange((24.0, &device)).into_shape([2, 3, 4]);

    assert_eq!(rt::sum(&a), 276.0);
    assert_eq!(rt::sum_axes(&a, -1).shape(), &[2, 3]);
    assert_eq!(rt::sum_axes(&a, (0, 2)).shape(), &[3]);
    assert_eq!(rt::sum_with_args(&a, (-1, true)).shape(), &[2, 3, 1]);
    assert_eq!(rt::mean(&a), 11.5);
    assert_eq!(rt::max(&a), 23.0);
    assert_eq!(rt::min(&a), 0.0);
    assert_eq!(rt::argmax(&a), 23);
    assert_eq!(rt::count_nonzero(&a), 23);
    let v = rt::arange((4.0, &device));
    assert_eq!(rt::cumulative_sum(&v, None).to_vec(), vec![0.0, 1.0, 3.0, 6.0]);
    assert!(rt::all(rt::greater(&a, -1.0)));
}

#[test]
fn sorting_and_set() {
    let device = device();
    let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);

    assert_eq!(rt::sort(&a, None).to_vec(), vec![1, 1, 2, 3, 3]);
    assert_eq!(rt::argsort(&a, None).to_vec(), vec![1, 4, 3, 0, 2]);
    assert_eq!(rt::unique_values(&a).to_vec(), vec![1, 2, 3]);
    let test = rt::tensor_from_nested!([1, 3], &device);
    assert_eq!(rt::isin(&a, &test, false).to_vec(), vec![true, true, true, false, true]);

    let sorted = rt::tensor_from_nested!([1, 3, 5, 7], &device);
    let v = rt::tensor_from_nested!([2, 6], &device);
    assert_eq!(rt::searchsorted(&sorted, &v, None).to_vec(), vec![1, 3]);

    let d = rt::tensor_from_nested!([1, 4, 9, 16], &device);
    assert_eq!(rt::diff(&d, -1, 1, None, None).to_vec(), vec![3, 5, 7]);
}

#[test]
fn compare_allclose() {
    let device = device();
    let a = rt::tensor_from_nested!([1.0, 2.0, 3.0], &device);
    let b = rt::tensor_from_nested!([1.0, 2.0, 3.0 + 1e-12], &device);
    assert!(rt::allclose(&a, &b, None));
}
