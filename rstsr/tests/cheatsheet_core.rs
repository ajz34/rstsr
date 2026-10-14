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
    assert_eq!(e.trace(()).to_scalar(), 3);

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
fn creation_from_existing() {
    let device = device();
    let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    assert_eq!(a.zeros_like().sum(), 0.0);
    assert_eq!(a.ones_like().sum(), 6.0);
    assert_eq!((&a).tril().reshape(-1).to_vec(), vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0]);
    assert_eq!((&a).triu().reshape(-1).to_vec(), vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0]);
    assert_eq!((&a).diag().to_vec(), vec![1.0, 5.0]);
    let v = rt::tensor_from_nested!([1.0, 2.0, 3.0], &device);
    assert_eq!(v.diag().shape(), &[3, 3]);
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

    // write through a mutable view
    let mut w = rt::arange((12, &device)).into_shape([3, 4]);
    w.i_mut((.., 3)).assign(&rt::tensor_from_nested!([100, 200, 300], &device));
    assert_eq!(w.i((.., 3)).to_vec(), vec![100, 200, 300]);
    let mut u = rt::arange((12, &device)).into_shape([3, 4]);
    *&mut u.i_mut(0..2) += &rt::tensor_from_nested!([[1, 1, 1, 1], [1, 1, 1, 1]], &device);
    assert_eq!(u.i(0).to_vec(), vec![1, 2, 3, 4]);
    let mut t = rt::arange((12, &device)).into_shape([3, 4]);
    t.i_mut(2..).add_assign(&rt::tensor_from_nested!([10, 10, 10, 10], &device));
    assert_eq!(t.i(2).to_vec(), vec![18, 19, 20, 21]);
}

#[test]
fn indexing_array() {
    let device = device();
    let m = rt::arange((12, &device)).into_shape([3, 4]);

    // integer-array indexing
    assert_eq!(m.array_index([2, 0]).i(0).to_vec(), vec![8, 9, 10, 11]);
    assert_eq!(m.array_index(([2, 0], [1, 3])).to_vec(), vec![9, 3]);
    assert_eq!(m.array_index(([2, 0], 1)).to_vec(), vec![9, 1]);
    assert_eq!(m.array_index((.., [3, 1])).i(0).to_vec(), vec![3, 1]);
    assert_eq!(m.index_select(1, [3, 1]).i(0).to_vec(), vec![3, 1]);
    let idx = rt::tensor_from_nested!([[3], [1], [0]], &device);
    assert_eq!(m.take_along_axis(&idx, -1).shape(), &[3, 1]);

    // boolean-array indexing
    let mask = rt::tensor_from_nested!(
        [[true, false, true, false], [false, false, false, true], [false, false, true, false]],
        &device
    );
    assert_eq!(m.mask_select(&mask).to_vec(), vec![0, 2, 7, 10]);
    assert_eq!(m.array_index(&mask).to_vec(), vec![0, 2, 7, 10]);
    let prefix = rt::tensor_from_nested!([true, false, true], &device);
    assert_eq!(m.array_index(&prefix).shape(), &[2, 4]);
    assert_eq!(m.bool_select(1, [true, false, true, false]).shape(), &[3, 2]);
    let col_mask = rt::tensor_from_nested!([true, false, true, false], &device);
    assert_eq!(m.array_index((.., &col_mask)).shape(), &[3, 2]);
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
    assert_eq!(t.tile([2, 2]).reshape(-1).to_vec(), vec![1, 2, 1, 2, 1, 2, 1, 2]);
    assert_eq!(t.repeat((2, None)).to_vec(), vec![1, 1, 2, 2]);
    assert_eq!(a.roll((2, None)).to_vec(), vec![1, 2, 0]);

    let sq = rt::arange((9, &device)).into_shape([3, 3]);
    assert_eq!(sq.trace(()).to_scalar(), 12);
}

#[test]
fn operations_operators() {
    let device = device();
    let a = rt::tensor_from_nested!([1, 2, 3], &device);
    let b = rt::tensor_from_nested!([4, 5, 6], &device);

    assert_eq!((&a + &b).to_vec(), vec![5, 7, 9]);
    assert_eq!((&a * 2).to_vec(), vec![2, 4, 6]);
    // `.rem` is the `%` matmul; the elementwise remainder is the free function only
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

    assert_eq!((&a).sqrt().to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!((&a).clip((2, 5)).to_vec(), vec![2.0, 4.0, 5.0]);
    assert_eq!((&a).greater(3.0).to_vec(), vec![false, true, true]);
    let cond = (&a).greater(3.0);
    let y = rt::tensor_from_nested!([10.0, 20.0, 30.0], &device);
    assert_eq!(cond.r#where(&y, &a).to_vec(), vec![1.0, 20.0, 30.0]);
    assert_eq!(a.mapv(|x| x + 1.0).to_vec(), vec![2.0, 5.0, 10.0]);
}

#[test]
fn reduction_basic() {
    let device = device();
    let a = rt::arange((24.0, &device)).into_shape([2, 3, 4]);

    assert_eq!(a.sum(), 276.0);
    assert_eq!(a.sum_axes(-1).shape(), &[2, 3]);
    assert_eq!(a.sum_axes((0, 2)).shape(), &[3]);
    assert_eq!(a.sum_with_args((-1, true)).shape(), &[2, 3, 1]);
    assert_eq!(a.mean(), 11.5);
    assert_eq!(a.max(), 23.0);
    assert_eq!(a.min(), 0.0);
    assert_eq!(a.argmax(), 23);
    assert_eq!(a.count_nonzero(), 23);
    let v = rt::arange((4.0, &device));
    assert_eq!(v.cumulative_sum(None).to_vec(), vec![0.0, 1.0, 3.0, 6.0]);
    assert!((&a).greater(-1.0).all());
}

#[test]
fn sorting_and_set() {
    let device = device();
    let a = rt::tensor_from_nested!([3, 1, 3, 2, 1], &device);

    assert_eq!(a.sort(None).to_vec(), vec![1, 1, 2, 3, 3]);
    assert_eq!(a.argsort(None).to_vec(), vec![1, 4, 3, 0, 2]);
    assert_eq!(a.unique_values().to_vec(), vec![1, 2, 3]);
    let test = rt::tensor_from_nested!([1, 3], &device);
    assert_eq!(a.isin(&test, false).to_vec(), vec![true, true, true, false, true]);

    let sorted = rt::tensor_from_nested!([1, 3, 5, 7], &device);
    let v = rt::tensor_from_nested!([2, 6], &device);
    assert_eq!(sorted.searchsorted(&v, None).to_vec(), vec![1, 3]);

    let d = rt::tensor_from_nested!([1, 4, 9, 16], &device);
    assert_eq!(d.diff(-1, 1, None, None).to_vec(), vec![3, 5, 7]);
}

#[test]
fn compare_allclose() {
    let device = device();
    let a = rt::tensor_from_nested!([1.0, 2.0, 3.0], &device);
    let b = rt::tensor_from_nested!([1.0, 2.0, 3.0 + 1e-12], &device);
    assert!(a.allclose(&b, None));
}
