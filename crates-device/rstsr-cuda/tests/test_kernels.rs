//! Kernel-round tests: strided/offset assigns, cast assigns, sequence
//! creation, elementwise ops, stream mixing — all vs CPU reference
//! (copy-and-compare parity shape).

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;
use std::sync::Arc;



/// D2H of a tensor's visible elements by layout gather (house helper).
fn host_vec1<T, B, D, R>(t: &TensorAny<R, T, B, D>) -> Vec<T>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    T: Clone + Send + Sync + 'static,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToCpuAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    let raw = t.to_raw_f().unwrap();
    DeviceToCpuAPI::<T>::to_cpu_vec::<DataOwned<<B as DeviceRawAPI<T>>::Raw>>(&Storage::new(
        DataOwned::from(raw),
        t.device().clone(),
    ))
    .unwrap()
}

/// Layout-ordered host values of an f64 tensor (N-d safe: the D2H goes
/// through the trusted CPU device, so strided views gather correctly).
fn host_nd<'a, Tr, B, D>(t: &TensorAny<Tr, f64, B, D>) -> Vec<f64>
where
    Tr: DataAPI<Data = <B as DeviceRawAPI<f64>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceToCpuAPI<f64> + DeviceCreationAnyAPI<f64> + OpAssignArbitaryAPI<f64, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<f64>::to_cpu_vec(contig.storage()).unwrap()
}

macro_rules! require_cuda {
    () => {
        if !funcs::is_available() {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

/* #region assign */

#[test]
fn assign_strided_view_sources() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::arange((0., 12., 1., &d)).into_shape([3, 4]);

    // transpose (strided view) -> to_contig: the D11 regression shape
    let at = a.swapaxes(0, 1);
    let at_contig = at.to_contig(RowMajor);
    let want = host_nd(&at);
    assert_eq!(host_nd(&at_contig), want);

    // row slice of the transpose (partially strided, offset != 0)
    let row_view = at.i(1);
    let row = row_view.to_contig(RowMajor);
    let want_row = host_vec1(&row_view);
    assert_eq!(host_vec1(&row), want_row);

    // assign through a strided source without intermediate to_contig
    let mut c: Tensor<f64, _, _> = rt::zeros(([4, 3], &d));
    c.assign(&at);
    assert_eq!(host_nd(&c), want);
}

#[test]
fn assign_cast_pairs() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::arange((0., 16., 1., &d));

    // f64 -> f32
    let mut c: Tensor<f32, _, _> = rt::zeros(([16], &d));
    c.assign(&a);
    let want: Vec<f32> = (0..16).map(|v| v as f32).collect();
    assert_eq!(host_vec1(&c), want);

    // f64 -> i32
    let mut c2: Tensor<i32, _, _> = rt::zeros(([16], &d));
    c2.assign(&a);
    let want2: Vec<i32> = (0..16).collect();
    assert_eq!(host_vec1(&c2), want2);

    // bool -> f32
    // (pin the host device: DeviceCpu resolves to DeviceFaer under
    // faer_as_default, whose conversion impls are feature-gated)
    let b_cpu: Tensor<bool, DeviceCpuSerial, _> = rt::asarray((vec![true, false, true, true], &DeviceCpuSerial::default()));
    let b = b_cpu.to_device(&d);
    let mut c3: Tensor<f32, _, _> = rt::zeros(([4], &d));
    c3.assign(&b);
    assert_eq!(host_vec1(&c3), vec![1.0f32, 0.0, 1.0, 1.0]);
}

#[test]
fn fill_strided_and_full() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let c: Tensor<f64, _, _> = rt::full(([2, 3], 2.5f64, &d));
    assert_eq!(host_nd(&c), vec![2.5f64; 6]);
    // strided fill through a mutable slice view (partially strided dst)
    let mut d2: Tensor<f64, _, _> = rt::zeros(([4, 3], &d));
    d2.i_mut(1..3).fill(7.0f64);
    let host = host_nd(&d2);
    assert_eq!(&host[3..9], &vec![7.0f64; 6][..]);
    assert_eq!(&host[0..3], &vec![0.0f64; 3][..]);
}

/* #endregion */

/* #region creation */

#[test]
fn arange_linspace_values() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::arange((0., 10., 2., &d));
    assert_eq!(host_vec1(&a), vec![0.0f64, 2.0, 4.0, 6.0, 8.0]);
    let l: Tensor<f64, _, _> = rt::linspace((0., 1., 5, &d));
    assert_eq!(host_vec1(&l), vec![0.0f64, 0.25, 0.5, 0.75, 1.0]);
}

/* #endregion */

/* #region elementwise ops */

#[test]
fn binary_arith_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::arange((0., 32., 1., &d)).into_shape([4, 8]);
    let b = rt::arange((100., 132., 1., &d)).into_shape([4, 8]);
    let a_cpu = a.to_device(&dcpu);
    let b_cpu = b.to_device(&dcpu);

    let s = rt::add(&a, &b);
    assert_eq!(host_nd(&s), host_nd(&rt::add(&a_cpu, &b_cpu)));

    // strided inputs: row-slice views (partially strided, offset != 0)
    let a_sl = a.i(1..3);
    let b_sl = b.i(1..3);
    let st = rt::add(&a_sl, &b_sl);
    let want = rt::add(&a_cpu.i(1..3), &b_cpu.i(1..3));
    assert_eq!(host_nd(&st), host_nd(&want));

    // broadcast input: [4,1] column view + [1,8] row view
    let col = a.i((.., 0..1));
    let row_cpu = b_cpu.i((0..1, ..));
    let row = row_cpu.to_device(&d);
    let sb = rt::add(&col, &row);
    let want_b = rt::add(&a_cpu.i((.., 0..1)), &row_cpu);
    assert_eq!(host_nd(&sb), host_nd(&want_b));
}

#[test]
fn compare_ops_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::arange((0., 8., 1., &d));
    let b = rt::arange((4., 12., 1., &d));
    let a_cpu = a.to_device(&dcpu);
    let b_cpu = b.to_device(&dcpu);
    let m = rt::lt(&a, &b);
    assert_eq!(host_vec1(&m), host_vec1(&rt::lt(&a_cpu, &b_cpu)));
    let e = rt::eq(&a, &b);
    assert_eq!(host_vec1(&e), host_vec1(&rt::eq(&a_cpu, &b_cpu)));
}

#[test]
fn unary_float_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::linspace((0.1, 1.0, 16, &d));
    let a_cpu = a.to_device(&dcpu);
    let s = rt::sin(&a);
    let want: Vec<f64> = host_vec1(&rt::sin(&a_cpu));
    assert!(host_vec1(&s).iter().zip(want.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
    let e = rt::exp(&a);
    let want_e: Vec<f64> = host_vec1(&rt::exp(&a_cpu));
    assert!(host_vec1(&e).iter().zip(want_e.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
}

/* #endregion */

/* #region streams + same_device */

#[test]
fn same_device_relaxed_streams() {
    require_cuda!();
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let s2obj = Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2));
    let mut d2 = d1.clone();
    d2.set_stream(s2obj);
    // different streams, same ordinal: same-device (D8)
    assert!(d1.same_device(&d2));
    // a cross-stream op succeeds and orders correctly
    let a = rt::arange((0., 16., 1., &d1));
    let b = rt::arange((0., 16., 1., &d2));
    let s = rt::add(&a, &b);
    let want: Vec<f64> = (0..16).map(|v| (v as f64) * 2.0).collect();
    assert_eq!(host_vec1(&s), want);
}

#[test]
fn mixed_stream_assign() {
    require_cuda!();
    // produce on stream A, assign on stream B (foreign-stream input)
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let s2obj = Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2));
    let mut d2 = d1.clone();
    d2.set_stream(s2obj);

    let a = rt::arange((0., 8., 1., &d1));
    let mut c: Tensor<f64, _, _> = rt::zeros(([8], &d2));
    c.assign(&a);
    let want: Vec<f64> = (0..8).map(|v| v as f64).collect();
    assert_eq!(host_vec1(&c), want);
}

/* #endregion */



/* #region DeviceFaer interop (feature = "faer") */

#[cfg(feature = "faer")]
#[test]
fn faer_to_cuda_and_back() {
    require_cuda!();
    let dcpu = DeviceFaer::default();
    let d = DeviceCuda::new(0);
    let a_cpu = rt::arange((0., 12., 1., &dcpu)).into_shape([3, 4]);
    let a = a_cpu.to_device(&d);
    let s = rt::add(&a, &a);
    let s_back = s.to_device(&dcpu);
    let want = host_nd(&rt::add(&a_cpu, &a_cpu));
    assert_eq!(host_nd(&s_back), want);

    // into_device (owned transfer)
    let owned = a_cpu.clone().into_device(&d);
    assert_eq!(host_nd(&owned), host_nd(&a_cpu));
}

/* #endregion */
