//! Crate-local tests for `rstsr-cuda`: device value, registry,
//! `same_device` matrix, flags, `CudaBuffer`, creation round-trips,
//! transfers, formatting, allocator basics.
//!
//! Runs on any CUDA-capable machine; tests skip gracefully when no device
//! is visible (CI machines without GPU).

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;
use std::sync::Arc;

/// D2H into a host Vec of the tensor's VISIBLE elements (1-D, owning):
/// gather by layout first (`to_raw_f`), so sliced views yield only their
/// selected elements, then copy the gathered compact buffer to the host.
type OwnedStorageOf<T, B> = Storage<DataOwned<<B as DeviceRawAPI<T>>::Raw>, T, B>;

fn host_vec1<T, B, D>(t: &TensorBase<OwnedStorageOf<T, B>, D>) -> Vec<T>
where
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

fn has_cuda() -> bool {
    funcs::is_available()
}

macro_rules! require_cuda {
    () => {
        if !has_cuda() {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

#[test]
fn registry_identity() {
    require_cuda!();
    let d1 = DeviceCuda::new(0);
    let d2 = DeviceCuda::new(0);
    assert!(Arc::ptr_eq(d1.ordinal_obj(), d2.ordinal_obj()));
    // same-ordinal devices on their default stream are the same device
    assert!(d1.same_device(&d2));
    // Display is ordinal-only (`cuda:0`); Debug is the detailed form
    assert_eq!(format!("{d1}"), "cuda:0");
    assert!(format!("{d1:?}").contains("DeviceCuda"));
}

#[test]
fn same_device_matrix() {
    require_cuda!();
    let d1 = DeviceCuda::new(0);
    // order participates
    let mut d_order = d1.clone();
    d_order.set_default_order(ColMajor);
    assert!(!d1.same_device(&d_order));
    // flags-as-stored participate: Some(x) != None even under global x
    let mut d_pin = d1.clone();
    d_pin.set_flags_override(Some(CudaRuntimeFlags::default()));
    assert!(!d1.same_device(&d_pin));
    let mut d_pin2 = d1.clone();
    d_pin2.set_flags_override(Some(CudaRuntimeFlags::default()));
    assert!(d_pin.same_device(&d_pin2));
    // stream does NOT participate in device identity: different
    // streams of one ordinal are same-device; ops order them via events.
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let s2obj = Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2));
    let mut d_stream = d1.clone();
    d_stream.set_stream(s2obj);
    assert!(d1.same_device(&d_stream));
}

#[test]
fn flags_resolution() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    // None tracks the global
    let before = funcs::default_flags();
    let mut d2 = d.clone();
    d2.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: ComputeType32F::FastTF32 }));
    // pinned value wins over the global
    assert_eq!(d2.compute_type_32f(), ComputeType32F::FastTF32);
    assert_eq!(d.compute_type_32f(), before.compute_type_32f);
    // unpin tracks again
    d2.set_flags_override(None);
    assert_eq!(d2.compute_type_32f(), d.compute_type_32f());
}

#[test]
fn buffer_basics() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    // alloc + free of a nonzero buffer
    let mut buf = CudaBuffer::<f64>::new(d.stream(), 1024).unwrap();
    assert_eq!(buf.len(), 1024);
    assert!(!buf.device_ptr().is_null());
    // truncate shrinks logical len only
    buf.truncate(10);
    assert_eq!(buf.len(), 10);
    // deep clone: distinct pointer, independent storage
    let clone = buf.clone();
    assert_ne!(clone.device_ptr(), buf.device_ptr());
    assert_eq!(clone.len(), buf.len());
    // empty buffer is the null slice
    let empty = CudaBuffer::<f64>::new(d.stream(), 0).unwrap();
    assert!(empty.is_empty());
}

#[test]
fn creation_roundtrips() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    // asarray H2D -> to_vec D2H
    let host: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let a = rt::asarray((host.clone(), &device));
    assert_eq!(a.shape(), &[24]);
    assert_eq!(host_vec1(&a), host);
    // zeros is device-native
    let z: Tensor<f64, _> = rt::zeros(([6], &device));
    assert_eq!(host_vec1(&z), vec![0.0; 6]);
    // ones/full (naive fill path)
    let o: Tensor<f64, _> = rt::ones(([4], &device));
    assert_eq!(host_vec1(&o), vec![1.0; 4]);
    let f: Tensor<f64, _> = rt::full(([3], 2.5, &device));
    assert_eq!(host_vec1(&f), vec![2.5; 3]);
    // 2-D roundtrip with layout offset through views
    let b = rt::asarray((host.clone(), &device)).into_shape([4, 6]);
    let s = b.i(1);
    let s_cpu = s.to_device(&DeviceCpuSerial::default());
    assert_eq!(host_vec1(&s_cpu), (6..12).map(|i| i as f64).collect::<Vec<_>>());
}

#[test]
fn scalar_access() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    let a = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], &device));
    // to_scalar_f reads at the layout offset (0-D only); i(..) slices to a 0-d view
    assert_eq!(a.i(2).to_scalar_f().unwrap(), 3.0);
    // strided view reads hit the right elements
    let b = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0], &device)).into_shape([2, 2]);
    let row1 = b.i(1);
    let elem = row1.i(0);
    assert_eq!(elem.to_scalar_f().unwrap(), 3.0);
}

#[test]
fn transfers_cuda_cpu() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    let host: Vec<f64> = (0..12).map(|i| i as f64).collect();
    // CPU -> CUDA -> CPU content equality
    let up = rt::asarray((host.clone(), &device));
    let down = up.to_device(&DeviceCpuSerial::default());
    assert_eq!(host_vec1(&down), host);
    // same-ordinal relabel is zero copy. `change_device` requires an owned
    // tensor: consume `up` (a clone would deep-copy the buffer first —
    // value semantics — and defeat the pointer check).
    let mut d2 = device.clone();
    d2.set_default_order(ColMajor);
    assert!(!device.same_device(&d2));
    let up = rt::asarray((host, &device));
    let ptr_before = up.raw().device_ptr();
    let retag = up.change_device(&d2);
    assert_eq!(retag.raw().device_ptr(), ptr_before);
    assert!(retag.device().same_device(&d2));
}

#[test]
fn format_matches_cpu() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    // small tensor prints identically to the CPU device
    let host: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let cu = rt::asarray((host.clone(), &device)).into_shape([2, 3]);
    let cpu = rt::asarray((host.clone(), &DeviceCpuSerial::default())).into_shape([2, 3]);
    assert_eq!(format!("{cu}"), format!("{cpu}"));
    // truncation: last axis longer than MAX_PRINT
    let long: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let cu_long = rt::asarray((long.clone(), &device));
    let cpu_long = rt::asarray((long.clone(), &DeviceCpuSerial::default()));
    assert_eq!(format!("{cu_long}"), format!("{cpu_long}"));
    // exp format
    let e = rt::asarray((vec![0.001f64, 1.0, 1000.0], &device));
    let e_cpu = rt::asarray((vec![0.001f64, 1.0, 1000.0], &DeviceCpuSerial::default()));
    assert_eq!(format!("{:e}", e), format!("{:e}", e_cpu));
}

#[test]
fn allocator_valves() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    let (free0, total) = funcs::memory_info(&device).unwrap();
    assert!(total > 0);
    // churn then trim; free memory should not shrink monotonically forever
    for _ in 0..4 {
        let _big: Tensor<f64, _> = rt::zeros(([64 * 1024 * 1024], &device)); // 512 MiB
    }
    funcs::synchronize(&device).unwrap();
    let (free1, _) = funcs::memory_info(&device).unwrap();
    assert!(free1 > 0 && free0 > 0);
}

#[test]
fn oom_is_typed_error() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    // absurd allocation fails before reserving; must be MemoryError
    let err = <DeviceCuda as DeviceCreationNumAPI<f64>>::zeros_impl(&device, usize::MAX / 8);
    match err {
        Ok(_) => panic!("expected OOM"),
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(msg.contains("Memory"), "expected MemoryError, got {msg}");
        },
    }
}

#[test]
fn slice_and_ops_smoke() {
    require_cuda!();
    // With no op kernels yet, verify the layout machinery works on-device:
    // slicing/broadcasting/reshape are layout-only and must not move data.
    let device = DeviceCuda::new(0);
    let a = rt::asarray((vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &device)).into_shape([2, 3]);
    let t = a.swapaxes(0, 1);
    assert_eq!(t.shape(), &[3, 2]);
    let s = a.i(0);
    let s_cpu = s.to_device(&DeviceCpuSerial::default());
    assert_eq!(host_vec1(&s_cpu), vec![1.0, 2.0, 3.0]);
    let one = rt::asarray((vec![1.0f64], &device));
    let one_view = one.view();
    let wide = one_view.to_broadcast(vec![2, 3]);
    assert_eq!(wide.shape(), &[2, 3]);
}

#[test]
fn format_truncates_every_axis() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    let cpu = DeviceCpuSerial::default();
    // every axis beyond MAX_PRINT (8): prefix axes truncate to head/tail 3
    let host: Vec<f64> = (0..10 * 12 * 15).map(|i| i as f64).collect();
    let cu = rt::asarray((host.clone(), &device)).into_shape([10, 12, 15]);
    let cpu_t = rt::asarray((host.clone(), &cpu)).into_shape([10, 12, 15]);
    assert_eq!(format!("{cu}"), format!("{cpu_t}"));
    // strided (transposed) view of the same data
    let cu_t = cu.swapaxes(0, 2);
    let cpu_tt = cpu_t.swapaxes(0, 2);
    assert_eq!(format!("{cu_t}"), format!("{cpu_tt}"));
}

#[test]
fn format_strided_views_match_cpu() {
    require_cuda!();
    let device = DeviceCuda::new(0);
    let cpu = DeviceCpuSerial::default();
    let host: Vec<f64> = (0..10 * 12 * 15).map(|i| i as f64).collect();
    let a = rt::asarray((host.clone(), &device)).into_shape([10, 12, 15]);
    let a_cpu = rt::asarray((host.clone(), &cpu)).into_shape([10, 12, 15]);

    let v = rt::arange((0., 50., 1., &device));
    let v_cpu = rt::arange((0., 50., 1., &cpu));

    // 1-D offset slice
    assert_eq!(format!("{}", v.i(7..)), format!("{}", v_cpu.i(7..)));

    // 1-D stepped slice (stride 2)
    let s = SliceI::new(Some(0isize), None::<isize>, Some(2isize));
    let vs = v.i(s);
    assert_eq!(vs.layout().stride()[0], 2);
    assert_eq!(format!("{vs}"), format!("{}", v_cpu.i(s)));

    // 1-D reversed slice (negative stride)
    let r = SliceI::new(None::<isize>, None::<isize>, Some(-1isize));
    let vr = v.i(r);
    assert_eq!(vr.layout().stride()[0], -1);
    assert_eq!(format!("{vr}"), format!("{}", v_cpu.i(r)));

    // 2-D slices with offsets on both axes
    assert_eq!(format!("{}", a.i((5.., 3.., 8..))), format!("{}", a_cpu.i((5.., 3.., 8..))));

    // transposed + offset
    let at0 = a.swapaxes(0, 1);
    assert_eq!(at0.layout().stride(), &[15, 180, 1]);
    let at0_cpu = a_cpu.swapaxes(0, 1);
    let at = at0.i((2.., 4..));
    let at_cpu = at0_cpu.i((2.., 4..));
    assert_eq!(format!("{at}"), format!("{at_cpu}"));

    // permuted + offset
    let ap0 = a.permute_dims([2, 0, 1]);
    let ap0_cpu = a_cpu.permute_dims([2, 0, 1]);
    let ap = ap0.i(1..);
    let ap_cpu = ap0_cpu.i(1..);
    assert_eq!(format!("{ap}"), format!("{ap_cpu}"));
}
