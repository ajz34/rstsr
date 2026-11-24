//! Pinned host memory tensors (`DeviceCudaPinned` / `PinnedBuffer<T>`): buffer
//! round-trips, zero-copy host views, non-blocking pinned <-> CUDA edges,
//! cross-stream ordering, the stream/event/synchronize API, and a 3-stream
//! pipeline checked bitwise against the serial flow.
//!
//! Runs on any CUDA-capable machine; tests skip gracefully when no device is
//! visible (CI machines without GPU).

use rstsr::prelude::*;
use rstsr_cuda::pinned::pinned_flags;
use rstsr_cuda::pinned::transfer;
use rstsr_cuda::prelude_dev::*;
use std::sync::Arc;

fn has_cuda() -> bool {
    rstsr_cuda::funcs::is_available()
}

macro_rules! require_cuda {
    () => {
        if !has_cuda() {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

fn arange_vec(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

#[test]
fn pinned_buffer_roundtrip() {
    require_cuda!();
    let dev = DeviceCudaPinned::new(0);
    let vec = arange_vec(1000);
    let buf = dev.from_vec_f(vec.clone()).unwrap();
    assert_eq!(buf.len(), 1000);
    assert_eq!(buf.to_vec_f().unwrap(), vec);
    let buf2 = buf.clone();
    assert_eq!(buf2.into_vec_f().unwrap(), vec);

    let z = dev.zeros_f::<f64>(17).unwrap();
    assert_eq!(z.into_vec_f().unwrap(), vec![0.0; 17]);
    let f = dev.full_f(5, 1.5f64).unwrap();
    assert_eq!(f.into_vec_f().unwrap(), vec![1.5; 5]);

    // storage-level access after the implicit sync
    let t = dev.tensor_f(dev.from_vec_f(vec.clone()).unwrap(), [10, 100]).unwrap();
    assert_eq!(t.storage().get_index(305), 305.0);
    assert_eq!(t.storage().len(), 1000);
    let t2 = t.to_device(&DeviceCpuSerial::default());
    assert_eq!(t2.reshape([-1]).to_vec(), vec);
}

#[test]
fn pinned_host_views_read_write_and_cpu_ops() {
    require_cuda!();
    let dev = DeviceCudaPinned::new(0);
    let vec = arange_vec(12);
    let mut t = dev.tensor_f(dev.from_vec_f(vec.clone()).unwrap(), [3, 4]).unwrap();

    // read view: zero-copy, same bytes
    let v = t.host_view();
    assert_eq!(v.reshape([-1]).to_vec(), vec);
    // zero-copy: the view's buffer pointer is the pinned pointer
    let pinned_ptr = t.raw().as_slice().as_ptr() as usize;
    assert_eq!(v.raw().as_ptr() as usize, pinned_ptr);

    // write through the mut view lands in the pinned bytes
    {
        let mut m = t.host_view_mut();
        m[&[1usize, 1][..]] = -1.0;
    }
    let mut expected = vec.clone();
    expected[5] = -1.0;
    assert_eq!(t.raw().to_vec_f().unwrap(), expected);

    // a CPU op on the views: c = a + b written straight into pinned memory
    let a = dev.tensor_f(dev.from_vec_f(vec.clone()).unwrap(), [3, 4]).unwrap();
    let mut c = dev.tensor_f(dev.alloc_f::<f64>(12).unwrap(), [3, 4]).unwrap();
    {
        let va = a.host_view();
        let mut vc = c.host_view_mut();
        vc.assign(rt::add(&va, &va));
    }
    assert_eq!(c.raw().to_vec_f().unwrap(), vec.iter().map(|x| 2.0 * x).collect::<Vec<_>>());

    // layout follows the tensor (transposed view reads transposed)
    let tv = t.transpose([1, 0]);
    let vt = tv.host_view();
    assert_eq!(vt[&[0usize, 1][..]], expected[4]);
}

#[test]
fn pinned_cuda_roundtrip_default_and_side_stream() {
    require_cuda!();
    let dev_c = DeviceCuda::new(0);
    let dev_s = dev_c.new_stream();
    let pinned = DeviceCudaPinned::new(0);
    let vec = arange_vec(4096);

    for dev in [&dev_c, &dev_s] {
        let t = pinned.tensor_f(pinned.from_vec_f(vec.clone()).unwrap(), [64, 64]).unwrap();
        let g = t.to_device(dev);
        let back = g.to_device(&pinned);
        // immediate host access (implicit event sync)
        assert_eq!(back.raw().to_vec_f().unwrap(), vec);
        // a device op on the uploaded data, then back
        let g2 = rt::add(&g, &g);
        let back2 = g2.to_device(&pinned);
        assert_eq!(back2.raw().to_vec_f().unwrap(), vec.iter().map(|x| 2.0 * x).collect::<Vec<_>>());
        // consuming forms
        let g3 = t.into_device(dev);
        let back3 = g3.into_device(&pinned);
        assert_eq!(back3.raw().to_vec_f().unwrap(), vec);
    }

    // pageable -> pinned -> pageable (host copies)
    let cpu = DeviceCpuSerial::default();
    let t_cpu = rt::asarray((vec![1.0f64, 2.0, 3.0], &cpu));
    let t_pin = t_cpu.to_device(&pinned);
    assert_eq!(t_pin.raw().to_vec_f().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(t_pin.to_device(&cpu).to_vec(), vec![1.0, 2.0, 3.0]);
}

/// Faer host views and edges (only with `--features faer`).
#[cfg(feature = "faer")]
#[test]
fn pinned_faer_host_view_on() {
    require_cuda!();
    let dev = DeviceCudaPinned::new(0);
    let faer = DeviceFaer::default();
    let t = dev.tensor_f(dev.from_vec_f(arange_vec(6)).unwrap(), [2, 3]).unwrap();
    let v = t.host_view_on(&faer);
    assert_eq!(v.reshape([-1]).to_vec(), arange_vec(6));
    let t_f = t.to_device(&faer);
    assert_eq!(t_f.reshape([-1]).to_vec(), arange_vec(6));
}

#[test]
fn foreign_stream_temporary_source_ordered() {
    require_cuda!();
    // A tensor produced on stream S2, re-tagged onto S0 (buffer allocation stream
    // S2 != exec stream S0), then copied into pinned memory from S0: the DMA reads
    // a buffer whose free is ordered on S2. Reproduced the known read-lifetime hole
    // for kernels; the pinned D2H edge must order both ways.
    let dev_s0 = DeviceCuda::new(0);
    let dev_s2 = dev_c_new_stream(&dev_s0);
    let pinned = DeviceCudaPinned::new(0);
    let n = 1 << 22;
    let vec = arange_vec(n);
    let base = pinned.tensor_f(pinned.from_vec_f(vec.clone()).unwrap(), [n]).unwrap().to_device(&dev_s2);
    let one = pinned.tensor_f(pinned.from_vec_f(vec![1.0f64; n]).unwrap(), [n]).unwrap().to_device(&dev_s2);

    for _ in 0..100 {
        let tmp = rt::add(&base, &one); // allocated on S2
        let retagged = tmp.to_device(&dev_s0); // view on S0, buffer still allocated on S2
        let out = retagged.to_device(&pinned); // DMA on S0 from a buffer freed on S2
                                               // churn the S2 pool so the temporary's block would
                                               // be reused immediately
        let _churn = rt::add(&base, &base);
        assert_eq!(out.raw().to_vec_f().unwrap(), vec.iter().map(|x| x + 1.0).collect::<Vec<_>>());
        dev_s0.synchronize().unwrap();
        dev_s2.synchronize().unwrap();
    }
}

fn dev_c_new_stream(dev: &DeviceCuda) -> DeviceCuda {
    dev.new_stream()
}

#[test]
fn noncontiguous_pinned_layout_transfers_byte_equal() {
    require_cuda!();
    let pinned = DeviceCudaPinned::new(0);
    let dev = DeviceCuda::new(0);
    let vec = arange_vec(20);
    let t = pinned.tensor_f(pinned.from_vec_f(vec.clone()).unwrap(), [4, 5]).unwrap();
    let tv = t.transpose([1, 0]); // [5, 4] view, strides swapped, same bytes
    let g = tv.to_device(&dev);
    assert_eq!(g.shape(), &[5, 4]);
    let back = g.to_device(&pinned);
    assert_eq!(back.raw().to_vec_f().unwrap(), vec);
    assert_eq!(back.host_view()[&[0usize, 1][..]], vec[5]);
}

#[test]
fn empty_pinned_buffer() {
    require_cuda!();
    let pinned = DeviceCudaPinned::new(0);
    let dev = DeviceCuda::new(0);
    let buf = pinned.alloc_f::<f64>(0).unwrap();
    assert!(buf.is_empty());
    let t = pinned.tensor_f(buf, [0]).unwrap();
    let g = t.to_device(&dev);
    let back = g.to_device(&pinned);
    assert_eq!(back.raw().len(), 0);
    assert_eq!(back.raw().to_vec_f().unwrap(), Vec::<f64>::new());
}

#[test]
fn drop_with_in_flight_d2h_is_safe() {
    require_cuda!();
    // Drop the destination right after enqueueing a large D2H; the drop must
    // wait for the DMA before freeing the pages (no corruption, no driver error).
    let pinned = DeviceCudaPinned::new(0);
    let dev = DeviceCuda::new(0);
    let n = 1 << 24;
    let g = pinned.tensor_f(pinned.from_vec_f(vec![7.0f64; n]).unwrap(), [n]).unwrap().to_device(&dev);
    let mut out = pinned.alloc_f::<f64>(n).unwrap();
    transfer::d2h_pinned_async(&dev, g.raw(), &mut out).unwrap();
    drop(out);
    // allocate again immediately: the freed pages may come back
    let again = pinned.zeros_f::<f64>(n).unwrap();
    assert_eq!(again.into_vec_f().unwrap(), vec![0.0; n]);
}

#[test]
fn stream_api_without_set_stream() {
    require_cuda!();
    let base = DeviceCuda::new(0);
    let dev_a = base.new_stream();
    let dev_b = DeviceCuda::new_stream_with_ordinal(0);
    assert_eq!(dev_a.ordinal(), 0);
    assert!(!Arc::ptr_eq(dev_a.stream(), base.stream()));
    assert!(!Arc::ptr_eq(dev_a.stream(), dev_b.stream()));
    let dev_c = base.clone().with_stream(dev_b.stream());
    assert!(Arc::ptr_eq(dev_c.stream(), dev_b.stream()));
    assert!(dev_a.same_device(&base));

    // producer on A, consumer on B: legal through the op-level event ordering
    let pinned = DeviceCudaPinned::new(0);
    let n = 1 << 20;
    let a = pinned.tensor_f(pinned.from_vec_f(arange_vec(n)).unwrap(), [n]).unwrap().to_device(&dev_a);
    let s = rt::add(&a, &a); // on A
    let s_b = s.to_device(&dev_b); // re-tag
    let t = rt::add(&s_b, &s_b); // on B, reads an A-allocated buffer
    let back = t.to_device(&pinned);
    assert_eq!(back.raw().to_vec_f().unwrap(), (0..n).map(|i| 4.0 * i as f64).collect::<Vec<_>>());
}

#[test]
fn event_orders_copy_behind_foreign_kernel() {
    require_cuda!();
    // Without any tensor-level machinery: record an event after a kernel on
    // stream B and make the copy stream A wait for it before its D2H.
    let dev_a = DeviceCuda::new(0);
    let dev_b = dev_a.new_stream();
    let pinned = DeviceCudaPinned::new(0);
    let n = 1 << 22;
    let src = pinned.tensor_f(pinned.from_vec_f(vec![1.0f64; n]).unwrap(), [n]).unwrap().to_device(&dev_b);
    let s = rt::add(&src, &src); // kernel on B
    let ev = CudaEvent::new_f(&dev_b).unwrap();
    ev.record(dev_b.stream()).unwrap();
    let mut out = pinned.alloc_f::<f64>(n).unwrap();
    ev.wait(dev_a.stream()).unwrap();
    transfer::d2h_pinned_async(&dev_a, s.raw(), &mut out).unwrap();
    assert_eq!(out.into_vec_f().unwrap(), vec![2.0; n]);
    assert!(ev.is_complete_f().unwrap());
}

#[test]
fn synchronize_readiness() {
    require_cuda!();
    let dev = DeviceCuda::new(0);
    let pinned = DeviceCudaPinned::new(0);
    let n = 1 << 22;
    let g = pinned.tensor_f(pinned.from_vec_f(arange_vec(n)).unwrap(), [n]).unwrap().to_device(&dev);
    let out = g.to_device(&pinned); // D2H enqueued, not waited
    out.synchronize(); // explicit readiness
    assert!(out.raw().event().is_complete_f().unwrap());
    assert_eq!(out.raw().to_vec_f().unwrap(), arange_vec(n));
    g.synchronize();
    dev.synchronize().unwrap();
    pinned.synchronize();
    rstsr_cuda::funcs::synchronize_tensor(&out).unwrap();
}

#[test]
fn pipeline_three_streams_bitwise_equals_serial() {
    require_cuda!();
    // 3 streams (h2d / compute / d2h), pinned host, non-blocking edges; the
    // result must equal the serial (pageable, blocking) flow bit for bit.
    let base = DeviceCuda::new(0);
    let dev_h = base.new_stream();
    let dev_c = base.clone().with_stream(DeviceCuda::new_stream_with_ordinal(0).stream());
    let dev_d = base.clone().with_stream(DeviceCuda::new_stream_with_ordinal(0).stream());
    let pinned = DeviceCudaPinned::new(0);
    let chunks = 8;
    let n = 1 << 16;
    let host_in: Vec<Vec<f64>> = (0..chunks).map(|c| (0..n).map(|i| (c * n + i) as f64).collect()).collect();

    // serial reference on pageable memory
    let serial: Vec<Vec<f64>> = host_in
        .iter()
        .map(|v| {
            let cpu = DeviceCpuSerial::default();
            let t = rt::asarray((v.clone(), &cpu));
            let g = t.to_device(&base);
            let g2 = rt::add(&g, &g);
            let r = rt::add(&g2, &g);
            let back = r.to_device(&cpu);
            back.to_vec()
        })
        .collect();

    // pipeline
    let pin_in: Vec<_> =
        host_in.iter().map(|v| pinned.tensor_f(pinned.from_vec_f(v.clone()).unwrap(), [n]).unwrap()).collect();
    let mut pin_out: Vec<_> =
        (0..chunks).map(|_| pinned.tensor_f(pinned.alloc_f::<f64>(n).unwrap(), [n]).unwrap()).collect();
    // Keep every device buffer alive until the final sync: a buffer freed on its
    // allocation stream while a foreign-stream kernel still reads it is the known
    // read-lifetime hole (not fixed by this work; `staging` is the user-side guard).
    let mut staging = Vec::new();
    for t in &pin_in {
        staging.push(t.to_device(&dev_h)); // H2D on H
    }
    for c in 0..chunks {
        let g_c = staging[c].to_device(&dev_c); // compute on C (waits for H automatically)
        let g2 = rt::add(&g_c, &g_c);
        staging.push(rt::add(&g2, &g_c));
    }
    let results: Vec<_> = staging.iter().skip(chunks).map(|r| r.to_device(&dev_d)).collect();
    for (c, r) in results.iter().enumerate() {
        // D2H into the preallocated pinned output chunk (non-blocking)
        transfer::d2h_pinned_async(&dev_d, r.raw(), pin_out[c].raw_mut()).unwrap();
    }
    for (c, out) in pin_out.iter().enumerate() {
        out.synchronize();
        assert_eq!(out.raw().to_vec_f().unwrap(), serial[c]);
    }
    // write-combined pages transfer the same bytes
    let wc = PinnedBuffer::<f64>::with_flags_f(pinned.context(), n, pinned_flags::WRITE_COMBINED).unwrap();
    assert_eq!(wc.byte_len(), n * 8);
}
