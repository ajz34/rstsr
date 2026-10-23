//! Elementwise-kernel benchmark vs CuPy (the perf gate: not >2x slower).
//!
//! Scenarios mirror the kernel round's families: fill, strided assign
//! (to_contig of a transposed view), cast assign, binary add (contig +
//! transposed), unary sin, arange. Timing = per-op wall time after warmup
//! (sync-before/after; each op is enqueued once per iteration).
//!
//! Run: cargo bench --bench bench_elementwise (writes bench.out next to it)

use std::time::Instant;

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

fn bench<F: FnMut()>(name: &str, n_iter: usize, mut f: F) -> f64 {
    // warmup: first calls compile kernels (NVRTC) + driver-JIT modules
    for _ in 0..30 {
        f();
    }
    funcs::synchronize(&DeviceCuda::new(0)).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        f();
    }
    funcs::synchronize(&DeviceCuda::new(0)).unwrap();
    let dt = t0.elapsed().as_secs_f64() / n_iter as f64;
    println!("{name}: {:.3} ms/op", dt * 1e3);
    dt
}

fn main() {
    if !funcs::is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let d = DeviceCuda::new(0);
    let n_iter = 200;

    println!("=== fill f64 [16 MiB] ===");
    let n = 2_000_000usize;
    let dt_r = bench("rstsr fill", n_iter, || {
        let c: Tensor<f64, _, _> = rt::full(([n], 2.5, &d));
        std::hint::black_box(&c);
    });

    println!("=== add f64 contig [16 MiB] ===");
    let a = rt::arange((0., n as f64, 1., &d));
    let b = rt::arange((0., n as f64, 1., &d));
    let dt_r2 = bench("rstsr add", n_iter, || {
        let s = rt::add(&a, &b);
        std::hint::black_box(&s);
    });

    println!("=== add f64 transposed-B [4x4 MiB] ===");
    let a2 = rt::arange((0., 1_000_000., 1., &d)).into_shape([1000, 1000]);
    let b2 = rt::arange((0., 1_000_000., 1., &d)).into_shape([1000, 1000]);
    let b2t = b2.swapaxes(0, 1);
    bench("rstsr add-strided", n_iter, || {
        let s = rt::add(&a2, &b2t);
        std::hint::black_box(&s);
    });

    println!("=== sin f64 [16 MiB] ===");
    bench("rstsr sin", n_iter, || {
        let s = rt::sin(&a);
        std::hint::black_box(&s);
    });

    println!("=== to_contig transposed [4 MiB] ===");
    let a2t = a2.swapaxes(0, 1);
    bench("rstsr to_contig(T)", n_iter, || {
        let s = a2t.to_contig(RowMajor);
        std::hint::black_box(&s);
    });

    println!("=== arange f64 [2e5] ===");
    bench("rstsr arange", n_iter, || {
        let s = rt::arange((0., 2e5, 1., &d));
        std::hint::black_box(&s);
    });

    // reference numbers are produced by bench_cupy.py in the task dir;
    // this binary prints its own numbers for the table.
    let _ = (dt_r, dt_r2);
}
