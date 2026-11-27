//! Matmul benchmark cells: type triple x shape x driver, median-of-R wall timing per cell (sync
//! once per repeat; each product enqueued once per iteration).
//!
//! Output: one `triple,shape,driver,median_ms,tflops` CSV line per cell, then the same as a
//! markdown table. `RSTSR_BENCH_TIME` is the time budget per repeat in seconds (default 0.2) and
//! `RSTSR_BENCH_REPEAT` the number of repeats (default 5); the iteration count follows from one
//! timed call.
//!
//! Run: cargo bench --bench bench_matmul   (or: cargo run --release --bench bench_matmul)
//!
//! Triples: f32 -> f32, f64 -> f64, c64 -> c64, f16 -> f16, f16 -> f32.
//! Shapes: square 2048, square 512, tall-skinny 65536x64x64, batch 256 x (256,256,256).
//! Drivers: the default chain, then every driver that serves the triple alone; the naive kernel
//! on square 512 only (it is minutes on the larger shapes). A complex product counts as four
//! real ones.
//!
//! The two special forms follow (`syrk` = `a @ a.t()` for square `[n, n]` and for `[4096, 64]`
//! and `[4096, 1024]` operands,
//! `dot k` = the inner dot of two `[k]` vectors), on the chain under the default
//! `MatMulUseSyrk` (for `f32` also under `Allow`), and on the gemm alone (the classic driver off
//! switches the symmetric form off). A `syrk` row counts the flops of the full
//! gemm it replaces, so its TFLOP/s compares with the gemm rows. `GB/s` is the memory traffic of
//! the operands and the output read or written once, a ceiling-bound measure for the dot.

use std::time::Instant;

use num::Complex;
use rstsr::prelude::*;
use rstsr_cuda::matmul::{ComputeType32F, MatMulDrivers, MatMulUseSyrk, TensorCudaMatMulAPI};
use rstsr_cuda::prelude_dev::*;
use rstsr_cuda::registry::is_available;

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn device(drivers: MatMulDrivers) -> DeviceCuda {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = drivers;
    DeviceCuda::new(0).with_flags(flags)
}

/// The symmetric-form shapes: `(label, n, k)` of the `[n, k]` operand.
fn syrk_shapes() -> Vec<(&'static str, usize, usize)> {
    vec![("syrk 2048", 2048, 2048), ("syrk 512", 512, 512), ("syrk 4096x64", 4096, 64), ("syrk 4096x1024", 4096, 1024)]
}

/// The inner-dot lengths.
const DOT_LENS: [usize; 2] = [1 << 22, 1 << 24];

/// The shapes: `(label, m, k, n, batch)`.
fn shapes() -> Vec<(&'static str, usize, usize, usize, usize)> {
    vec![
        ("square 2048", 2048, 2048, 2048, 1),
        ("square 512", 512, 512, 512, 1),
        ("tall-skinny 65536x64x64", 65536, 64, 64, 1),
        ("batch 256 x (256,256,256)", 256, 256, 256, 256),
    ]
}

/// Time `f` (one product per call) on `d`: the iteration count from one timed call, then the
/// median of `repeat` timed runs.
fn bench_cell<T>(d: &DeviceCuda, f: impl Fn() -> T) -> f64 {
    let budget = std::env::var("RSTSR_BENCH_TIME").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.2);
    let repeat = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
    // warmup: kernel JIT, the cuBLASLt heuristic, the cuBLAS workspace
    for _ in 0..3 {
        let _ = f();
    }
    d.synchronize();
    let t0 = Instant::now();
    let _ = f();
    d.synchronize();
    let one = t0.elapsed().as_secs_f64().max(1e-7);
    let n_iter = ((budget / one) as usize).clamp(1, 5000);
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let _ = f();
        }
        d.synchronize();
        samples.push(t0.elapsed().as_secs_f64() / n_iter as f64);
    }
    samples.sort_by(|x, y| x.total_cmp(y));
    samples[samples.len() / 2]
}

struct Row {
    triple: &'static str,
    shape: &'static str,
    driver: String,
    ms: f64,
    tflops: f64,
    gbps: f64,
}

fn main() {
    if !is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    // per triple: (name, real products per call, the serving drivers, the cell builder); the
    // naive kernel is timed on square 512 for every triple
    macro_rules! cells {
        ($triple:literal, $products:expr, $bytes:expr, [$($driver:ident),*], $mk:expr) => {{
            let drivers: Vec<(&'static str, MatMulDrivers)> = vec![$((stringify!($driver), MatMulDrivers::$driver)),*];
            for (label, m, k, n, p) in shapes() {
                let flops = 2.0 * $products * m as f64 * k as f64 * n as f64 * p as f64;
                let mk = $mk;
                let mut settings: Vec<(&'static str, MatMulDrivers)> = vec![("chain", MatMulDrivers::all())];
                settings.extend(drivers.clone());
                if label == "square 512" {
                    settings.push(("NAIVE", MatMulDrivers::NAIVE));
                }
                for (driver, flags) in settings {
                    // a fresh device per cell: the first call of a shape pays the Lt heuristic
                    let d = device(flags);
                    let f = mk(&d, m, k, n, p);
                    let s = bench_cell(&d, || f());
                    rows.push(Row {
                        triple: $triple,
                        shape: label,
                        driver: driver.to_lowercase(),
                        ms: s * 1e3,
                        tflops: flops / s / 1e12,
                        gbps: ($bytes)(m, k, n, p) as f64 / s / 1e9,
                    });
                }
            }
        }};
    }

    // a helper: `[p,] m, k` shaped tensor of the data on `d`
    macro_rules! operand {
        ($t:ty, $d:expr, $seed:expr, $m:expr, $k:expr, $p:expr, $conv:expr) => {{
            let v: Vec<$t> = data($p * $m * $k, $seed).into_iter().map($conv).collect();
            let t: Tensor<$t, DeviceCuda, IxD> =
                rt::asarray((v, $d)).into_shape(if $p > 1 { vec![$p, $m, $k] } else { vec![$m, $k] });
            t
        }};
    }

    macro_rules! bytes {
        ($ab:expr, $c:expr) => {
            |m: usize, k: usize, n: usize, p: usize| (m * k + k * n) * p * $ab + m * n * p * $c
        };
    }

    cells!("f32 -> f32", 1.0, bytes!(4, 4), [CUBLAS_LT, CUBLAS_EX, CUBLAS, TILED], |d: &DeviceCuda, m, k, n, p| {
        let a = operand!(f32, d, 1, m, k, p, |x| x as f32);
        let b = operand!(f32, d, 2, k, n, p, |x| x as f32);
        move || rt::matmul(&a, &b)
    });
    cells!("f64 -> f64", 1.0, bytes!(8, 8), [CUBLAS_LT, CUBLAS_EX, CUBLAS, TILED], |d: &DeviceCuda, m, k, n, p| {
        let a = operand!(f64, d, 1, m, k, p, |x| x);
        let b = operand!(f64, d, 2, k, n, p, |x| x);
        move || rt::matmul(&a, &b)
    });
    cells!("c64 -> c64", 4.0, bytes!(16, 16), [CUBLAS_LT, CUBLAS_EX, CUBLAS], |d: &DeviceCuda, m, k, n, p| {
        let a = operand!(Complex<f64>, d, 1, m, k, p, |x| Complex::new(x, 0.5 * x));
        let b = operand!(Complex<f64>, d, 2, k, n, p, |x| Complex::new(x, -0.5 * x));
        move || rt::matmul(&a, &b)
    });
    cells!("f16 -> f16", 1.0, bytes!(2, 2), [CUBLAS_LT, CUBLAS_EX], |d: &DeviceCuda, m, k, n, p| {
        let a = operand!(half::f16, d, 1, m, k, p, half::f16::from_f64);
        let b = operand!(half::f16, d, 2, k, n, p, half::f16::from_f64);
        move || rt::matmul(&a, &b)
    });
    cells!("f16 -> f32", 1.0, bytes!(2, 4), [CUBLAS_LT, CUBLAS_EX], |d: &DeviceCuda, m, k, n, p| {
        let a = operand!(half::f16, d, 1, m, k, p, half::f16::from_f64);
        let b = operand!(half::f16, d, 2, k, n, p, half::f16::from_f64);
        move || a.matmul_with_mode(&b, ComputeType32F::Default).out::<f32>()
    });

    // the symmetric form: `a @ a.t()`, the chain (which takes the form where it pays) against
    // the gemm alone; the flops are those of the full gemm it replaces
    macro_rules! syrk_cells {
        ($triple:literal, $products:expr, $t:ty, $conv:expr, $esz:expr) => {
            for (label, n, k) in syrk_shapes() {
                // the chain under the default flag, the chain under `Allow` (which adds `f32`),
                // and the gemm
                let mut settings = vec![
                    ("chain", MatMulDrivers::all(), MatMulUseSyrk::default()),
                    ("gemm", MatMulDrivers::all().minus(MatMulDrivers::CUBLAS), MatMulUseSyrk::default()),
                ];
                if stringify!($t) == "f32" {
                    settings.push(("chain, Allow", MatMulDrivers::all(), MatMulUseSyrk::Allow));
                }
                for (driver, flags, use_syrk) in settings {
                    let mut runtime = CudaRuntimeFlags::default();
                    runtime.matmul_drivers = flags;
                    runtime.matmul_use_syrk = use_syrk;
                    let d = DeviceCuda::new(0).with_flags(runtime);
                    let v: Vec<$t> = data(n * k, 1).into_iter().map($conv).collect();
                    let a: Tensor<$t, DeviceCuda, IxD> = rt::asarray((v, &d)).into_shape([n, k]);
                    let s = bench_cell(&d, || rt::matmul(&a, a.t()));
                    rows.push(Row {
                        triple: $triple,
                        shape: label,
                        driver: driver.to_string(),
                        ms: s * 1e3,
                        tflops: 2.0 * $products * n as f64 * n as f64 * k as f64 / s / 1e12,
                        gbps: (2 * n * k + n * n) as f64 * $esz as f64 / s / 1e9,
                    });
                }
            }
        };
    }
    syrk_cells!("f32 -> f32", 1.0, f32, |x| x as f32, 4);
    syrk_cells!("f64 -> f64", 1.0, f64, |x| x, 8);
    syrk_cells!("c64 -> c64", 4.0, Complex<f64>, |x| Complex::new(x, 0.5 * x), 16);

    // the inner dot of two vectors: memory-bound, so `GB/s` is the figure to read
    for k in DOT_LENS {
        for (driver, flags) in [("chain", MatMulDrivers::all()), ("cublasLt", MatMulDrivers::CUBLAS_LT)] {
            let d = device(flags);
            let v: Vec<f64> = data(k, 3);
            let x: Tensor<f64, DeviceCuda, IxD> = rt::asarray((v, &d)).into_shape(vec![k]);
            let s = bench_cell(&d, || rt::matmul(&x, x.t()));
            let label = if k == DOT_LENS[0] { "dot 2^22" } else { "dot 2^24" };
            rows.push(Row {
                triple: "f64 -> f64",
                shape: label,
                driver: driver.to_string(),
                ms: s * 1e3,
                tflops: 0.0,
                gbps: 2.0 * k as f64 * 8.0 / s / 1e9,
            });
        }
    }

    println!("triple,shape,driver,median_ms,tflops,gbs");
    for r in &rows {
        println!("{},{},{},{:.4},{:.3},{:.1}", r.triple, r.shape, r.driver, r.ms, r.tflops, r.gbps);
    }
    println!("\n| triple | shape | driver | ms | TFLOP/s | GB/s |\n|---|---|---|---|---|---|");
    for r in &rows {
        println!("| {} | {} | {} | {:.4} | {:.2} | {:.1} |", r.triple, r.shape, r.driver, r.ms, r.tflops, r.gbps);
    }
}
