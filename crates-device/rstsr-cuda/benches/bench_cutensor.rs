//! cuTENSOR wrapper benchmark cells: the one-shot path (plan per call) vs a held [`Plan`]
//! (execute only) vs the cuBLAS-chain matmul for the same contraction, plus the streaming
//! families (reduction, elementwise binary) and the trinary contraction.
//!
//! Output: one `family,dtype,shape,path,median_ms,metric` CSV line per cell (TFLOP/s for
//! contractions, GB/s for the streaming families, a percentage for the multi-stream cells:
//! 100 x sequential wall time over concurrent, 100 = serialized, lower = more overlap), then the
//! same as a markdown table. `RSTSR_BENCH_TIME` is the time budget per repeat in seconds (default
//! 0.2) and `RSTSR_BENCH_REPEAT` the number of repeats (default 5); the iteration count follows
//! from one timed call.
//!
//! Run: cargo bench --bench bench_cutensor   (or: cargo run --release --bench bench_cutensor)
//!
//! The one-shot pays the planning cost (20-50 us per call, plus a one-time ~28 ms lazy cuBLAS
//! initialization on the first cuBLAS-mapped problem) on every call; the held plan pays planning
//! once outside the timed region. The cuBLAS rows are the same `ij,jk->ik` product through the
//! crate's matmul chain, for an honest placement of the cuTENSOR contraction. The `overlap`
//! family runs a contraction on the default stream against a 256 MiB D2H / H2D on a second
//! stream and thread — the multi-stream premise of the unlocked execute path, measured.

// The cuTENSOR wrapper exists only under the `cutensor_bindings` cfg (CUDA >= 12.0 pins); the
// bench binary is empty otherwise.
#![cfg_attr(not(cutensor_bindings), allow(dead_code, unused_imports))]

use std::time::Instant;

use num::Complex;
use rstsr::cuda::cutensor_wrap;
use rstsr::prelude::*;
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

/// Time `f` on `d`: the iteration count from one timed call, then the median of `repeat` runs.
fn bench_cell<T>(d: &DeviceCuda, mut f: impl FnMut() -> T) -> f64 {
    let budget = std::env::var("RSTSR_BENCH_TIME").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.2);
    let repeat = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
    // warmup: kernel JIT, the lazy cuBLAS initialization inside the first cuBLAS-mapped estimate
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

/// Wall time of one call of `f` (enqueue + device sync), median of `RSTSR_BENCH_REPEAT` runs.
fn once_ms<T>(d: &DeviceCuda, mut f: impl FnMut() -> T) -> f64 {
    let repeat = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
    for _ in 0..3 {
        let _ = f();
    }
    d.synchronize();
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        d.synchronize();
        let t0 = Instant::now();
        let _ = f();
        d.synchronize();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|x, y| x.total_cmp(y));
    samples[samples.len() / 2]
}

/// `(sequential, concurrent)` wall times in ms of `main_fn` on this thread against `side_fn` on
/// a spawned thread: sequential runs the side first (joined) then the main; concurrent starts
/// the side and runs the main without waiting. Median of `RSTSR_BENCH_REPEAT` runs.
fn overlap_ms(d: &DeviceCuda, mut main_fn: impl FnMut(), side_fn: impl Fn() + Clone + Send + 'static) -> (f64, f64) {
    let repeat = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
    let mut seq = Vec::with_capacity(repeat);
    let mut par = Vec::with_capacity(repeat);
    for _ in 0..3 {
        main_fn();
        side_fn();
    }
    d.synchronize();
    for _ in 0..repeat {
        d.synchronize();
        let t0 = Instant::now();
        let side = std::thread::spawn(side_fn.clone());
        side.join().expect("side thread");
        main_fn();
        d.synchronize();
        seq.push(t0.elapsed().as_secs_f64() * 1e3);
        d.synchronize();
        let t0 = Instant::now();
        let side = std::thread::spawn(side_fn.clone());
        main_fn();
        side.join().expect("side thread");
        d.synchronize();
        par.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.total_cmp(y));
        v[v.len() / 2]
    };
    (med(&mut seq), med(&mut par))
}

/// One bench cell: the closure and its display name.
type Cell<'a> = (&'a str, Box<dyn FnMut() + 'a>);

/// One ceiling cell: dtype, shape label, edge length, closure.
type CeilingCell<'a> = (&'static str, &'static str, usize, Box<dyn FnMut() + 'a>);

struct Row {
    family: &'static str,
    dtype: &'static str,
    shape: String,
    path: String,
    ms: f64,
    metric: f64,
}

#[cfg(cutensor_bindings)]
fn main() {
    if !is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let dev = DeviceCuda::new(0);
    let Some(handle) = dev.ordinal_obj().cutensor().cloned() else {
        eprintln!("libcutensor is not loadable");
        return;
    };
    eprintln!("cuTENSOR runtime version: {:?}", handle.version());
    let mut rows: Vec<Row> = Vec::new();

    // ---- contraction: one-shot vs held plan vs cuBLAS-chain matmul -------------------------
    for (dtype, shape) in [
        ("f32", (1024usize, 1024, 1024)),
        ("f32", (2048, 2048, 2048)),
        ("f64", (1024, 1024, 1024)),
        ("f64", (2048, 2048, 2048)),
        ("c64", (1024, 1024, 1024)),
    ] {
        let (m, k, n) = shape;
        let flop_scale = if dtype == "c64" { 4.0 } else { 1.0 };
        match dtype {
            "f64" => {
                let av = data(m * k, 70);
                let bv = data(k * n, 71);
                let a = rt::asarray((av, &dev)).into_shape([m, k]);
                let b = rt::asarray((bv, &dev)).into_shape([k, n]);
                let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
                // held plan: create once outside the timed region
                let mut d = rt::zeros(([m, n], &dev));
                let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
                let desc_a = cutensor_wrap::TensorDescriptor::new::<f64>(&handle, &a.view()).unwrap();
                let desc_b = cutensor_wrap::TensorDescriptor::new::<f64>(&handle, &b.view()).unwrap();
                let desc_d = cutensor_wrap::TensorDescriptor::new::<f64>(&handle, &d.view()).unwrap();
                let op = cutensor_wrap::OperationDescriptor::new_contraction::<f64>(
                    &handle,
                    &desc_a,
                    &[0, 1],
                    ident(),
                    &desc_b,
                    &[1, 2],
                    ident(),
                    &desc_d,
                    &[0, 2],
                    &desc_d,
                    &[0, 2],
                    compute,
                )
                .unwrap();
                let pref = cutensor_wrap::PlanPreference::new(
                    &handle,
                    cudarc::cutensor::sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
                    cudarc::cutensor::sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
                )
                .unwrap();
                let plan = cutensor_wrap::Plan::new(&handle, op, &pref, 0).unwrap();
                let cells: Vec<Cell> = vec![
                    (
                        "one-shot",
                        Box::new(|| {
                            let mut d = rt::zeros(([m, n], &dev));
                            cutensor_wrap::contract(
                                &dev,
                                a.view(),
                                &[0, 1],
                                ident(),
                                b.view(),
                                &[1, 2],
                                ident(),
                                None::<TensorView<'_, f64, DeviceCuda, IxD>>,
                                d.view_mut(),
                                &[0, 2],
                                1.0f64,
                                0.0f64,
                                compute,
                                None,
                            )
                            .unwrap();
                        }),
                    ),
                    (
                        "held plan",
                        Box::new(|| {
                            plan.contract(
                                &dev,
                                a.view(),
                                b.view(),
                                None::<TensorView<'_, f64, DeviceCuda, IxD>>,
                                d.view_mut(),
                                1.0f64,
                                0.0f64,
                                None,
                            )
                            .unwrap();
                        }),
                    ),
                    (
                        "cublas chain",
                        Box::new(|| {
                            let _ = rt::matmul(&a, &b);
                        }),
                    ),
                ];
                for (path, f) in cells {
                    let ms = bench_cell(&dev, f) * 1e3;
                    let tflops = 2.0 * flop_scale * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
                    rows.push(Row {
                        family: "contraction",
                        dtype,
                        shape: format!("{m}x{k}x{n}"),
                        path: path.to_string(),
                        ms,
                        metric: tflops,
                    });
                }
            },
            "f32" => {
                let av: Vec<f32> = data(m * k, 70).into_iter().map(|x| x as f32).collect();
                let bv: Vec<f32> = data(k * n, 71).into_iter().map(|x| x as f32).collect();
                let a = rt::asarray((av, &dev)).into_shape([m, k]);
                let b = rt::asarray((bv, &dev)).into_shape([k, n]);
                let compute = cutensor_wrap::compute_desc::desc_32f().unwrap();
                let mut d = rt::zeros(([m, n], &dev));
                let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
                let desc_a = cutensor_wrap::TensorDescriptor::new::<f32>(&handle, &a.view()).unwrap();
                let desc_b = cutensor_wrap::TensorDescriptor::new::<f32>(&handle, &b.view()).unwrap();
                let desc_d = cutensor_wrap::TensorDescriptor::new::<f32>(&handle, &d.view()).unwrap();
                let op = cutensor_wrap::OperationDescriptor::new_contraction::<f32>(
                    &handle,
                    &desc_a,
                    &[0, 1],
                    ident(),
                    &desc_b,
                    &[1, 2],
                    ident(),
                    &desc_d,
                    &[0, 2],
                    &desc_d,
                    &[0, 2],
                    compute,
                )
                .unwrap();
                let pref = cutensor_wrap::PlanPreference::new(
                    &handle,
                    cudarc::cutensor::sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
                    cudarc::cutensor::sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
                )
                .unwrap();
                let plan = cutensor_wrap::Plan::new(&handle, op, &pref, 0).unwrap();
                let cells: Vec<Cell> = vec![
                    (
                        "one-shot",
                        Box::new(|| {
                            let mut d = rt::zeros(([m, n], &dev));
                            cutensor_wrap::contract(
                                &dev,
                                a.view(),
                                &[0, 1],
                                ident(),
                                b.view(),
                                &[1, 2],
                                ident(),
                                None::<TensorView<'_, f32, DeviceCuda, IxD>>,
                                d.view_mut(),
                                &[0, 2],
                                1.0f32,
                                0.0f32,
                                compute,
                                None,
                            )
                            .unwrap();
                        }),
                    ),
                    (
                        "held plan",
                        Box::new(|| {
                            plan.contract(
                                &dev,
                                a.view(),
                                b.view(),
                                None::<TensorView<'_, f32, DeviceCuda, IxD>>,
                                d.view_mut(),
                                1.0f32,
                                0.0f32,
                                None,
                            )
                            .unwrap();
                        }),
                    ),
                    (
                        "cublas chain",
                        Box::new(|| {
                            let _ = rt::matmul(&a, &b);
                        }),
                    ),
                ];
                for (path, mut f) in cells {
                    let ms = bench_cell(&dev, &mut f) * 1e3;
                    let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
                    rows.push(Row {
                        family: "contraction",
                        dtype,
                        shape: format!("{m}x{k}x{n}"),
                        path: path.to_string(),
                        ms,
                        metric: tflops,
                    });
                }
            },
            "c64" => {
                let data_c = |n: usize, seed: u64| -> Vec<Complex<f64>> {
                    let (re, im) = (data(n, seed), data(n, seed ^ 0x5f5f));
                    re.into_iter().zip(im).map(|(r, i)| Complex::new(r, i)).collect()
                };
                let av = data_c(m * k, 72);
                let bv = data_c(k * n, 73);
                let a = rt::asarray((av, &dev)).into_shape([m, k]);
                let b = rt::asarray((bv, &dev)).into_shape([k, n]);
                let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
                let cells: Vec<Cell> = vec![
                    (
                        "one-shot",
                        Box::new(|| {
                            let mut d = rt::zeros(([m, n], &dev));
                            cutensor_wrap::contract(
                                &dev,
                                a.view(),
                                &[0, 1],
                                ident(),
                                b.view(),
                                &[1, 2],
                                ident(),
                                None::<TensorView<'_, Complex<f64>, DeviceCuda, IxD>>,
                                d.view_mut(),
                                &[0, 2],
                                Complex::new(1.0, 0.0),
                                Complex::new(0.0, 0.0),
                                compute,
                                None,
                            )
                            .unwrap();
                        }),
                    ),
                    (
                        "cublas chain",
                        Box::new(|| {
                            let _ = rt::matmul(&a, &b);
                        }),
                    ),
                ];
                for (path, f) in cells {
                    let ms = bench_cell(&dev, f) * 1e3;
                    let tflops = 8.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
                    rows.push(Row {
                        family: "contraction",
                        dtype,
                        shape: format!("{m}x{k}x{n}"),
                        path: path.to_string(),
                        ms,
                        metric: tflops,
                    });
                }
            },
            _ => unreachable!(),
        }
    }

    // ---- multi-stream overlap: contraction on stream 1 vs a transfer on stream 2 ------------
    // Execute is unlocked by design (the wrapper's multi-stream premise): one handle serves an
    // f64 contraction on the default stream from this thread while a second thread drives a
    // 256 MiB transfer on another stream. Per sample, `seq` runs transfer then contraction back
    // to back and `par` runs them concurrently; the metric is 100 x seq/par (100 = serialized,
    // lower = more of the contraction hidden under the transfer). Transfers: D2H of a device
    // buffer, H2D of the same payload.
    {
        fn contract_f64(
            dev: &DeviceCuda,
            a: &Tensor<f64, DeviceCuda, IxD>,
            b: &Tensor<f64, DeviceCuda, IxD>,
            d: &mut Tensor<f64, DeviceCuda, IxD>,
            compute: cudarc::cutensor::sys::cutensorComputeDescriptor_t,
        ) {
            cutensor_wrap::contract(
                dev,
                a.view(),
                &[0, 1],
                ident(),
                b.view(),
                &[1, 2],
                ident(),
                None::<TensorView<'_, f64, DeviceCuda, IxD>>,
                d.view_mut(),
                &[0, 2],
                1.0f64,
                0.0f64,
                compute,
                None,
            )
            .expect("contraction");
        }
        let dev2 = dev.new_stream_f().expect("second stream");
        let (m, k, n) = (1024usize, 1024usize, 1024usize);
        let a = rt::asarray((data(m * k, 90), &dev)).into_shape([m, k]);
        let b = rt::asarray((data(k * n, 91), &dev)).into_shape([k, n]);
        let mut d = rt::asarray((vec![0.0f64; m * n], &dev)).into_shape([m, n]);
        let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
        let n_copy = (256usize << 20) / core::mem::size_of::<f64>();
        let big = std::sync::Arc::new(rt::asarray((data(n_copy, 92), &dev2)).into_shape([n_copy]));
        let host =
            std::sync::Arc::new(rt::asarray((data(n_copy, 92), &DeviceCpuSerial::default())).into_shape([n_copy]));
        let shape = format!("{m}x{k}x{n} vs {} MiB", (n_copy * 8) >> 20);
        let t_contract = once_ms(&dev, || contract_f64(&dev, &a, &b, &mut d, compute));
        let d2h = || {
            let big = big.clone();
            move || {
                let out: Vec<f64> = DeviceToCpuAPI::<f64>::to_cpu_vec(big.view().storage()).expect("d2h");
                std::hint::black_box(out.len());
            }
        };
        let h2d = || {
            let host = host.clone();
            let dev2 = dev2.clone();
            move || {
                let t = host.to_device_f(&dev2).expect("h2d");
                std::hint::black_box(&t);
            }
        };
        let t_d2h = once_ms(&dev2, d2h());
        let (d2h_seq, d2h_par) = overlap_ms(&dev, || contract_f64(&dev, &a, &b, &mut d, compute), d2h());
        let t_h2d = once_ms(&dev2, h2d());
        let (h2d_seq, h2d_par) = overlap_ms(&dev, || contract_f64(&dev, &a, &b, &mut d, compute), h2d());
        for (name, alone, seq_ms, par_ms) in [("d2h", t_d2h, d2h_seq, d2h_par), ("h2d", t_h2d, h2d_seq, h2d_par)] {
            let factor = 100.0 * seq_ms / par_ms;
            let mut push = |path: &str, ms: f64| {
                rows.push(Row {
                    family: "overlap",
                    dtype: "f64",
                    shape: shape.clone(),
                    path: path.to_string(),
                    ms,
                    metric: factor,
                });
            };
            push("contract alone", t_contract);
            push(&format!("{name} alone"), alone);
            push(&format!("{name} ; contract (seq)"), seq_ms);
            push(&format!("{name} || contract (par)"), par_ms);
        }
    }

    // ---- measured ceilings: large cuBLAS-chain GEMMs per dtype -------------------------------
    // The "maximum achievable" reference on this machine: the same matmul chain at a size large
    // enough to saturate. Theory: FP32 shader rate of consumer Blackwell; FP64 = FP32/64; c64
    // counted as 4 real flops per MAC.
    {
        let av64 = data(4096 * 4096, 80);
        let a64 = rt::asarray((av64, &dev)).into_shape([4096, 4096]);
        let bv64 = data(4096 * 4096, 81);
        let b64 = rt::asarray((bv64, &dev)).into_shape([4096, 4096]);
        let av32: Vec<f32> = data(8192 * 8192, 82).into_iter().map(|x| x as f32).collect();
        let a32 = rt::asarray((av32, &dev)).into_shape([8192, 8192]);
        let bv32: Vec<f32> = data(8192 * 8192, 83).into_iter().map(|x| x as f32).collect();
        let b32 = rt::asarray((bv32, &dev)).into_shape([8192, 8192]);
        let data_c = |n: usize, seed: u64| -> Vec<Complex<f64>> {
            let (re, im) = (data(n, seed), data(n, seed ^ 0x5f5f));
            re.into_iter().zip(im).map(|(r, i)| Complex::new(r, i)).collect()
        };
        let ac = rt::asarray((data_c(4096 * 4096, 84), &dev)).into_shape([4096, 4096]);
        let bc = rt::asarray((data_c(4096 * 4096, 85), &dev)).into_shape([4096, 4096]);
        let cells: Vec<CeilingCell> = vec![
            (
                "f64",
                "4096x4096x4096",
                4096,
                Box::new(|| {
                    let _ = rt::matmul(&a64, &b64);
                }),
            ),
            (
                "f32",
                "8192x8192x8192",
                8192,
                Box::new(|| {
                    let _ = rt::matmul(&a32, &b32);
                }),
            ),
            (
                "c64",
                "4096x4096x4096",
                4096,
                Box::new(|| {
                    let _ = rt::matmul(&ac, &bc);
                }),
            ),
        ];
        for (dtype, shape, n, mut f) in cells {
            let ms = bench_cell(&dev, &mut f) * 1e3;
            let tflops = 2.0 * n as f64 * n as f64 * n as f64 / (ms * 1e-3) / 1e12;
            rows.push(Row {
                family: "ceiling (measured)",
                dtype,
                shape: shape.to_string(),
                path: "cublas chain".to_string(),
                ms,
                metric: tflops,
            });
        }
    }

    // ---- trinary contraction (one-shot) -----------------------------------------------------
    {
        let (i, j, k, l) = (512usize, 512, 512, 512);
        let av = data(i * j, 74);
        let bv = data(j * k, 75);
        let cv = data(k * l, 76);
        let a = rt::asarray((av, &dev)).into_shape([i, j]);
        let b = rt::asarray((bv, &dev)).into_shape([j, k]);
        let c = rt::asarray((cv, &dev)).into_shape([k, l]);
        let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
        let f = || {
            let mut e = rt::zeros(([i, l], &dev));
            cutensor_wrap::contract_trinary(
                &dev,
                a.view(),
                &[0, 1],
                ident(),
                b.view(),
                &[1, 2],
                ident(),
                c.view(),
                &[2, 3],
                ident(),
                None::<TensorView<'_, f64, DeviceCuda, IxD>>,
                e.view_mut(),
                &[0, 3],
                1.0f64,
                0.0f64,
                compute,
                None,
            )
            .unwrap();
        };
        let ms = bench_cell(&dev, f) * 1e3;
        // two chained products: (a @ b) @ c
        let tflops = (2.0 * i as f64 * j as f64 * k as f64 + 2.0 * i as f64 * k as f64 * l as f64) / (ms * 1e-3) / 1e12;
        rows.push(Row {
            family: "trinary contraction",
            dtype: "f64",
            shape: format!("{i}x{j}x{k}x{l}"),
            path: "one-shot".to_string(),
            ms,
            metric: tflops,
        });
    }

    // ---- reduction: ij->j ADD, one-shot (GB/s of read A + write D) --------------------------
    {
        let (m, k) = (4096usize, 4096);
        let av = data(m * k, 77);
        let a = rt::asarray((av, &dev)).into_shape([m, k]);
        let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
        let f = || {
            let mut d = rt::zeros(([k], &dev));
            cutensor_wrap::reduce(
                &dev,
                a.view(),
                &[0, 1],
                ident(),
                None::<TensorView<'_, f64, DeviceCuda, IxD>>,
                d.view_mut(),
                &[1],
                add(),
                1.0f64,
                0.0f64,
                compute,
                None,
            )
            .unwrap();
        };
        let ms = bench_cell(&dev, f) * 1e3;
        let bytes = (m * k + k) as f64 * 8.0;
        rows.push(Row {
            family: "reduction",
            dtype: "f64",
            shape: format!("{m}x{k} -> {k}"),
            path: "one-shot".to_string(),
            ms,
            metric: bytes / (ms * 1e-3) / 1e9,
        });
    }

    // ---- elementwise binary: d = a + c, one-shot (GB/s of read A, C + write D) --------------
    {
        let (r, c) = (2048usize, 2048);
        let av = data(r * c, 78);
        let cv = data(r * c, 79);
        let a = rt::asarray((av, &dev)).into_shape([r, c]);
        let b = rt::asarray((cv, &dev)).into_shape([r, c]);
        let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
        let f = || {
            let mut d = rt::zeros(([r, c], &dev));
            cutensor_wrap::elementwise_binary(
                &dev,
                a.view(),
                &[0, 1],
                ident(),
                Some(b.view()),
                d.view_mut(),
                &[0, 1],
                add(),
                1.0f64,
                1.0f64,
                compute,
                None,
            )
            .unwrap();
        };
        let ms = bench_cell(&dev, f) * 1e3;
        let bytes = (r * c * 3) as f64 * 8.0;
        rows.push(Row {
            family: "elementwise",
            dtype: "f64",
            shape: format!("{r}x{c}"),
            path: "one-shot".to_string(),
            ms,
            metric: bytes / (ms * 1e-3) / 1e9,
        });
    }

    // ---- output: CSV then markdown ----------------------------------------------------------
    println!("family,dtype,shape,path,median_ms,metric");
    for r in &rows {
        println!("{},{},{},{},{:.3},{:.1}", r.family, r.dtype, r.shape, r.path, r.ms, r.metric);
    }
    println!("\n| family | dtype | shape | path | median ms | TFLOP/s, GB/s or overlap % |");
    println!("|---|---|---|---|---|---|");
    for r in &rows {
        println!("| {} | {} | {} | {} | {:.3} | {:.1} |", r.family, r.dtype, r.shape, r.path, r.ms, r.metric);
    }
}

fn ident() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY
}

fn add() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_ADD
}

#[cfg(not(cutensor_bindings))]
fn main() {}
