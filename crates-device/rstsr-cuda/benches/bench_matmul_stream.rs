//! Overlapping H2D uploads with matmul on two streams (pinned host memory), against the serial
//! pageable flow and the controls that remove one ingredient each.
//!
//! Chunks of `[n, n]` operands are uploaded and multiplied, `c[i] = a[i] @ b[i]`:
//!
//! - `h2d`: the uploads alone, on stream H (pinned, non-blocking);
//! - `matmul`: the products alone, on stream C, from device-resident operands;
//! - `serial`: pageable host + blocking `to_device`, products on the default stream (the flow
//!   without pinned memory or streams);
//! - `pipeline_1stream`: pinned host, uploads and products on one stream (no overlap possible);
//! - `pipeline_drop`: two streams (H: uploads, C: products), the uploaded buffers dropped at the
//!   end of each iteration (correct; the frees order stream H behind the consumer of each chunk);
//! - `pipeline`: two streams, the uploaded buffers kept until the final synchronize.
//!
//! Issue order is per chunk (upload `i`, product `i`, upload `i+1`, ...): an event recorded at
//! issue time covers everything enqueued on its stream so far, so a product would wait for every
//! upload already issued, not for its own one only, if all uploads were issued first.
//!
//! Assertions: the same product on every path (bitwise equal to `serial` on the machine the
//! numbers below were taken on, else within 1e-4); `pipeline` not slower than `serial`;
//! `pipeline_1stream` and `pipeline_drop` above 90% of the sum of the phases (no overlap), which
//! is what makes the `pipeline` number an overlap. Timing ratios only, no absolute numbers.
//!
//! Run: cargo bench --bench bench_matmul_stream
//! Env: `RSTSR_BENCH_N` (matrix order, default 4096), `RSTSR_BENCH_CHUNKS` (default 4),
//! `RSTSR_BENCH_ROUNDS` (default 5), `RSTSR_BENCH_DTYPE` (`f32` default, or `f64`).

use std::time::{Duration, Instant};

use rstsr::prelude::*;
use rstsr_cuda::prelude_dev::*;

macro_rules! run {
    ($t:ty, $n:expr, $chunks:expr, $rounds:expr) => {
        run::<$t>($n, $chunks, $rounds)
    };
}

fn main() {
    if !rstsr_cuda::registry::is_available() {
        eprintln!("skipping: no CUDA device visible");
        return;
    }
    let n = env_usize("RSTSR_BENCH_N", 4096);
    let chunks = env_usize("RSTSR_BENCH_CHUNKS", 4);
    let rounds = env_usize("RSTSR_BENCH_ROUNDS", 5);
    match std::env::var("RSTSR_BENCH_DTYPE").as_deref() {
        Ok("f64") => run!(f64, n, chunks, rounds),
        Ok("f32") | Err(_) => run!(f32, n, chunks, rounds),
        Ok(other) => eprintln!("unknown RSTSR_BENCH_DTYPE {other:?}: use f32 or f64"),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Uniform data in `[-1, 1]`, as `T`.
fn data<T: Copy + From<f32>>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            T::from(x as f32)
        })
        .collect()
}

fn run<T>(n: usize, chunks: usize, rounds: usize)
where
    T: Copy + From<f32> + Send + Sync + 'static + num::Zero + num::One + core::ops::Mul<Output = T> + CudaMatMulMode,
    f64: From<T>,
{
    type Dev<T> = Tensor<T, DeviceCuda, IxD>;
    type Pin<T> = Tensor<T, DeviceCudaPinned, IxD>;
    type Cpu<T> = Tensor<T, DeviceCpuSerial, IxD>;

    let base = DeviceCuda::new(0);
    let pinned = DeviceCudaPinned::new(0);
    let cpu = DeviceCpuSerial::default();
    let dev_h = base.new_stream();
    let dev_c = base.new_stream();
    let mib = |count: usize| (count * n * n * core::mem::size_of::<T>()) as f64 / 1048576.0;
    println!("{chunks} chunks of [{n}, {n}] {} ({:.0} MiB uploaded per chunk)", std::any::type_name::<T>(), mib(2));

    // fixtures, outside every timed region
    let host = |seed: u64| -> Vec<T> { data(n * n, seed) };
    let pin = |seed: u64| -> Pin<T> { pinned.tensor_f(pinned.from_vec_f(host(seed)).unwrap(), [n, n]).unwrap() };
    let host_tensor = |seed: u64| -> Cpu<T> { rt::asarray((host(seed), &cpu)).into_shape([n, n]) };
    let pin_a: Vec<Pin<T>> = (0..chunks).map(|c| pin(1 + 10 * c as u64)).collect();
    let pin_b: Vec<Pin<T>> = (0..chunks).map(|c| pin(2 + 10 * c as u64)).collect();
    let cpu_a: Vec<Cpu<T>> = (0..chunks).map(|c| host_tensor(1 + 10 * c as u64)).collect();
    let cpu_b: Vec<Cpu<T>> = (0..chunks).map(|c| host_tensor(2 + 10 * c as u64)).collect();
    let read = |t: &Dev<T>| -> Vec<f64> {
        DeviceToCpuAPI::<T>::to_cpu_vec(t.to_contig(RowMajor).storage())
            .rstsr_unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    };
    let rel_err = |got: &[Vec<f64>], want: &[Vec<f64>]| -> f64 {
        got.iter()
            .flatten()
            .zip(want.iter().flatten())
            .map(|(&g, &w)| (g - w).abs() / (1.0 + w.abs()))
            .fold(0.0, f64::max)
    };

    // warm every path once: the cuBLASLt heuristic cache, the per-stream workspaces, the device
    // tensors of the first product (the JIT has no part in a cuBLAS product)
    let dev_a: Vec<Dev<T>> = pin_a.iter().map(|t| t.to_device(&dev_c)).collect();
    let dev_b: Vec<Dev<T>> = pin_b.iter().map(|t| t.to_device(&dev_c)).collect();
    let warm: Vec<_> = dev_a.iter().zip(&dev_b).map(|(a, b)| rt::matmul(a, b)).collect();
    dev_c.synchronize();
    drop(warm);

    let phase_h2d = || -> Duration {
        let t0 = Instant::now();
        let keep: Vec<_> = pin_a.iter().chain(&pin_b).map(|t| t.to_device(&dev_h)).collect();
        dev_h.synchronize();
        let d = t0.elapsed();
        drop(keep);
        d
    };
    let phase_matmul = || -> Duration {
        let t0 = Instant::now();
        let keep: Vec<_> = dev_a.iter().zip(&dev_b).map(|(a, b)| rt::matmul(a, b)).collect();
        dev_c.synchronize();
        let d = t0.elapsed();
        drop(keep);
        d
    };
    let serial = || -> (Duration, Vec<Vec<f64>>) {
        let t0 = Instant::now();
        let keep: Vec<_> =
            cpu_a.iter().zip(&cpu_b).map(|(a, b)| rt::matmul(a.to_device(&base), b.to_device(&base))).collect();
        base.synchronize();
        let d = t0.elapsed();
        (d, keep.iter().map(&read).collect())
    };
    let pipeline = |two_streams: bool, keep_alive: bool| -> (Duration, Vec<Vec<f64>>) {
        let c = if two_streams { &dev_c } else { &dev_h };
        let t0 = Instant::now();
        let mut products: Vec<Dev<T>> = Vec::new();
        let mut keep: Vec<(Dev<T>, Dev<T>)> = Vec::new();
        for (x, y) in pin_a.iter().zip(&pin_b) {
            let a = x.to_device(&dev_h); // H2D on H, enqueued
            let b = y.to_device(&dev_h);
            // on C; ordered after the upload of this chunk only
            products.push(rt::matmul(a.to_device(c), b.to_device(c)));
            if keep_alive {
                keep.push((a, b));
            }
        }
        dev_h.synchronize();
        c.synchronize();
        let d = t0.elapsed();
        (d, products.iter().map(&read).collect())
    };

    let names = ["h2d", "matmul", "serial", "pipeline_1stream", "pipeline_drop", "pipeline"];
    let mut acc: Vec<Vec<Duration>> = vec![Vec::new(); names.len()];
    let mut out: Vec<Vec<Vec<f64>>> = vec![Vec::new(); names.len()];
    for r in 0..rounds {
        acc[0].push(phase_h2d());
        acc[1].push(phase_matmul());
        let (t, o) = serial();
        acc[2].push(t);
        out[2] = o;
        let (t, o) = pipeline(false, true);
        acc[3].push(t);
        out[3] = o;
        let (t, o) = pipeline(true, false);
        acc[4].push(t);
        out[4] = o;
        let (t, o) = pipeline(true, true);
        acc[5].push(t);
        out[5] = o;
        let _ = r;
    }
    let med: Vec<Duration> = acc.iter().map(|v| median(&mut v.clone())).collect();
    let (t_h, t_m, t_s, t_1, t_drop, t_p) = (med[0], med[1], med[2], med[3], med[4], med[5]);
    let bytes = (chunks * 2 * n * n * core::mem::size_of::<T>()) as f64;
    let flops = 2.0 * chunks as f64 * (n as f64).powi(3);
    println!();
    println!("phase totals (isolated, interleaved median over {rounds} rounds)");
    println!("  h2d      {:8.2} ms   {:.1} GB/s", ms(t_h), bytes / t_h.as_secs_f64() / 1e9);
    println!("  matmul   {:8.2} ms   {:.1} TFLOP/s", ms(t_m), flops / t_m.as_secs_f64() / 1e12);
    println!("  predicted serial (sum)         {:8.2} ms", ms(t_h + t_m));
    println!("  predicted pipeline (max term)  {:8.2} ms", ms(t_h.max(t_m)));
    println!();
    println!("mode                ms   vs serial   overlap");
    for (name, m) in names.iter().zip(&med) {
        let overlap = match *name {
            "pipeline" | "pipeline_drop" | "pipeline_1stream" => Some(ms(t_h.max(t_m)) / ms(*m) * 100.0),
            _ => None,
        };
        println!(
            "{name:<16} {:8.2}     {:5.2}x   {}",
            ms(*m),
            ms(t_s) / ms(*m),
            overlap.map_or(String::new(), |o| format!("{o:5.1}%")),
        );
    }
    println!();
    println!("time saved against the sum of the phases: {:.1}%", (1.0 - ms(t_p) / ms(t_h + t_m)) * 100.0);

    // the same product on every path
    for (name, o) in names.iter().skip(2).zip(&out[2..]) {
        let bitwise = o.iter().flatten().zip(out[2].iter().flatten()).all(|(&a, &b)| a.to_bits() == b.to_bits());
        let err = rel_err(o, &out[2]);
        println!("{name:<16} bitwise equal to serial: {bitwise}; max rel err {err:.1e}");
        assert!(err <= 1e-4, "{name}: max rel err {err:.1e}");
    }
    // the overlap is real: the pipeline is not slower than the serial flow, and the two controls
    // (one stream; frees per chunk) sit at the sum of the phases
    assert!(ms(t_p) <= 1.05 * ms(t_s), "pipeline {:.2} ms is slower than serial {:.2} ms", ms(t_p), ms(t_s));
    let sum = ms(t_h + t_m);
    assert!(ms(t_1) > 0.9 * sum, "pipeline_1stream {:.2} ms is below the phase sum {sum:.2} ms", ms(t_1));
    assert!(ms(t_drop) > 0.9 * sum, "pipeline_drop {:.2} ms is below the phase sum {sum:.2} ms", ms(t_drop));
    println!("assertions passed: same product on every path; pipeline <= serial; the controls sit at the phase sum");
}
