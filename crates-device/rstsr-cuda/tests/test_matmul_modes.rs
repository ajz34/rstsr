//! Compute modes of the matmul drivers, on hardware: every `(triple, mode)` of `f32` and `f64`
//! on each cuBLAS driver that serves it, against a CPU `f64` reference.
//!
//! The tolerance is per `(triple, mode)`: the exact modes are held to ordinary summation error,
//! the ones that down-convert the inputs (`Fast16F`, `Fast16BF`, `FastTF32`) to the precision of
//! the conversion, the fixed-point emulation to the width cuBLAS picks by default. A driver that
//! cannot serve a mode must say so: forced alone it is an error naming the reason, never a
//! silently different precision.
//!
//! The printed `max rel err` per mode is the measurement the tolerances were set from.

use rstsr::prelude::*;
use rstsr_cuda::dtype::CudaDType;
use rstsr_cuda::funcs;
use rstsr_cuda::matmul::{ComputeType32F, ComputeType64F};
use rstsr_cuda::prelude_dev::*;

/// Uniform data in `[-1, 1]`.
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

macro_rules! require_cuda {
    () => {
        if !std::panic::catch_unwind(funcs::is_available).unwrap_or(false) {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

fn device(drivers: MatMulDrivers, edit: impl FnOnce(&mut CudaRuntimeFlags)) -> DeviceCuda {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = drivers;
    edit(&mut flags);
    DeviceCuda::new(0).with_flags(flags)
}

/// The error of `a @ b` on `d` (which must fail), with `f64` operands.
fn matmul_err(d: &DeviceCuda, a: Vec<f64>, b: Vec<f64>) -> String {
    let (a, b) = (square(d, a, 2), square(d, b, 2));
    match rt::matmul_f(&a, &b) {
        Ok(_) => panic!("the matmul should have failed"),
        Err(e) => format!("{e:?}").chars().filter(|&c| c != '\n').collect(),
    }
}

/// The error of `a @ b` on `d` (which must fail), with `f32` operands rounded from `f64` data.
fn matmul_err_f32(d: &DeviceCuda, a: Vec<f64>, b: Vec<f64>) -> String {
    let (a, b) = (
        square(d, a.into_iter().map(|x| x as f32).collect(), 2),
        square(d, b.into_iter().map(|x| x as f32).collect(), 2),
    );
    match rt::matmul_f(&a, &b) {
        Ok(_) => panic!("the matmul should have failed"),
        Err(e) => format!("{e:?}").chars().filter(|&c| c != '\n').collect(),
    }
}

/// A `[n, n]` device tensor.
fn square<T: CudaDType + Copy + Clone + Send + Sync + 'static + num::Zero + num::One + core::ops::Mul<Output = T>>(
    d: &DeviceCuda,
    v: Vec<T>,
    n: usize,
) -> Tensor<T, DeviceCuda, IxD> {
    rt::asarray((v, d)).into_shape([n, n])
}

/// `max |got - want| / max |want|`, elementwise over the `f64` values of both.
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0, |s: f64, &x| s.max(x.abs()));
    got.iter().zip(want).map(|(&g, &w)| (g - w).abs()).fold(0.0, |e: f64, d| e.max(d)) / scale
}

/// `a @ b` on the device with `dtype` storage, compared with the CPU `f64` product of the
/// `f32`-rounded operands (for `f32`) or of the operands themselves (for `f64`).
fn check_f32(d: &DeviceCuda, what: &str, m: usize, k: usize, n: usize, tol: f64) {
    let dcpu = DeviceCpuSerial::default();
    let av = data(m * k, 1);
    let bv = data(k * n, 2);
    let a32: Tensor<f32, DeviceCpuSerial, _> =
        rt::asarray((av.iter().map(|&x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([m, k]);
    let b32: Tensor<f32, DeviceCpuSerial, _> =
        rt::asarray((bv.iter().map(|&x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([k, n]);
    let a64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((av.iter().map(|&x| x as f32 as f64).collect::<Vec<_>>(), &dcpu)).into_shape([m, k]);
    let b64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((bv.iter().map(|&x| x as f32 as f64).collect::<Vec<_>>(), &dcpu)).into_shape([k, n]);
    let want = rt::matmul(&a64, &b64);
    let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
    let got = rt::matmul(a32.to_device(d), b32.to_device(d));
    let got: Vec<f64> = DeviceToCpuAPI::<f32>::to_cpu_vec(got.to_contig(RowMajor).storage())
        .rstsr_unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let err = rel_err(&got, &want);
    eprintln!("  [f32 {what}] max rel err {err:.3e}");
    assert!(err <= tol, "{what}: max rel err {err:.3e} > {tol:.1e}");
}

fn check_f64(d: &DeviceCuda, what: &str, m: usize, k: usize, n: usize, tol: f64) {
    let dcpu = DeviceCpuSerial::default();
    let a: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(m * k, 3), &dcpu)).into_shape([m, k]);
    let b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(k * n, 4), &dcpu)).into_shape([k, n]);
    let want = rt::matmul(&a, &b);
    let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
    let got = rt::matmul(a.to_device(d), b.to_device(d));
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(got.to_contig(RowMajor).storage()).rstsr_unwrap();
    let err = rel_err(&got, &want);
    eprintln!("  [f64 {what}] max rel err {err:.3e}");
    assert!(err <= tol, "{what}: max rel err {err:.3e} > {tol:.1e}");
}

/// Whether this build's `cudarc` bindings carry the math-mode entry the classic driver needs.
fn classic_serves(mode: ComputeType32F) -> bool {
    use ComputeType32F::*;
    match mode {
        Default => true,
        Fast16BF => false,
        Pedantic | Fast16F | FastTF32 => cfg!(cublas_math_mode),
        _ => true,
    }
}

/// Every 32F mode on `cublasGemmEx` (always served) and on the classic entries (served where a
/// math mode exists for it), through the device flags.
#[test]
fn modes_32f() {
    require_cuda!();
    let (m, k, n) = (200usize, 300usize, 180usize);
    let cases: [(ComputeType32F, &str, f64); 5] = [
        (ComputeType32F::Default, "default", 1e-5),
        (ComputeType32F::Pedantic, "pedantic", 1e-5),
        (ComputeType32F::FastTF32, "fast tf32", 1e-2),
        (ComputeType32F::Fast16F, "fast 16f", 1e-2),
        (ComputeType32F::Fast16BF, "fast 16bf", 5e-2),
    ];
    for (mode, name, tol) in cases {
        let ex = device(MatMulDrivers::CUBLAS_EX, |f| f.compute_type_32f = mode);
        check_f32(&ex, &format!("{name}, cublasEx"), m, k, n, tol);
        // the default chain takes the same driver
        let all = device(MatMulDrivers::all(), |f| f.compute_type_32f = mode);
        check_f32(&all, &format!("{name}, chain"), m, k, n, tol);

        let classic = device(MatMulDrivers::CUBLAS, |f| f.compute_type_32f = mode);
        if classic_serves(mode) {
            check_f32(&classic, &format!("{name}, cublas"), m, k, n, tol);
        } else {
            let msg = matmul_err_f32(&classic, data(4, 5), data(4, 6));
            let why = if mode == ComputeType32F::Fast16BF { "bf16" } else { "cudarc bindings" };
            assert!(msg.contains("cublas: cannot serve") && msg.contains(why), "{name}: {msg}");
        }
    }
}

/// Every 64F mode on both drivers; the fixed-point emulation on the classic handle only.
#[test]
fn modes_64f() {
    require_cuda!();
    let (m, k, n) = (200usize, 300usize, 180usize);
    for (mode, name, tol) in
        [(ComputeType64F::Default, "default", 1e-13), (ComputeType64F::Pedantic, "pedantic", 1e-13)]
    {
        let ex = device(MatMulDrivers::CUBLAS_EX, |f| f.compute_type_64f = mode);
        check_f64(&ex, &format!("{name}, cublasEx"), m, k, n, tol);
        let all = device(MatMulDrivers::all(), |f| f.compute_type_64f = mode);
        check_f64(&all, &format!("{name}, chain"), m, k, n, tol);
        let classic = device(MatMulDrivers::CUBLAS, |f| f.compute_type_64f = mode);
        if cfg!(cublas_math_mode) {
            check_f64(&classic, &format!("{name}, cublas"), m, k, n, tol);
        }
    }

    let emu = ComputeType64F::EmulatedFixedPoint;
    let classic = device(MatMulDrivers::CUBLAS, |f| f.compute_type_64f = emu);
    let all = device(MatMulDrivers::all(), |f| f.compute_type_64f = emu);
    if cfg!(cublas_fixedpoint_math) {
        // the default mantissa width of cuBLAS: measured, then held with a margin
        check_f64(&classic, "fixed point, cublas", m, k, n, 1e-9);
        check_f64(&all, "fixed point, chain", m, k, n, 1e-9);
    } else {
        let msg = matmul_err(&classic, data(4, 5), data(4, 6));
        assert!(msg.contains("cublas: cannot serve") && msg.contains("fixed-point"), "{msg}");
    }
    // `cublasGemmEx` is not a candidate: no Ex-only chain, and the error says why
    let ex = device(MatMulDrivers::CUBLAS_EX, |f| f.compute_type_64f = emu);
    let msg = matmul_err(&ex, data(4, 5), data(4, 6));
    assert!(msg.contains("no enabled driver is a candidate"), "{msg}");
}

/// A mantissa width cannot be applied by this `cudarc`: with the emulation asked for it is an
/// error naming the field, and it is not read by any other mode.
#[test]
fn fixedpoint_mantissa_bits() {
    require_cuda!();
    let emu = device(MatMulDrivers::all(), |f| {
        f.compute_type_64f = ComputeType64F::EmulatedFixedPoint;
        f.fixedpoint_mantissa_bits = 30;
    });
    let msg = matmul_err(&emu, data(4, 5), data(4, 6));
    assert!(msg.contains("fixedpoint_mantissa_bits"), "{msg}");

    let d = device(MatMulDrivers::all(), |f| f.fixedpoint_mantissa_bits = 30);
    let _ = rt::matmul(square(&d, data(4, 5), 2), square(&d, data(4, 6), 2));
}

/// `cublasGemmEx` with the batched and the transposed argument shapes: a strided batch, and a
/// transposed operand broadcast against it.
#[test]
fn cublas_ex_batched_and_transposed() {
    require_cuda!();
    let d = device(MatMulDrivers::CUBLAS_EX, |_| {});
    let dcpu = DeviceCpuSerial::default();
    let (p, m, k, n) = (4usize, 32usize, 40usize, 24usize);
    let a: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(p * m * k, 7), &dcpu)).into_shape([p, m, k]);
    let b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(p * k * n, 8), &dcpu)).into_shape([p, k, n]);
    let want = rt::matmul(&a, &b);
    let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
    let got = rt::matmul(a.to_device(&d), b.to_device(&d));
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(got.to_contig(RowMajor).storage()).rstsr_unwrap();
    let err = rel_err(&got, &want);
    assert!(err <= 1e-13, "batched: max rel err {err:.3e}");

    // a transposed right operand against the broadcast batch of `a`
    let a2: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(n * k, 9), &dcpu)).into_shape([n, k]);
    let bt = a2.t();
    assert_eq!(bt.stride(), &[1, k as isize], "fixture: transposed operand");
    let want = rt::matmul(&a, &bt);
    let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
    let got = rt::matmul(a.to_device(&d), bt.to_device(&d));
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(got.to_contig(RowMajor).storage()).rstsr_unwrap();
    let err = rel_err(&got, &want);
    assert!(err <= 1e-13, "transposed b: max rel err {err:.3e}");
}
