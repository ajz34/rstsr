//! Compute modes of the matmul drivers, on hardware: every `(triple, mode)` of `f32` and `f64`
//! on each cuBLAS driver that serves it, against a CPU `f64` reference.
//!
//! The tolerance is per `(triple, mode)`: the exact modes are held to ordinary summation error,
//! the ones that down-convert the inputs (`Fast16F`, `Fast16BF`, `FastTF32`) to the precision of
//! the conversion, the fixed-point emulation to the width cuBLAS picks by default. A driver that
//! cannot serve a mode must say so: forced alone it is an error naming the reason, never a
//! silently different precision. cuBLASLt serves every 32F mode (`Fast16BF` included) and
//! `Default`/`Pedantic` of 64F; the argument shapes (batch, broadcast, transposes) run on each
//! cuBLAS driver alone, and pointer alignment on cuBLASLt.
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

/// Every 32F mode on cuBLASLt and `cublasGemmEx` (always served) and on the classic entries
/// (served where a math mode exists for it), through the device flags.
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
        let lt = device(MatMulDrivers::CUBLAS_LT, |f| f.compute_type_32f = mode);
        check_f32(&lt, &format!("{name}, cublasLt"), m, k, n, tol);
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

/// Every 64F mode on the three drivers; the fixed-point emulation on the classic handle only.
#[test]
fn modes_64f() {
    require_cuda!();
    let (m, k, n) = (200usize, 300usize, 180usize);
    for (mode, name, tol) in
        [(ComputeType64F::Default, "default", 1e-13), (ComputeType64F::Pedantic, "pedantic", 1e-13)]
    {
        let lt = device(MatMulDrivers::CUBLAS_LT, |f| f.compute_type_64f = mode);
        check_f64(&lt, &format!("{name}, cublasLt"), m, k, n, tol);
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
    // neither cuBLASLt nor `cublasGemmEx` is a candidate: no Lt-only or Ex-only chain
    for forced in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX] {
        let d = device(forced, |f| f.compute_type_64f = emu);
        let msg = matmul_err(&d, data(4, 5), data(4, 6));
        assert!(msg.contains("no enabled driver is a candidate"), "{msg}");
    }
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

/// The batched and the transposed argument shapes, on each cuBLAS driver alone: a strided batch,
/// a transposed right operand broadcast against the batch of `a`, a transposed left operand, and
/// a broadcast batch of `a` against a batched `b`. The batch stride of cuBLASLt's layouts is an
/// element count, like the strided-batched entries: this is where a byte count would show.
#[test]
fn cublas_batched_and_transposed() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (p, m, k, n) = (4usize, 32usize, 40usize, 24usize);

    let a: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(p * m * k, 7), &dcpu)).into_shape([p, m, k]);
    let b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(p * k * n, 8), &dcpu)).into_shape([p, k, n]);
    let c2: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(n * k, 9), &dcpu)).into_shape([n, k]);
    let a2: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(m * k, 10), &dcpu)).into_shape([m, k]);
    let at: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data(m * k, 11), &dcpu)).into_shape([k, m]).t().to_owned();

    type Cpu = Tensor<f64, DeviceCpuSerial, IxD>;
    let cases: [(&str, Cpu, Cpu); 4] = [
        ("batched", a.clone(), b.clone()),
        ("transposed b, broadcast", a.clone(), c2.t().to_owned()),
        ("transposed a", at, b.clone()),
        ("a broadcast over the batch", a2.clone(), b.clone()),
    ];
    for forced in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::CUBLAS] {
        let d = device(forced, |f| f.trace_matmul = true);
        for (what, a, b) in &cases {
            let want = rt::matmul(a, b);
            let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
            let got = rt::matmul(a.to_device(&d), b.to_device(&d));
            let got = DeviceToCpuAPI::<f64>::to_cpu_vec(got.to_contig(RowMajor).storage()).rstsr_unwrap();
            let err = rel_err(&got, &want);
            assert!(err <= 1e-13, "{what}, {forced:?}: max rel err {err:.3e}");
        }
    }
}

/// Operands and output that are views one element into their buffers: the addresses are not
/// 16-byte aligned. The chain must get the product right, and cuBLASLt alone must either get it
/// right or decline (an algorithm's alignment requirement is checked by `cublasLtMatmul`).
#[test]
fn cublas_lt_misaligned_views() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (32usize, 40usize, 24usize);
    // an owning tensor whose layout starts one element into a buffer of `total + 1` (addresses
    // off the 16-byte grid), on the device; the data also as a CPU vector
    let shifted =
        |rows: usize, cols: usize, seed: u64, d: &DeviceCuda| -> (Vec<f64>, Vec<f64>, Tensor<f64, DeviceCuda, IxD>) {
            let total = rows * cols;
            let v = data(total + 1, seed);
            let t: Tensor<f64, DeviceCuda, _> = rt::asarray((v.clone(), d));
            let (storage, _) = t.into_raw_parts();
            let layout = Layout::<IxD>::new(vec![rows, cols], vec![cols as isize, 1], 1).unwrap();
            (v[1..].to_vec(), v, Tensor::new(storage, layout))
        };
    for forced in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::all()] {
        let d = device(forced, |f| f.trace_matmul = true);
        let (av, _, a) = shifted(m, k, 12, &d);
        let (bv, _, b) = shifted(k, n, 13, &d);
        let (cv, cv_full, mut c) = shifted(m, n, 14, &d);
        assert_eq!((a.offset(), b.offset(), c.offset()), (1, 1, 1), "fixture: shifted views");
        let want =
            rt::matmul(rt::asarray((av, &dcpu)).into_shape([m, k]), rt::asarray((bv, &dcpu)).into_shape([k, n]));
        let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
        // only cuBLASLt alone may refuse: the chain has the other drivers behind it
        let refused =
            |e: &dyn core::fmt::Debug| forced == MatMulDrivers::CUBLAS_LT && format!("{e:?}").contains("cublasLt");

        // the fresh output of `a @ b`
        match rt::matmul_f(&a, &b) {
            Ok(got) => {
                let got = DeviceToCpuAPI::<f64>::to_cpu_vec(got.to_contig(RowMajor).storage()).rstsr_unwrap();
                let err = rel_err(&got, &want);
                assert!(err <= 1e-13, "fresh output, {forced:?}: max rel err {err:.3e}");
            },
            Err(e) => assert!(refused(&e), "{forced:?}: {e:?}"),
        }
        // `c = 2 a b + c`, into the shifted `c` (read and written in place)
        let before = DeviceToCpuAPI::<f64>::to_cpu_vec(c.to_contig(RowMajor).storage()).rstsr_unwrap();
        eprintln!("  c before matches fixture: {}", rel_err(&before, &cv));
        {
            // the same offset layout on the CPU device, read back the same way
            let cpu = Tensor::<f64, DeviceCpuSerial, IxD>::new(
                Storage::new(DataOwned::from(cv_full.clone()), dcpu.clone()),
                Layout::<IxD>::new(vec![m, n], vec![n as isize, 1], 1).unwrap(),
            );
            let back = DeviceToCpuAPI::<f64>::to_cpu_vec(cpu.to_contig(RowMajor).storage()).rstsr_unwrap();
            eprintln!("  cpu offset read-back matches: {}", rel_err(&back, &cv));
            let dev_plain = rt::asarray((cv.clone(), &d)).into_shape([m, n]);
            let plain = DeviceToCpuAPI::<f64>::to_cpu_vec(dev_plain.to_contig(RowMajor).storage()).rstsr_unwrap();
            eprintln!("  cuda plain read-back matches: {}", rel_err(&plain, &cv));
            let raw = DeviceToCpuAPI::<f64>::to_cpu_vec(c.to_contig(RowMajor).storage()).rstsr_unwrap();
            eprintln!(
                "  before[0..3] = {:?} v[0..3] = {:?} v[1..4] = {:?}",
                &before[0..3],
                &cv_full[0..3],
                &cv_full[1..4]
            );
            eprintln!("  cpu back[0..3] = {:?}", &back[0..3]);
            eprintln!("  rel(before, v[0..]) = {}", rel_err(&before, &cv_full[..768]));
            let _ = raw;
        }
        eprintln!("  want[0..3] = {:?} cv[0..3] = {:?}", &want[0..3], &cv[0..3]);
        match op_mutc_refa_refb_matmul(&mut c, &a, &b, 2.0, 1.0) {
            Ok(()) => {
                // a fresh elementwise result reads `c` through its layout (a whole-storage read
                // would start at the buffer, not at the offset)
                let out = &c + 0.0;
                let got = DeviceToCpuAPI::<f64>::to_cpu_vec(out.to_contig(RowMajor).storage()).rstsr_unwrap();
                let err = rel_err(&got, &(0..want.len()).map(|i| 2.0 * want[i] + cv[i]).collect::<Vec<_>>());
                assert!(err <= 1e-13, "beta path, {forced:?}: max rel err {err:.3e}");
            },
            Err(e) => assert!(refused(&e), "{forced:?}: {e:?}"),
        }
    }
}
