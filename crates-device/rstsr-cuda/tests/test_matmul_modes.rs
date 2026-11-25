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

use half::{bf16, f16};
use num::Complex;
use rstsr::prelude::*;
use rstsr_cuda::dtype::CudaDType;
use rstsr_cuda::funcs;
use rstsr_cuda::matmul::{matmul_with_mode_f, ComputeType16F, ComputeType32F, ComputeType64F, TensorCudaMatMulAPI};
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
        let want = rt::matmul(rt::asarray((av, &dcpu)).into_shape([m, k]), rt::asarray((bv, &dcpu)).into_shape([k, n]));
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

type Cpu<T> = Tensor<T, DeviceCpuSerial, IxD>;

/// `a @ b` with the operands stored as `S` (rounded from the `f64` data) and the output `C`, on
/// `d`, against the CPU `f64` product of the **rounded** operands.
#[allow(clippy::too_many_arguments)]
fn check_rounded<S, C>(
    d: &DeviceCuda,
    what: &str,
    m: usize,
    k: usize,
    n: usize,
    tol: f64,
    to_s: fn(f64) -> S,
    of_s: fn(S) -> f64,
    of_c: fn(C) -> f64,
) where
    S: CudaDType + CudaMatMulMode + Copy + Clone + Send + Sync + 'static + num::Zero,
    C: CudaDType + CudaMatMulMode + Copy + Clone + Send + Sync + 'static + num::Zero + num::One,
{
    let dcpu = DeviceCpuSerial::default();
    let av: Vec<S> = data(m * k, 1).into_iter().map(to_s).collect();
    let bv: Vec<S> = data(k * n, 2).into_iter().map(to_s).collect();
    let up = |v: Vec<S>, cnt: usize| -> Vec<f64> {
        assert_eq!(v.len(), cnt);
        v.into_iter().map(of_s).collect()
    };
    let a64: Cpu<f64> = rt::asarray((up(av.clone(), m * k), &dcpu)).into_shape([m, k]);
    let b64: Cpu<f64> = rt::asarray((up(bv.clone(), k * n), &dcpu)).into_shape([k, n]);
    let want = rt::matmul(&a64, &b64);
    let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();

    let a: Tensor<S, DeviceCuda, IxD> = rt::asarray((av, d)).into_shape([m, k]);
    let b: Tensor<S, DeviceCuda, IxD> = rt::asarray((bv, d)).into_shape([k, n]);
    let mut c: Tensor<C, DeviceCuda, IxD> = rt::full(([m * n], C::one(), d)).into_shape([m, n]);
    a.matmul_with_output::<S, C, IxD, IxD>(&b, &mut c.view_mut());
    let got: Vec<f64> =
        DeviceToCpuAPI::<C>::to_cpu_vec(c.to_contig(RowMajor).storage()).rstsr_unwrap().into_iter().map(of_c).collect();
    let err = rel_err(&got, &want);
    eprintln!("  [{what}] max rel err {err:.3e}");
    assert!(err <= tol, "{what}: max rel err {err:.3e} > {tol:.1e}");
}

/// The cuBLAS drivers alone, then the chain. The classic entries take `Pedantic` as a handle math
/// mode, which this build's bindings carry from CUDA 12.9 on: without them the classic driver is
/// left out of a `Pedantic` run (it refuses, as `modes_32f` checks).
fn drivers(classic_serves_pedantic: bool) -> impl Iterator<Item = MatMulDrivers> {
    let all = [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::CUBLAS, MatMulDrivers::all()];
    all.into_iter().filter(move |f| classic_serves_pedantic || *f != MatMulDrivers::CUBLAS)
}

/// `c32` and `c64` on each cuBLAS driver, with `Default` and `Pedantic`; the `Fast*` modes and
/// the fixed-point emulation resolve to the default for complex outputs.
#[test]
fn modes_complex() {
    require_cuda!();
    let (m, k, n) = (64usize, 96usize, 48usize);
    for (mode, name) in [(ComputeType32F::Default, "default"), (ComputeType32F::Pedantic, "pedantic")] {
        for forced in drivers(mode == ComputeType32F::Default || cfg!(cublas_math_mode)) {
            let d = device(forced, |f| f.compute_type_32f = mode);
            check_rounded::<Complex<f32>, Complex<f32>>(
                &d,
                &format!("c32 {name}, {forced:?}"),
                m,
                k,
                n,
                1e-5,
                |x| Complex::new(x as f32, 0.0),
                |z| z.re as f64,
                |z| z.re as f64,
            );
        }
    }
    for (mode, name) in [(ComputeType64F::Default, "default"), (ComputeType64F::Pedantic, "pedantic")] {
        for forced in drivers(mode == ComputeType64F::Default || cfg!(cublas_math_mode)) {
            let d = device(forced, |f| f.compute_type_64f = mode);
            check_rounded::<Complex<f64>, Complex<f64>>(
                &d,
                &format!("c64 {name}, {forced:?}"),
                m,
                k,
                n,
                1e-13,
                Complex::from,
                |z| z.re,
                |z| z.re,
            );
        }
    }
    // the default chain (naive) gets the same answer for the modes cuBLAS cannot express
    for mode in [ComputeType32F::FastTF32, ComputeType32F::Fast16F, ComputeType32F::Fast16BF] {
        let d = device(MatMulDrivers::all(), |f| f.compute_type_32f = mode);
        check_rounded::<Complex<f32>, Complex<f32>>(
            &d,
            "c32 fast mode -> default, chain",
            m,
            k,
            n,
            1e-5,
            |x| Complex::new(x as f32, 0.0),
            |z| z.re as f64,
            |z| z.re as f64,
        );
    }
    // the fixed-point emulation is real `f64` only: a `c64` output keeps the default
    let d = device(MatMulDrivers::all(), |f| f.compute_type_64f = ComputeType64F::EmulatedFixedPoint);
    check_rounded::<Complex<f64>, Complex<f64>>(
        &d,
        "c64 fixed point -> default, chain",
        m,
        k,
        n,
        1e-13,
        Complex::from,
        |z| z.re,
        |z| z.re,
    );
}

/// The half triples on cuBLASLt and `cublasGemmEx`, and through the default chain: `f16 -> f16`
/// in every 16F mode, `f16 -> f32`, `bf16 -> bf16`, `bf16 -> f32` in the default and `Pedantic`.
/// The tolerance is the rounding of the **output** storage (the operands are rounded before the
/// reference is taken).
#[test]
fn modes_half() {
    require_cuda!();
    let (m, k, n) = (80usize, 256usize, 40usize);
    for (mode, name) in [
        (ComputeType16F::Default, "default"),
        (ComputeType16F::Pedantic, "pedantic"),
        (ComputeType16F::Acc16F, "acc16f"),
    ] {
        // Acc16F keeps fewer bits of the sum: measured, then held with a margin
        let tol = if mode == ComputeType16F::Acc16F { 5e-3 } else { 1e-2 };
        for forced in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::all()] {
            let d = device(forced, |f| f.compute_type_16f = mode);
            check_rounded::<f16, f16>(
                &d,
                &format!("f16 -> f16 {name}, {forced:?}"),
                m,
                k,
                n,
                tol,
                f16::from_f64,
                f64::from,
                f64::from,
            );
        }
    }
    for (mode, name) in [(ComputeType32F::Default, "default"), (ComputeType32F::Pedantic, "pedantic")] {
        for forced in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::all()] {
            let d = device(forced, |f| f.compute_type_32f = mode);
            check_rounded::<f16, f32>(
                &d,
                &format!("f16 -> f32 {name}, {forced:?}"),
                m,
                k,
                n,
                1e-5,
                f16::from_f64,
                f64::from,
                f64::from,
            );
            check_rounded::<bf16, f32>(
                &d,
                &format!("bf16 -> f32 {name}, {forced:?}"),
                m,
                k,
                n,
                1e-5,
                bf16::from_f64,
                f64::from,
                f64::from,
            );
        }
    }
    for (mode, name) in [
        (ComputeType16F::Default, "default"),
        (ComputeType16F::Pedantic, "pedantic"),
        (ComputeType16F::Acc16F, "acc16f"),
    ] {
        for forced in [MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX, MatMulDrivers::all()] {
            let d = device(forced, |f| f.compute_type_16f = mode);
            check_rounded::<bf16, bf16>(
                &d,
                &format!("bf16 -> bf16 {name}, {forced:?}"),
                m,
                k,
                n,
                1e-2,
                bf16::from_f64,
                f64::from,
                f64::from,
            );
        }
    }
    // a 32F `Fast*` mode does not apply to half inputs with an `f32` output: the default
    let d = device(MatMulDrivers::all(), |f| f.compute_type_32f = ComputeType32F::FastTF32);
    check_rounded::<f16, f32>(
        &d,
        "f16 -> f32 fast tf32 -> default, chain",
        m,
        k,
        n,
        1e-5,
        f16::from_f64,
        f64::from,
        f64::from,
    );
}

/// The entries that pass the mode per call: `matmul_from_mode`, `matmul_with_mode` and
/// `matmul_with_output_mode`, with `f16` operands into an `f32` output and with `f32` operands,
/// whatever the device flags say. `rt::matmul` and `%` infer the `f16` output from `Mul`.
#[test]
fn tensor_level_modes() {
    require_cuda!();
    let d = device(MatMulDrivers::all(), |f| f.compute_type_32f = ComputeType32F::Default);
    let (m, k, n) = (24usize, 64usize, 12usize);
    let a: Tensor<f16, DeviceCuda, IxD> =
        rt::asarray((data(m * k, 1).into_iter().map(f16::from_f64).collect::<Vec<_>>(), &d)).into_shape([m, k]);
    let b: Tensor<f16, DeviceCuda, IxD> =
        rt::asarray((data(k * n, 2).into_iter().map(f16::from_f64).collect::<Vec<_>>(), &d)).into_shape([k, n]);

    let plain: Tensor<f32, DeviceCuda, IxD> =
        matmul_with_mode_f::<f16, f16, f32, IxD, IxD, IxD>(&a, &b, 1.0, ComputeType32F::Pedantic).rstsr_unwrap();
    let with_output: Tensor<f32, DeviceCuda, IxD> =
        a.matmul_with_mode::<f16, f32, IxD, IxD>(&b, ComputeType32F::Pedantic);
    let mut c: Tensor<f32, DeviceCuda, IxD> = rt::full(([m * n], 2.0f32, &d)).into_shape([m, n]);
    c.matmul_from_mode(&a, &b, 1.0, 0.0, ComputeType32F::Pedantic);
    let mut scaled: Tensor<f32, DeviceCuda, IxD> = rt::full(([m * n], 2.0f32, &d)).into_shape([m, n]);
    scaled.matmul_from_mode(&a, &b, 3.0, 0.5, ComputeType32F::Pedantic);
    let read = |t: &Tensor<f32, DeviceCuda, IxD>| -> Vec<f32> {
        DeviceToCpuAPI::<f32>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
    };
    let (p, o, s) = (read(&plain), read(&c), read(&scaled));
    assert_eq!(p, read(&with_output), "matmul_with_output_mode");
    assert_eq!(p, o, "matmul_from_mode");
    for (x, y) in p.iter().zip(&s) {
        assert!((3.0 * x + 1.0 - y).abs() <= 1e-5 * (1.0 + (3.0 * x).abs()), "matmul_from_mode with alpha, beta");
    }
    // f32 operands: the mode given, not the flags'
    let a32: Tensor<f32, DeviceCuda, IxD> =
        rt::asarray((data(m * k, 3).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &d)).into_shape([m, k]);
    let b32: Tensor<f32, DeviceCuda, IxD> =
        rt::asarray((data(k * n, 4).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &d)).into_shape([k, n]);
    let pedantic: Tensor<f32, DeviceCuda, IxD> =
        a32.matmul_with_mode::<f32, f32, IxD, IxD>(&b32, ComputeType32F::Pedantic);
    let fast: Tensor<f32, DeviceCuda, IxD> = a32.matmul_with_mode::<f32, f32, IxD, IxD>(&b32, ComputeType32F::FastTF32);
    assert_ne!(read(&pedantic), read(&fast), "the mode of the call is the one run");

    // operands of different dtypes need no mode entry: `matmul_from` has no `Mul` bound, and the
    // chain takes the mixed triple (on the naive driver: the reference is the `f32` product)
    let b64: Tensor<f64, DeviceCuda, IxD> =
        rt::asarray((data(k * n, 2).into_iter().map(|x| x as f32 as f64).collect::<Vec<_>>(), &d)).into_shape([k, n]);
    let a32h: Tensor<f32, DeviceCuda, IxD> =
        rt::asarray((data(m * k, 1).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &d)).into_shape([m, k]);
    let mut mixed: Tensor<f64, DeviceCuda, IxD> = rt::full(([m * n], f64::NAN, &d)).into_shape([m, n]);
    mixed.matmul_from(&a32h, &b64, 1.0, 0.0);
    let mixed = DeviceToCpuAPI::<f64>::to_cpu_vec(mixed.to_contig(RowMajor).storage()).rstsr_unwrap();
    assert!(mixed.iter().all(|x| x.is_finite()), "f32 @ f64 -> f64 on the chain");

    // `rt::matmul` and `%` give the `f16` output
    let h: Tensor<f16, DeviceCuda, IxD> = rt::matmul(&a, &b);
    let op: Tensor<f16, DeviceCuda, IxD> = &a % &b;
    let host = |t: &Tensor<f16, DeviceCuda, IxD>| -> Vec<f16> {
        DeviceToCpuAPI::<f16>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
    };
    assert_eq!(host(&h), host(&op));
    let (hp, hh) = (read(&plain), host(&h));
    for (x, y) in hp.iter().zip(hh) {
        assert!((x - f32::from(y)).abs() <= 1e-2 * (1.0 + x.abs()), "f16 output against the f32 output");
    }
}

/// The argument shapes of the layout matrix: a plain product, a strided batch, a transposed right
/// operand broadcast against the batch of `a`, a transposed left operand, a broadcast batch of `a`
/// against a batched `b`, and a stepped (strided) `a`. `mk(data(seed, count), shape)` makes a
/// tensor and `sh(batch, [m, k])` places the batch axis as the default order has it (leading
/// under row-major, trailing under column-major), so the same cases are built on the CPU (`f64`
/// of the rounded values) and on the device (the storage type), each in its own order.
macro_rules! shape_cases {
    ($mk:expr, $sh:expr, $data:expr, $m:expr, $k:expr, $n:expr, $p:expr) => {{
        let (m, k, n, p) = ($m, $k, $n, $p);
        let a = $mk($data(7, p * m * k), $sh(p, [m, k]));
        let b = $mk($data(8, p * k * n), $sh(p, [k, n]));
        let c2 = $mk($data(9, n * k), $sh(1, [n, k]));
        let a2 = $mk($data(10, m * k), $sh(1, [m, k]));
        let at = $mk($data(11, k * m), $sh(1, [k, m]));
        let stepped0 = $mk($data(12, 4 * m * k), $sh(1, [2 * m, 2 * k]));
        let stepped = stepped0.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
        vec![
            ("plain", $mk($data(13, m * k), $sh(1, [m, k])), $mk($data(14, k * n), $sh(1, [k, n]))),
            ("batched", a.clone(), b.clone()),
            ("transposed b, broadcast", a.clone(), c2.t().to_owned()),
            ("transposed a", at.t().to_owned(), b.clone()),
            ("a broadcast over the batch", a2.clone(), b.clone()),
            ("stepped a", stepped.to_owned(), $mk($data(15, k * n), $sh(1, [k, n]))),
        ]
    }};
}

/// `[batch, dims]` under row-major, `[dims, batch]` under column-major (a unit batch is dropped).
fn sh(order: FlagOrder, batch: usize, dims: [usize; 2]) -> Vec<usize> {
    let mut out = Vec::new();
    if batch > 1 && order == RowMajor {
        out.push(batch);
    }
    out.extend_from_slice(&dims);
    if batch > 1 && order == ColMajor {
        out.push(batch);
    }
    out
}

/// Every serving cuBLAS driver alone, the naive kernel (the on-device oracle) and the chain, for
/// `to_s` operands stored as `S` and the output `C`, in both default orders.
#[allow(clippy::too_many_arguments)]
fn matrix<S, C>(what: &str, cublas_classic: bool, to_s: fn(f64) -> S, of_s: fn(S) -> f64, of_c: fn(C) -> f64, tol: f64)
where
    S: CudaDType + CudaMatMulMode + Copy + Clone + Send + Sync + 'static + num::Zero,
    C: CudaDType + CudaMatMulMode + Copy + Clone + Send + Sync + 'static + num::Zero + num::One,
{
    let (m, k, n, p) = (12usize, 32usize, 8usize, 3usize);
    let rounded = |seed: u64, cnt: usize| -> Vec<f64> { data(cnt, seed).into_iter().map(to_s).map(of_s).collect() };
    let stored = |seed: u64, cnt: usize| -> Vec<S> { data(cnt, seed).into_iter().map(to_s).collect() };

    let mut drivers = vec![MatMulDrivers::CUBLAS_LT, MatMulDrivers::CUBLAS_EX];
    if cublas_classic {
        drivers.push(MatMulDrivers::CUBLAS);
    }
    drivers.extend([MatMulDrivers::NAIVE, MatMulDrivers::all()]);
    for order in [RowMajor, ColMajor] {
        let mut dcpu = DeviceCpuSerial::default();
        dcpu.set_default_order(order);
        let cpu = |v: Vec<f64>, s: Vec<usize>| -> Cpu<f64> { rt::asarray((v, &dcpu)).into_shape(s) };
        let cases = shape_cases!(cpu, |b: usize, d: [usize; 2]| sh(order, b, d), rounded, m, k, n, p);
        for forced in &drivers {
            let mut flags = CudaRuntimeFlags::default();
            flags.matmul_drivers = *forced;
            let mut d = DeviceCuda::new(0).with_flags(flags);
            d.set_default_order(order);
            let on_device = shape_cases!(
                |v: Vec<S>, s: Vec<usize>| rt::asarray((v, &d)).into_shape(s),
                |b: usize, dims: [usize; 2]| sh(order, b, dims),
                stored,
                m,
                k,
                n,
                p
            );
            for ((shape, a, b), (_, ag, bg)) in cases.iter().zip(&on_device) {
                let want = rt::matmul(a, b);
                let c_shape = want.shape().to_vec();
                let want = DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap();
                let mut c: Tensor<C, DeviceCuda, IxD> = rt::full((c_shape, C::one(), &d));
                ag.matmul_with_output::<S, C, IxD, IxD>(bg, &mut c.view_mut());
                let got: Vec<f64> = DeviceToCpuAPI::<C>::to_cpu_vec(c.to_contig(RowMajor).storage())
                    .rstsr_unwrap()
                    .into_iter()
                    .map(of_c)
                    .collect();
                let err = rel_err(&got, &want);
                assert!(err <= tol, "{what} {shape}, {order:?}, {forced:?}: max rel err {err:.3e} > {tol:.1e}");
            }
        }
    }
    eprintln!("  [{what}] 6 shapes x 2 orders x {} driver settings within {tol:.1e}", drivers.len());
}

/// The layout matrix over the type triples: tolerance per `(triple, mode)` here means the output
/// rounding (all in the `Default` mode), the same numbers as `modes_half` and `modes_complex`.
#[test]
fn triple_layout_matrix() {
    require_cuda!();
    let re = |x: f64| Complex::new(x, 0.0);
    let re_of = |z: Complex<f64>| z.re;
    matrix::<f32, f32>("f32 -> f32", true, |x| x as f32, f64::from, f64::from, 1e-5);
    matrix::<f64, f64>("f64 -> f64", true, |x| x, |x| x, |x| x, 1e-13);
    matrix::<Complex<f32>, Complex<f32>>(
        "c32 -> c32",
        true,
        |x: f64| Complex::new(x as f32, 0.0),
        |z| z.re as f64,
        |z| z.re as f64,
        1e-5,
    );
    matrix::<Complex<f64>, Complex<f64>>("c64 -> c64", true, re, re_of, re_of, 1e-13);
    matrix::<f16, f16>("f16 -> f16", false, f16::from_f64, f64::from, f64::from, 1e-2);
    matrix::<f16, f32>("f16 -> f32", false, f16::from_f64, f64::from, f64::from, 1e-5);
    matrix::<bf16, bf16>("bf16 -> bf16", false, bf16::from_f64, f64::from, f64::from, 1e-2);
    matrix::<bf16, f32>("bf16 -> f32", false, bf16::from_f64, f64::from, f64::from, 1e-5);
}
