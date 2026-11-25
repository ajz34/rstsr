//! The naive matmul driver and the driver chain, on hardware.
//!
//! `tests/test_kernels.rs` covers the plain shapes through the default chain (cuBLAS first) and
//! `tests/test_matmul_tiled.rs` the tiled kernel alone. This file covers what is new in the chain:
//!
//! - the layouts the accelerated drivers decline (stepped operand, non-affine batch) now give the
//!   right answer through the default chain, and the same answer with naive forced;
//! - the type triples that only naive serves: integers, mixed operands, `f16 -> f32`;
//! - the enable set: a forced driver that cannot serve is an error naming it, never a wrong answer.
//!
//! Every operand is integer-valued, so products and sums are exact in every dtype used and the
//! comparison is `assert_eq!` (half excepted: `f16` storage rounds the inputs, so the `f16 -> f32`
//! case compares against the CPU product of the *rounded* inputs, still exact).

use half::f16;
use num::Complex;
use rstsr::prelude::*;
use rstsr_cuda::dtype::CudaDType;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

/// Integer-valued data in `[-4, 4]`, exactly representable in every dtype used here.
fn exact_data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 9) as f64 - 4.0
        })
        .collect()
}

fn forced(drivers: MatMulDrivers) -> DeviceCuda {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = drivers;
    DeviceCuda::new(0).with_flags(flags)
}

macro_rules! require_cuda {
    () => {
        if !std::panic::catch_unwind(funcs::is_available).unwrap_or(false) {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

/// Row-major host values of a tensor of any rank.
fn host_nd<'a, Tr, T, B, D>(t: &TensorAny<Tr, T, B, D>) -> Vec<T>
where
    T: Clone + Send + Sync + 'static,
    Tr: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToCpuAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<T>::to_cpu_vec(contig.storage()).rstsr_unwrap()
}

/// `a @ b` on one CUDA device against the CPU device, exactly.
fn check<'a, T, Ra, Rb>(
    d: &DeviceCuda,
    what: &str,
    a: &'a TensorAny<Ra, T, DeviceCpuSerial, IxD>,
    b: &'a TensorAny<Rb, T, DeviceCpuSerial, IxD>,
) where
    T: CudaDType
        + CudaMatMulMode
        + Copy
        + Clone
        + Send
        + Sync
        + 'static
        + PartialEq
        + core::fmt::Debug
        + core::ops::Mul<Output = T>
        + num::Zero
        + num::One,
    DeviceCpuSerial: DeviceRawAPI<T, Raw = Vec<T>>,
    Ra: DataAPI<Data = <DeviceCpuSerial as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    Rb: DataAPI<Data = <DeviceCpuSerial as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    DeviceCpuSerial: DeviceChangeAPI<'a, DeviceCuda, Ra, T, IxD> + DeviceChangeAPI<'a, DeviceCuda, Rb, T, IxD>,
{
    let want = rt::matmul(a, b);
    let (ag, bg) = (a.to_device(d), b.to_device(d));
    let got = rt::matmul(&ag, &bg);
    assert_eq!(host_nd(&got), host_nd(&want), "{what}");
}

/// The error of `a @ b` (which must fail).
fn matmul_err<'a, T, Ra, Rb>(a: &TensorAny<Ra, T, DeviceCuda, IxD>, b: &TensorAny<Rb, T, DeviceCuda, IxD>) -> String
where
    T: CudaDType
        + CudaMatMulMode
        + Copy
        + Clone
        + Send
        + Sync
        + 'static
        + PartialEq
        + core::ops::Mul<Output = T>
        + num::Zero
        + num::One,
    Ra: DataAPI<Data = <DeviceCuda as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    Rb: DataAPI<Data = <DeviceCuda as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    T: Copy,
{
    match rt::matmul_f(a, b) {
        Ok(_) => panic!("the matmul should have failed"),
        Err(e) => format!("{e:?}").chars().filter(|&c| c != '\n').collect(),
    }
}

/// `[m, k] @ [k, n]`, including shapes that are not multiples of any tile.
#[test]
fn naive_gemm22_vs_cpu() {
    require_cuda!();
    let d = forced(MatMulDrivers::NAIVE);
    let dcpu = DeviceCpuSerial::default();
    for &(m, k, n) in
        &[(32usize, 32usize, 32usize), (1, 1, 1), (1, 37, 1), (17, 33, 5), (130, 70, 129), (5, 1, 7), (7, 9, 1)]
    {
        let a: Tensor<f64, DeviceCpuSerial, _> =
            rt::asarray((exact_data(m * k, m as u64 + 1), &dcpu)).into_shape([m, k]);
        let b: Tensor<f64, DeviceCpuSerial, _> =
            rt::asarray((exact_data(k * n, n as u64 + 2), &dcpu)).into_shape([k, n]);
        check(&d, &format!("({m} x {k}) @ ({k} x {n})"), &a, &b);
    }
}

/// The layouts cuBLAS and the tiled kernel decline: a stepped operand (no unit-stride axis), a
/// reversed operand into a stepped output, and a non-affine two-axis batch.
#[test]
fn declined_layouts_run_on_naive() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (20usize, 16usize, 12usize);
    let mk = |n: usize, seed: u64, shape: &[usize]| -> Tensor<f64, DeviceCpuSerial, IxD> {
        rt::asarray((exact_data(n, seed), &dcpu)).into_shape(shape.to_vec())
    };
    let b = mk(k * n, 2, &[k, n]);
    let a = mk(m * k, 3, &[m, k]);
    let stepped0 = mk(4 * m * k, 4, &[2 * m, 2 * k]);
    let stepped = stepped0.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
    assert_eq!(stepped.stride(), &[4 * k as isize, 2]);
    let flipped = a.flip(1);
    let nonaffine0 = mk(2 * m * 3 * k, 5, &[2, m, 3, k]);
    let nonaffine = nonaffine0.transpose([2, 0, 1, 3]);
    assert_eq!(nonaffine.shape(), &[3, 2, m, k]);
    let b_np = mk(3 * 2 * k * n, 6, &[3, 2, k, n]);

    // through the default chain (cuBLAS and tiled decline, naive runs) ...
    let d = DeviceCuda::new(0);
    check(&d, "chain: stepped a", &stepped, &b);
    check(&d, "chain: a flipped on k", &flipped, &b);
    check(&d, "chain: non-affine batch", &nonaffine, &b_np);
    // ... and with naive forced; the answers agree bit for bit.
    let dn = forced(MatMulDrivers::NAIVE);
    check(&dn, "naive: stepped a", &stepped, &b);
    check(&dn, "naive: a flipped on k", &flipped, &b);
    check(&dn, "naive: non-affine batch", &nonaffine, &b_np);
}

/// A forced driver that cannot serve the layout is an error naming it, never a wrong answer.
#[test]
fn forced_driver_that_cannot_serve_is_an_error() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (20usize, 16usize, 12usize);
    let mk = |n: usize, seed: u64, shape: &[usize]| -> Tensor<f64, DeviceCpuSerial, IxD> {
        rt::asarray((exact_data(n, seed), &dcpu)).into_shape(shape.to_vec())
    };
    let b = mk(k * n, 2, &[k, n]);
    let stepped0 = mk(4 * m * k, 4, &[2 * m, 2 * k]);
    let stepped = stepped0.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
    let dt = forced(MatMulDrivers::TILED);
    let (a, b) = (stepped.to_device(&dt), b.to_device(&dt));
    let msg = matmul_err(&a, &b);
    eprintln!("  [tiled alone, stepped operand] {}", msg.chars().take(400).collect::<String>());
    assert!(msg.contains("tiled: cannot serve") && msg.contains("no unit-stride axis"), "{msg}");
    assert!(msg.contains("double x double -> double"), "the error names the type triple: {msg}");
    assert!(msg.contains("enabled MatMulDrivers(tiled)"), "{msg}");

    // an empty enable set names the set and the triple
    let de = forced(MatMulDrivers::none());
    let (a, b) = (stepped.to_device(&de), b.to_device(&de));
    let msg = matmul_err(&a, &b);
    assert!(msg.contains("MatMulDrivers(none)") && msg.contains("double x double -> double"), "{msg}");
}

/// Integer and mixed triples, served by the naive driver only.
#[test]
fn integer_and_mixed_triples() {
    require_cuda!();
    let d = forced(MatMulDrivers::NAIVE);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (12usize, 16usize, 8usize);
    let ai: Tensor<i32, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 1).iter().map(|&x| x as i32).collect::<Vec<_>>(), &dcpu)).into_shape([m, k]);
    let bi: Tensor<i32, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 2).iter().map(|&x| x as i32).collect::<Vec<_>>(), &dcpu)).into_shape([k, n]);
    check(&d, "i32 @ i32", &ai, &bi);
    // `u8` with a contraction short enough that no sum leaves the type: the CPU reference panics
    // on integer overflow where the device (like C and NumPy) wraps.
    let (m4, k4, n4) = (12usize, 4usize, 8usize);
    let bytes =
        |seed: u64, cnt: usize| -> Vec<u8> { exact_data(cnt, seed).into_iter().map(|x| (x as i8 + 4) as u8).collect() };
    let au: Tensor<u8, DeviceCpuSerial, _> = rt::asarray((bytes(3, m4 * k4), &dcpu)).into_shape([m4, k4]);
    let bu: Tensor<u8, DeviceCpuSerial, _> = rt::asarray((bytes(4, k4 * n4), &dcpu)).into_shape([k4, n4]);
    check(&d, "u8 @ u8", &au, &bu);

    // mixed operands: `f32 x f64 -> f64` through `matmul_from` (no `Mul` bound). The CPU serial
    // device has no mixed entry, so the reference is the f32 product of the f32 operands.
    let a32: Tensor<f32, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 5).iter().map(|&x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([m, k]);
    let b32: Tensor<f32, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 6).iter().map(|&x| x as f32).collect::<Vec<_>>(), &dcpu)).into_shape([k, n]);
    // the same integer-valued data as f64 (no cast API needed)
    let b64: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(k * n, 6), &dcpu)).into_shape([k, n]);
    let mut got: Tensor<f64, _, _> = rt::full(([m * n], f64::NAN, &d)).into_shape([m, n]);
    got.matmul_from(&a32.to_device(&d), &b64.to_device(&d), 1.0, 0.0);
    let want = rt::matmul(&a32, &b32);
    let got32: Vec<f32> = host_nd(&got).into_iter().map(|x| x as f32).collect();
    assert_eq!(got32, host_nd(&want), "f32 @ f64 -> f64");
    assert!(!host_nd(&got).contains(&f64::NAN));
}

/// Complex `c32` and `c64`, served by the naive driver only: batched, transposed operand, and a
/// stepped operand. Integer- or half-integer-valued in exact binary fractions, so exact.
#[test]
fn complex_triples() {
    require_cuda!();
    let d = forced(MatMulDrivers::NAIVE);
    let dcpu = DeviceCpuSerial::default();
    let (p, m, k, n) = (3usize, 12usize, 16usize, 8usize);
    let data = |seed: usize, cnt: usize| -> Vec<Complex<f64>> {
        (0..cnt)
            .map(|i| Complex::new((((i * 7 + seed) % 9) as f64 - 4.0) / 2.0, ((i + seed) % 5) as f64 - 2.0))
            .collect()
    };
    let mk = |seed: usize, shape: &[usize]| -> Tensor<Complex<f64>, DeviceCpuSerial, IxD> {
        let cnt = shape.iter().product();
        rt::asarray((data(seed, cnt), &dcpu)).into_shape(shape.to_vec())
    };
    let a = mk(1, &[p, m, k]);
    let b = mk(2, &[p, k, n]);
    let a2 = mk(1, &[m, k]);
    let base = mk(5, &[n, k]);
    let bt = base.t();
    let stepped0 = mk(6, &[2 * m, 2 * k]);
    let stepped = stepped0.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
    check(&d, "c64 batched", &a, &b);
    check(&d, "c64 @ c64.t()", &a2, &bt);
    check(&d, "c64 stepped operand", &stepped, &bt);

    let a32: Tensor<Complex<f32>, DeviceCpuSerial, _> =
        rt::asarray((data(3, m * k).iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect::<Vec<_>>(), &dcpu))
            .into_shape([m, k]);
    let b32: Tensor<Complex<f32>, DeviceCpuSerial, _> =
        rt::asarray((data(4, k * n).iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect::<Vec<_>>(), &dcpu))
            .into_shape([k, n]);
    check(&d, "c32", &a32, &b32);
}

/// `f16 x f16 -> f32` through `matmul_from` (no `Mul` bound): the inputs are rounded to `f16`
/// storage on both devices, so the CPU product of the rounded inputs is the exact expectation.
#[test]
fn f16_inputs_into_f32_output() {
    require_cuda!();
    let d = forced(MatMulDrivers::NAIVE);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (12usize, 64usize, 8usize);
    let round = |seed: u64, cnt: usize| -> Vec<f16> { exact_data(cnt, seed).into_iter().map(f16::from_f64).collect() };
    let a_cpu: Tensor<f16, DeviceCpuSerial, _> = rt::asarray((round(1, m * k), &dcpu)).into_shape([m, k]);
    let b_cpu: Tensor<f16, DeviceCpuSerial, _> = rt::asarray((round(2, k * n), &dcpu)).into_shape([k, n]);
    let (a, b) = (a_cpu.to_device(&d), b_cpu.to_device(&d));

    let mut c: Tensor<f32, _, _> = rt::full(([m * n], f32::NAN, &d)).into_shape([m, n]);
    c.matmul_from(&a, &b, 1.0, 0.0);
    // the CPU reference multiplies the rounded inputs as f32 (each `f16` is an exact `f32`)
    let up = |seed: u64, cnt: usize| -> Vec<f32> { round(seed, cnt).into_iter().map(f16::to_f32).collect() };
    let ar: Tensor<f32, DeviceCpuSerial, _> = rt::asarray((up(1, m * k), &dcpu)).into_shape([m, k]);
    let br: Tensor<f32, DeviceCpuSerial, _> = rt::asarray((up(2, k * n), &dcpu)).into_shape([k, n]);
    let want = rt::matmul(&ar, &br);
    assert_eq!(host_nd(&c), host_nd(&want), "f16 @ f16 -> f32");
    // NaN survivors would show a slot that was not written
    assert!(!host_nd(&c).contains(&f32::NAN));
}

/// `matmul_from` reads `c` when `beta != 0` and never reads it when `beta == 0`: a stepped or
/// reversed output view on the naive driver.
///
/// The CPU device refuses a strided `c`, so the reference is assembled by hand from the CPU
/// product and the base buffer.
#[test]
fn naive_matmul_from_strided_output() {
    require_cuda!();
    let d = forced(MatMulDrivers::NAIVE);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (20usize, 16usize, 12usize);
    let a: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * k, 7), &dcpu)).into_shape([m, k]);
    let b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(k * n, 8), &dcpu)).into_shape([k, n]);
    // The device buffer is the CPU base buffer `c0` (so `beta != 0` has something to read); the
    // reference is `c0` with `alpha * a @ b + beta * c0` written into the selected slots.
    let c0: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * 2 * n, 9), &dcpu)).into_shape([m, 2 * n]);
    let prod = rt::matmul(&a, &b);
    let (pv, cv) = (host_nd(&prod), host_nd(&c0));
    // index of element `i` of an axis of extent `d` sliced by `slice!(None, None, step)`
    let at = |i: usize, d: usize, step: isize| -> usize {
        if step > 0 {
            (step as usize) * i
        } else {
            d - 1 - ((step.unsigned_abs()) * i)
        }
    };
    macro_rules! case {
        ($what:expr, $rstep:expr, $cstep:expr, $alpha:expr, $beta:expr) => {{
            let mut want = cv.clone();
            for i in 0..m {
                for j in 0..n {
                    let (ri, cj) = (at(i, m, $rstep), at(j, 2 * n, $cstep));
                    want[ri * 2 * n + cj] = $alpha * pv[i * n + j] + $beta * cv[ri * 2 * n + cj];
                }
            }
            let mut got = c0.to_device(&d);
            let (ra, rb) = (a.to_device(&d), b.to_device(&d));
            let mut view = got.slice_mut(s![slice!(None, None, $rstep), slice!(None, None, $cstep)]);
            assert_eq!(view.shape(), &[m, n].into_iter().collect::<Vec<_>>(), "{}: fixture shape", $what);
            view.matmul_from(&ra, &rb, $alpha, $beta);
            let hv = host_nd(&got);
            // every slot: written ones against the reference, untouched ones against the base
            assert_eq!(hv.len(), cv.len());
            for i in 0..m {
                for j in 0..n {
                    let (ri, cj) = (at(i, m, $rstep), at(j, 2 * n, $cstep));
                    assert_eq!(hv[ri * 2 * n + cj], want[ri * 2 * n + cj], "{}: c[{}, {}]", $what, ri, cj);
                }
            }
        }};
    }
    case!("stepped c, beta != 0", 1isize, 2isize, 2.0f64, 3.0f64);
    case!("flipped c, beta != 0", -1isize, 2isize, 1.0f64, -1.5f64);
    case!("flipped c, beta = 0", -1isize, -2isize, 0.5f64, 0.0f64);
}

/// Both default orders agree with the CPU device on the same batched problem.
///
/// Under col-major the two matrix axes come first, so the batch axis moves to the end of the
/// shape there; the tensors are the same logical objects and the results are compared in
/// row-major.
#[test]
fn naive_both_orders() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (p, m, k, n) = (3usize, 12usize, 16usize, 8usize);
    let a: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(p * m * k, 10), &dcpu)).into_shape([p, m, k]);
    let b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(p * k * n, 11), &dcpu)).into_shape([p, k, n]);
    // reference: row-major everywhere
    let want = rt::matmul(&a, &b);
    {
        let d = forced(MatMulDrivers::NAIVE);
        let got = rt::matmul(a.to_device(&d), b.to_device(&d));
        assert_eq!(host_nd(&got), host_nd(&want), "row-major");
    }
    let mut d = forced(MatMulDrivers::NAIVE);
    d.set_default_order(ColMajor);
    let (at, bt) = (a.transpose([1, 2, 0]), b.transpose([1, 2, 0]));
    let ac = at.to_contig(ColMajor);
    let bc = bt.to_contig(ColMajor);
    assert_eq!(ac.stride(), &[1, m as isize, (m * k) as isize], "fixture: col-major contiguous");
    let got = rt::matmul(ac.to_device(&d), bc.to_device(&d));
    assert_eq!(got.shape(), &[m, n, p]);
    assert_eq!(host_nd(&got), host_nd(&want.transpose([1, 2, 0])), "col-major");
}

/// `bf16` on the naive kernel: `bf16 -> bf16` and `bf16 -> f32`. The kernel's half helpers are
/// per dtype (`__bfloat162float`, not `__half2float`): with the `f16` names the kernel did not
/// compile, which only a `bf16` triple forced onto naive shows.
#[test]
fn bf16_triples_on_naive() {
    require_cuda!();
    let d = forced(MatMulDrivers::NAIVE);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (8usize, 16usize, 6usize);
    let round = |seed: u64, cnt: usize| -> Vec<half::bf16> { exact_data(cnt, seed).into_iter().map(half::bf16::from_f64).collect() };
    let a: Tensor<half::bf16, DeviceCpuSerial, _> = rt::asarray((round(1, m * k), &dcpu)).into_shape([m, k]);
    let b: Tensor<half::bf16, DeviceCpuSerial, _> = rt::asarray((round(2, k * n), &dcpu)).into_shape([k, n]);
    let (ag, bg) = (a.to_device(&d), b.to_device(&d));

    let mut c: Tensor<f32, _, _> = rt::full(([m * n], f32::NAN, &d)).into_shape([m, n]);
    c.matmul_from(&ag, &bg, 1.0, 0.0);
    let up = |t: &Tensor<half::bf16, DeviceCpuSerial, IxD>| -> Tensor<f32, DeviceCpuSerial, IxD> {
        rt::asarray((host_nd(t).into_iter().map(f32::from).collect::<Vec<_>>(), &dcpu)).into_shape(t.shape().to_vec())
    };
    let want = rt::matmul(up(&a), up(&b));
    assert_eq!(host_nd(&c), host_nd(&want), "bf16 @ bf16 -> f32");
    assert!(!host_nd(&c).contains(&f32::NAN));

    let got: Tensor<half::bf16, DeviceCuda, IxD> = rt::matmul(&ag, &bg);
    let got: Vec<f32> = host_nd(&got).into_iter().map(f32::from).collect();
    let want: Vec<f32> = host_nd(&want);
    // bf16 output: within one bf16 rounding of the f32 product
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 4e-3 * (1.0 + w.abs()), "bf16 @ bf16 -> bf16");
    }
}
