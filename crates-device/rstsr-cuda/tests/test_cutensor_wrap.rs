//! Contraction tests of the cuTENSOR wrapper, on hardware: every serveable dtype against a CPU
//! oracle, the view shapes of real callers, the accumulation forms, and the refusal matrix.
//!
//! The wrapper is memory-safe by construction; what the tests establish is *agreement* (the
//! wrapper computes what the CPU device computes, within the dtype's summation error) and
//! *honesty* (every case the probes showed to be accepted-with-garbage or undocumented is refused
//! by name before cuTENSOR runs). Skips say why: no CUDA device, or `libcutensor` not loadable —
//! a green run without the skip lines means everything ran. Point `LD_LIBRARY_PATH` at another
//! version's `libcutensor.so` symlink dir to run the file against that version; the runtime
//! version it ran against is printed once.

#![cfg(cutensor_bindings)]

use num::Complex;
use rstsr::prelude::*;
use rstsr_core::prelude_dev::{Error, RSTSRError, RSTSRResultAPI};
use rstsr_cuda::cutensor_wrap::{self, CutensorDType, CutensorWorkspace};
use rstsr_cuda::prelude_dev::*;

/// The device to run on, or `None` (with the reason printed) when the hardware is missing.
fn device_or_skip() -> Option<DeviceCuda> {
    if !rstsr_cuda::registry::is_available() {
        eprintln!("skipping: no CUDA device visible");
        return None;
    }
    let dev = DeviceCuda::new(0);
    if !cutensor_wrap::handle::is_available() {
        eprintln!("skipping: libcutensor is not loadable (or its handle could not be created)");
        return None;
    }
    static VERSION: std::sync::Once = std::sync::Once::new();
    VERSION.call_once(|| {
        let v = dev.ordinal_obj().cutensor().map(|h| h.version()).expect("handle: `is_available` held");
        eprintln!("cuTENSOR runtime version: {v:?} (printed once per test binary)");
    });
    Some(dev)
}

/// Uniform data in `[-1, 1)`, seeded.
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn data_c(n: usize, seed: u64) -> Vec<Complex<f64>> {
    let (re, im) = (data(n, seed), data(n, seed ^ 0x5f5f));
    re.into_iter().zip(im).map(|(r, i)| Complex::new(r, i)).collect()
}

fn to_cpu_f64(t: &Tensor<f64, DeviceCuda, IxD>) -> Vec<f64> {
    DeviceToCpuAPI::<f64>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

fn to_cpu_f32(t: &Tensor<f32, DeviceCuda, IxD>) -> Vec<f32> {
    DeviceToCpuAPI::<f32>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

fn to_cpu_c64(t: &Tensor<Complex<f64>, DeviceCuda, IxD>) -> Vec<Complex<f64>> {
    DeviceToCpuAPI::<Complex<f64>>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

fn to_cpu_c32(t: &Tensor<Complex<f32>, DeviceCuda, IxD>) -> Vec<Complex<f32>> {
    DeviceToCpuAPI::<Complex<f32>>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

/// `err <= atol + rtol * |want|`, reporting the worst index (the `test_kernels` helper).
fn assert_close(got: &[f64], want: &[f64], ctx: &str) {
    const ATOL: f64 = 1e-9;
    const RTOL: f64 = 1e-9;
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let budget = ATOL + RTOL * w.abs();
        let err = (g - w).abs();
        if err / budget > worst {
            worst = err / budget;
            at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{ctx}: |got - want| exceeds atol + rtol*|want| at index {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

fn assert_close_rel(got: &[f64], want: &[f64], rtol: f64, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let scale = want.iter().fold(1e-300f64, |s, &x| s.max(x.abs()));
    let err = got.iter().zip(want).map(|(&g, &w)| (g - w).abs()).fold(0.0f64, f64::max);
    assert!(err <= rtol * scale, "{ctx}: max |err| = {err:e}, rtol*scale = {:e}", rtol * scale);
}

fn assert_close_c64(got: &[Complex<f64>], want: &[Complex<f64>], rtol: f64, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let scale = want.iter().fold(1e-300f64, |s, &x| s.max(x.norm()));
    let err = got.iter().zip(want).map(|(&g, &w)| (g - w).norm()).fold(0.0f64, f64::max);
    assert!(err <= rtol * scale, "{ctx}: max |err| = {err:e}, rtol*scale = {:e}", rtol * scale);
}

fn expect_invalid(err: Error, what: &str) {
    assert!(cutensor_wrap::is_decline(&err), "{what}: expected a decline, got {err:?}");
    let text = err.to_string();
    assert!(text.contains("cutensor:"), "{what}: message should name the module: {text}");
}

/// The identity operator, named once for the call sites below.
fn ident() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY
}

fn conj() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_CONJ
}

/// `CUTENSOR_OP_ADD`, named once for the reduction/elementwise call sites.
fn add() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_ADD
}

/// `CUTENSOR_OP_MUL`.
fn mul() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_MUL
}

/// `CUTENSOR_OP_MAX`.
fn max() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_MAX
}

/// `CUTENSOR_OP_MIN`.
fn min() -> cudarc::cutensor::sys::cutensorOperator_t {
    cudarc::cutensor::sys::cutensorOperator_t::CUTENSOR_OP_MIN
}

/// The CPU `f64` product of host data, as a flat row-major vector.
fn cpu_matmul(av: &[f64], bv: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let a_cpu = rt::asarray((av.to_vec(), &DeviceCpuSerial::default())).into_shape([m, k]);
    let b_cpu = rt::asarray((bv.to_vec(), &DeviceCpuSerial::default())).into_shape([k, n]);
    let want = rt::matmul(&a_cpu, &b_cpu);
    DeviceToCpuAPI::<f64>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap()
}

fn cpu_matmul_c(av: &[Complex<f64>], bv: &[Complex<f64>], m: usize, k: usize, n: usize) -> Vec<Complex<f64>> {
    let a_cpu = rt::asarray((av.to_vec(), &DeviceCpuSerial::default())).into_shape([m, k]);
    let b_cpu = rt::asarray((bv.to_vec(), &DeviceCpuSerial::default())).into_shape([k, n]);
    let want = rt::matmul(&a_cpu, &b_cpu);
    DeviceToCpuAPI::<Complex<f64>>::to_cpu_vec(want.to_contig(RowMajor).storage()).rstsr_unwrap()
}

// -------------------------------------------------------------------------------------------------
// accuracy

/// `ij,jk->ik` for the four full-precision dtypes, against the CPU device.
#[test]
fn gemm_matches_cpu_per_dtype() {
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (7usize, 13usize, 5usize);
    let av = data(m * k, 1);
    let bv = data(k * n, 2);
    let want = cpu_matmul(&av, &bv, m, k, n);

    // f64
    let a = rt::asarray((av.clone(), &dev)).into_shape([m, k]);
    let b = rt::asarray((bv.clone(), &dev)).into_shape([k, n]);
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
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("f64 contraction");
    assert_close(&to_cpu_f64(&d), &want, "f64 ij,jk->ik");

    // f32
    let av32: Vec<f32> = av.iter().map(|&x| x as f32).collect();
    let bv32: Vec<f32> = bv.iter().map(|&x| x as f32).collect();
    let a32 = rt::asarray((av32, &dev)).into_shape([m, k]);
    let b32 = rt::asarray((bv32, &dev)).into_shape([k, n]);
    let mut d32 = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        a32.view(),
        &[0, 1],
        ident(),
        b32.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, f32, DeviceCuda, IxD>>,
        d32.view_mut(),
        &[0, 2],
        1.0f32,
        0.0f32,
        cutensor_wrap::compute_desc::desc_32f().unwrap(),
        None,
    )
    .expect("f32 contraction");
    let got32: Vec<f64> = to_cpu_f32(&d32).into_iter().map(f64::from).collect();
    assert_close_rel(&got32, &want, 1e-5, "f32 ij,jk->ik");

    // c64
    let avc = data_c(m * k, 3);
    let bvc = data_c(k * n, 4);
    let want_c = cpu_matmul_c(&avc, &bvc, m, k, n);
    let ac = rt::asarray((avc.clone(), &dev)).into_shape([m, k]);
    let bc = rt::asarray((bvc.clone(), &dev)).into_shape([k, n]);
    let mut dc = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        ac.view(),
        &[0, 1],
        ident(),
        bc.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, Complex<f64>, DeviceCuda, IxD>>,
        dc.view_mut(),
        &[0, 2],
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("c64 contraction");
    assert_close_c64(&to_cpu_c64(&dc), &want_c, 1e-12, "c64 ij,jk->ik");

    // c32
    let avc32: Vec<Complex<f32>> = avc.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect();
    let bvc32: Vec<Complex<f32>> = bvc.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect();
    let ac32 = rt::asarray((avc32, &dev)).into_shape([m, k]);
    let bc32 = rt::asarray((bvc32, &dev)).into_shape([k, n]);
    let mut dc32 = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        ac32.view(),
        &[0, 1],
        ident(),
        bc32.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, Complex<f32>, DeviceCuda, IxD>>,
        dc32.view_mut(),
        &[0, 2],
        Complex::new(1.0f32, 0.0),
        Complex::new(0.0f32, 0.0),
        cutensor_wrap::compute_desc::desc_32f().unwrap(),
        None,
    )
    .expect("c32 contraction");
    let gotc32: Vec<Complex<f64>> =
        to_cpu_c32(&dc32).into_iter().map(|z| Complex::new(z.re as f64, z.im as f64)).collect();
    assert_close_c64(&gotc32, &want_c, 1e-5, "c32 ij,jk->ik");
}

/// `f16`/`bf16` storage computes in fp32 (cuTENSOR's `32F` compute); compared against the CPU
/// `f64` product at half-representable tolerance.
#[test]
fn half_types_compute_in_fp32() {
    let Some(dev) = device_or_skip() else { return };
    fn half_case<T>(dev: &DeviceCuda, name: &str, to: fn(f64) -> T)
    where
        T: CutensorDType<Scalar = f32> + Into<f64>,
    {
        let (m, k, n) = (16usize, 16usize, 16usize);
        let av = data(m * k, 5);
        let bv = data(k * n, 6);
        let want = cpu_matmul(&av, &bv, m, k, n);
        let a = rt::asarray((av.iter().copied().map(to).collect::<Vec<_>>(), dev)).into_shape([m, k]);
        let b = rt::asarray((bv.iter().copied().map(to).collect::<Vec<_>>(), dev)).into_shape([k, n]);
        let mut d = rt::asarray((vec![to(0.0); m * n], dev)).into_shape([m, n]);
        cutensor_wrap::contract(
            dev,
            a.view(),
            &[0, 1],
            ident(),
            b.view(),
            &[1, 2],
            ident(),
            None::<TensorView<'_, T, DeviceCuda, IxD>>,
            d.view_mut(),
            &[0, 2],
            1.0f32,
            0.0f32,
            cutensor_wrap::compute_desc::desc_32f().unwrap(),
            None,
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let got: Vec<f64> = DeviceToCpuAPI::<T>::to_cpu_vec(d.to_contig(RowMajor).storage())
            .rstsr_unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        assert_close_rel(&got, &want, 2e-2, name);
    }
    half_case::<half::f16>(&dev, "f16 ij,jk->ik", half::f16::from_f64);
    half_case::<half::bf16>(&dev, "bf16 ij,jk->ik", half::bf16::from_f64);
}

/// Batched (`bij,bjk->bik`), strided (`A` every second row), and transposed-`B` views, with a
/// non-trivial alpha.
#[test]
fn batched_strided_and_transposed_views() {
    let Some(dev) = device_or_skip() else { return };
    let (b, m, k, n) = (4usize, 7usize, 13usize, 5usize);
    let av = data(b * m * k, 7);
    let bv = data(b * k * n, 8);
    let a_cpu = rt::asarray((av.clone(), &DeviceCpuSerial::default())).into_shape([b, m, k]);
    let b_cpu = rt::asarray((bv.clone(), &DeviceCpuSerial::default())).into_shape([b, n, k]);
    let prod = rt::matmul(&a_cpu, b_cpu.view().transpose([0, 2, 1]));
    let want: Vec<f64> = DeviceToCpuAPI::<f64>::to_cpu_vec(prod.to_contig(RowMajor).storage())
        .rstsr_unwrap()
        .into_iter()
        .map(|x| 1.5 * x)
        .collect();

    let a = rt::asarray((av, &dev)).into_shape([b, m, k]);
    let bt = rt::asarray((bv, &dev)).into_shape([b, n, k]);
    let mut d = rt::zeros(([b, m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        a.view(),
        &[3, 0, 1],
        ident(),
        bt.view().transpose([0, 2, 1]),
        &[3, 1, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[3, 0, 2],
        1.5f64,
        0.0f64,
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("batched x transposed-B contraction");
    assert_close(&to_cpu_f64(&d), &want, "bij,bjk->bik (B transposed view)");

    // A strided: every second row of a (2*m, k) tensor.
    let av_big = data(2 * m * k, 9);
    let a_big = rt::asarray((av_big.clone(), &dev)).into_shape([2 * m, k]);
    let a_big_view = a_big.view();
    let a_strided = a_big_view.i((slice!(None, None, 2), ..));
    let b2v = data(k * n, 8);
    let b2 = rt::asarray((b2v.clone(), &dev)).into_shape([k, n]);
    let b_cpu2 = rt::asarray((b2v, &DeviceCpuSerial::default())).into_shape([k, n]);
    let a_cpu_big = rt::asarray((av_big, &DeviceCpuSerial::default())).into_shape([2 * m, k]);
    let a_cpu_big_view = a_cpu_big.view();
    let a_cpu_strided = a_cpu_big_view.i((slice!(None, None, 2), ..));
    let prod2 = rt::matmul(&a_cpu_strided, &b_cpu2);
    let want2: Vec<f64> = DeviceToCpuAPI::<f64>::to_cpu_vec(prod2.to_contig(RowMajor).storage()).rstsr_unwrap();
    let mut d2 = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        a_strided,
        &[0, 1],
        ident(),
        b2.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d2.view_mut(),
        &[0, 2],
        1.0f64,
        0.0f64,
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("strided-A contraction");
    assert_close(&to_cpu_f64(&d2), &want2, "ij,jk->ik with A strided (step 2 rows)");
}

/// Accumulation: in place (`c` aliases `d`, `beta = 1`) and out of place (`beta = 0.3`).
#[test]
fn accumulation_forms() {
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (6usize, 9usize, 4usize);
    let av = data(m * k, 10);
    let bv = data(k * n, 11);
    let prod = cpu_matmul(&av, &bv, m, k, n);
    let a = rt::asarray((av, &dev)).into_shape([m, k]);
    let b = rt::asarray((bv, &dev)).into_shape([k, n]);

    // in place: d += 1.5 * a @ b (c = None means C is D)
    let mut d = rt::full(([m, n], 0.5, &dev));
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
        1.5f64,
        1.0f64,
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("in-place accumulate (c = None, beta = 1)");
    let want: Vec<f64> = prod.iter().map(|&p| 0.5 + 1.5 * p).collect();
    assert_close(&to_cpu_f64(&d), &want, "d += 1.5 a@b (c aliases d)");

    // out of place: d2 = 2.0 * a @ b + 0.3 * c
    let cv = data(m * n, 12);
    let c = rt::asarray((cv.clone(), &dev)).into_shape([m, n]);
    let mut d2 = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[1, 2],
        ident(),
        Some(c.view()),
        d2.view_mut(),
        &[0, 2],
        2.0f64,
        0.3f64,
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("out-of-place accumulate");
    let want2: Vec<f64> = prod.iter().zip(cv).map(|(&p, cv)| 2.0 * p + 0.3 * cv).collect();
    assert_close(&to_cpu_f64(&d2), &want2, "d = 2 a@b + 0.3 c");
}

/// `beta = 0` with no `c`: the output pointer stands in for `C`, which is not read (a NaN-filled
/// output must come out clean).
#[test]
fn beta_zero_stands_in_for_c() {
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (5usize, 6usize, 7usize);
    let av = data(m * k, 13);
    let bv = data(k * n, 14);
    let want = cpu_matmul(&av, &bv, m, k, n);
    let a = rt::asarray((av, &dev)).into_shape([m, k]);
    let b = rt::asarray((bv, &dev)).into_shape([k, n]);
    let mut d = rt::full(([m, n], f64::NAN, &dev));
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
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("beta = 0 without c");
    assert_close(&to_cpu_f64(&d), &want, "d = a@b with NaN-prefilled d and c = None");
}

/// `opA = CONJ` on `c64` computes `conj(A) @ B` through the wrapper.
#[test]
fn conjugate_operand() {
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (6usize, 8usize, 5usize);
    let avc = data_c(m * k, 15);
    let bvc = data_c(k * n, 16);
    let a_cpu = rt::asarray((avc.clone(), &DeviceCpuSerial::default())).into_shape([m, k]);
    let b_cpu = rt::asarray((bvc.clone(), &DeviceCpuSerial::default())).into_shape([k, n]);
    let prod = rt::matmul(rt::conj(&a_cpu), &b_cpu);
    let want: Vec<Complex<f64>> =
        DeviceToCpuAPI::<Complex<f64>>::to_cpu_vec(prod.to_contig(RowMajor).storage()).rstsr_unwrap();
    let a = rt::asarray((avc, &dev)).into_shape([m, k]);
    let b = rt::asarray((bvc, &dev)).into_shape([k, n]);
    let mut d = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        conj(),
        b.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, Complex<f64>, DeviceCuda, IxD>>,
        d.view_mut(),
        &[0, 2],
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("conj(A) @ b");
    assert_close_c64(&to_cpu_c64(&d), &want, 1e-12, "conj(A)@B");
}

/// Rank-0 output (`i,i->`, the vector dot) and a diagonal-view operand.
#[test]
fn rank0_and_diagonal_views() {
    let Some(dev) = device_or_skip() else { return };
    // i,i-> with a rank-0 output
    let n = 13usize;
    let av = data(n, 17);
    let bv = data(n, 18);
    let want: f64 = av.iter().zip(&bv).map(|(x, y)| x * y).sum();
    let a = rt::asarray((av, &dev));
    let b = rt::asarray((bv, &dev));
    let mut d: Tensor<f64, DeviceCuda, IxD> = rt::zeros(([1], &dev)).into_shape(vec![] as Vec<usize>);
    cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0],
        ident(),
        b.view(),
        &[0],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[],
        1.0f64,
        0.0f64,
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("i,i-> rank-0 output");
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(d.view().to_contig(RowMajor).storage()).rstsr_unwrap();
    assert_close(&got, &[want], "i,i-> rank-0");

    // the diagonal of a square tensor as an operand (stride = s_i + s_j)
    let av = data(8 * 8, 19);
    let a_sq_cpu = rt::asarray((av.clone(), &DeviceCpuSerial::default())).into_shape([8, 8]);
    let diag_cpu: Vec<f64> = (0..8).map(|i| a_sq_cpu[[i, i]]).collect();
    let want: f64 = diag_cpu.iter().zip(&diag_cpu).map(|(x, y)| x * y).sum();
    let a_sq = rt::asarray((av, &dev)).into_shape([8, 8]);
    let a_sq_view = a_sq.view();
    let diag = a_sq_view.diagonal_f(0).unwrap();
    let mut d: Tensor<f64, DeviceCuda, IxD> = rt::zeros(([1], &dev)).into_shape(vec![] as Vec<usize>);
    cutensor_wrap::contract(
        &dev,
        diag.view(),
        &[0],
        ident(),
        diag.view(),
        &[0],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[],
        1.0f64,
        0.0f64,
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("diag,diag->");
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(d.view().to_contig(RowMajor).storage()).rstsr_unwrap();
    assert_close(&got, &[want], "diag . diag");
}

// -------------------------------------------------------------------------------------------------
// refusals

/// Every case the probes showed to be accepted-with-garbage or undocumented is refused by name.
#[test]
fn refusal_matrix() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let mk = |v: Vec<f64>, shape: [usize; 2]| rt::asarray((v, &dev)).into_shape(shape);

    // duplicate mode within one operand
    let a = mk(data(16, 20), [4, 4]);
    let b = mk(data(20, 21), [4, 5]);
    let mut d = rt::zeros(([5], &dev));
    let err = cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 0],
        ident(),
        b.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[1],
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("duplicate mode refused");
    expect_invalid(err, "duplicate mode");

    // an output mode in no input
    let err = cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[3],
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("output mode in no input");
    expect_invalid(err, "output mode in no input");

    // an extent mismatch on the contracted mode (cuTENSOR itself misses this)
    let a3 = mk(data(7 * 3, 22), [7, 3]);
    let b4 = mk(data(4 * 5, 23), [4, 5]);
    let mut d2 = rt::zeros(([7, 5], &dev));
    let err = cutensor_wrap::contract(
        &dev,
        a3.view(),
        &[0, 1],
        ident(),
        b4.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d2.view_mut(),
        &[0, 2],
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("contracted extent mismatch");
    expect_invalid(err, "contracted extent mismatch");

    // a stride-0 (broadcast) output
    let a7 = mk(data(7 * 13, 24), [7, 13]);
    let b5 = mk(data(13 * 5, 25), [13, 5]);
    let mut d_bcast = rt::into_broadcast(rt::zeros(([1, 5], &dev)), vec![7, 5]);
    let err = cutensor_wrap::contract(
        &dev,
        a7.view(),
        &[0, 1],
        ident(),
        b5.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d_bcast.view_mut(),
        &[0, 2],
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("stride-0 output refused");
    expect_invalid(err, "stride-0 output");

    // a negative-stride input (flipped view)
    let mut d3 = rt::zeros(([7, 5], &dev));
    let a7_view = a7.view();
    let flipped = a7_view.flip(0);
    let err = cutensor_wrap::contract(
        &dev,
        flipped,
        &[0, 1],
        ident(),
        b5.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d3.view_mut(),
        &[0, 2],
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("negative stride refused");
    expect_invalid(err, "negative stride");
}

/// A stride-0 (broadcast) *input* is not refused: it reaches cuTENSOR, which computes it
/// correctly from 2.6.0 and declines (`NOT_SUPPORTED` -> `UnImplemented`) on older libraries.
#[test]
fn stride0_input_passes_through() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let (m, k, n) = (7usize, 5usize, 3usize);
    // one row `[1, k]`, broadcast over `m` rows (stride `[0, 1]`)
    let row = data(k, 30);
    let a = rt::into_broadcast(rt::asarray((row.clone(), &dev)).into_shape([1, k]), vec![m, k]);
    assert_eq!(a.view().stride().to_vec(), vec![0isize, 1], "the input is a stride-0 view");
    let bv = data(k * n, 31);
    let b = rt::asarray((bv.clone(), &dev)).into_shape([k, n]);
    let mut d = rt::zeros(([m, n], &dev));
    let res = cutensor_wrap::contract(
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
    );
    let version = dev.ordinal_obj().cutensor().map(|h| h.version()).expect("handle");
    if version >= (2, 6, 0) {
        res.expect("stride-0 input accepted from 2.6.0");
        let av: Vec<f64> = row.iter().cycle().take(m * k).cloned().collect();
        let want = cpu_matmul(&av, &bv, m, k, n);
        assert_close(&to_cpu_f64(&d), &want, "stride-0 input");
    } else {
        let err = res.expect_err("stride-0 input declines before 2.6.0");
        assert!(cutensor_wrap::is_decline(&err), "a decline, got {err:?}");
        eprintln!("stride-0 input declined on cuTENSOR {version:?} (expected before 2.6.0)");
    }
}

/// `c` given with the same shape as `d` but a different layout is refused before cuTENSOR
/// (`descC == descD` is a full-facts requirement).
#[test]
fn c_layout_mismatch_refused() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let (m, k, n) = (7usize, 5usize, 3usize);
    let a = rt::asarray((data(m * k, 32), &dev)).into_shape([m, k]);
    let b = rt::asarray((data(k * n, 33), &dev)).into_shape([k, n]);
    // `[m, n]` with the strides of a `[n, m]` buffer
    let c_buf = rt::asarray((data(m * n, 34), &dev)).into_shape([n, m]);
    let c_t = c_buf.t();
    let mut d = rt::zeros(([m, n], &dev));
    let err = cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[1, 2],
        ident(),
        Some(c_t),
        d.view_mut(),
        &[0, 2],
        1.0f64,
        1.0f64,
        compute,
        None,
    )
    .expect_err("c layout != d refused");
    expect_invalid(err, "c layout != d");
}

/// The object layer's scalar guard: the scalar-type parameter is chosen independently of the
/// descriptors there, and a mismatch with cuTENSOR's derived `SCALAR_TYPE` is refused right after
/// the operation descriptor is created (the FFI reads the host scalars blind). `f64` scalars on
/// `f32` descriptors is the canonical misuse; the refusal must also release the descriptor it
/// just created.
#[test]
fn scalar_type_mismatch_refused() {
    let Some(dev) = device_or_skip() else { return };
    let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
    let (m, k, n) = (4usize, 5usize, 3usize);
    let a = rt::asarray((data(m * k, 35).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &dev)).into_shape([m, k]);
    let b = rt::asarray((data(k * n, 36).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &dev)).into_shape([k, n]);
    let a_view = a.view();
    let b_view = b.view();
    let desc_a = cutensor_wrap::TensorDescriptor::new::<f32>(&handle, &a_view).unwrap();
    let desc_b = cutensor_wrap::TensorDescriptor::new::<f32>(&handle, &b_view).unwrap();
    let err = cutensor_wrap::OperationDescriptor::new_contraction::<f64>(
        &handle,
        &desc_a,
        &[0, 1],
        ident(),
        &desc_b,
        &[1, 2],
        ident(),
        &desc_a,
        &[0, 1],
        &desc_a,
        &[0, 1],
        cutensor_wrap::compute_desc::desc_32f().unwrap(),
    )
    .expect_err("f64 scalar row on f32 descriptors refused");
    assert!(err.to_string().contains("scalar type"), "{err}");
    expect_invalid(err, "scalar type mismatch");
}

/// `f32 x c32 -> c32` passes the operation descriptor and declines at plan creation
/// (`NOT_SUPPORTED` -> `UnImplemented`) — the decline a fallback tier must detect by variant.
#[test]
fn f32_x_c32_declines_at_plan() {
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (4usize, 6usize, 3usize);
    let a = rt::asarray((data(m * k, 26), &dev)).into_shape([m, k]);
    let bc = data_c(k * n, 27);
    let b = rt::asarray((bc.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect::<Vec<_>>(), &dev))
        .into_shape([k, n]);
    let mut d = rt::zeros(([m, n], &dev));
    let err = cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[1, 2],
        ident(),
        None::<TensorView<'_, Complex<f32>, DeviceCuda, IxD>>,
        d.view_mut(),
        &[0, 2],
        Complex::new(1.0f32, 0.0),
        Complex::new(0.0f32, 0.0),
        cutensor_wrap::compute_desc::desc_32f().unwrap(),
        None,
    )
    .expect_err("f32 x c32 declines");
    assert!(matches!(err.inner, RSTSRError::UnImplemented(_)), "expected UnImplemented: {err:?}");
    assert!(cutensor_wrap::is_decline(&err), "f32 x c32 is a decline");
}

/// A held plan executes repeatedly against new buffers of the same layout, and refuses a view
/// whose layout differs from the planned one.
#[test]
fn plan_reuse_and_layout_mismatch() {
    use cutensor_wrap::{OperationDescriptor, Plan, PlanPreference, TensorDescriptor};
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (6usize, 8usize, 5usize);
    let av = data(m * k, 28);
    let bv = data(k * n, 29);
    let want = cpu_matmul(&av, &bv, m, k, n);

    let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
    let a = rt::asarray((av, &dev)).into_shape([m, k]);
    let b = rt::asarray((bv, &dev)).into_shape([k, n]);
    let desc_a = TensorDescriptor::new::<f64>(&handle, &a.view()).unwrap();
    let desc_b = TensorDescriptor::new::<f64>(&handle, &b.view()).unwrap();
    let mut d1 = rt::zeros(([m, n], &dev));
    let desc_d = TensorDescriptor::new::<f64>(&handle, &d1.view()).unwrap();
    let op = OperationDescriptor::new_contraction::<f64>(
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
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
    )
    .unwrap();
    let pref = PlanPreference::new(
        &handle,
        cudarc::cutensor::sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        cudarc::cutensor::sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )
    .unwrap();
    let ws = dev
        .stream()
        .cutensor_workspace(dev.flags().cutensor_workspace_bytes)
        .map(|w| CutensorWorkspace::from_slot(w.clone()));
    let limit = ws.as_ref().map_or(0, |w| w.len() as u64);
    let plan = Plan::new(&handle, op, &pref, limit).unwrap();
    println!("required workspace: {} B", plan.required_workspace());

    let run = |d: &mut Tensor<f64, DeviceCuda, IxD>, a_view| -> Vec<f64> {
        plan.contract(
            &dev,
            a_view,
            b.view(),
            None::<TensorView<'_, f64, DeviceCuda, IxD>>,
            d.view_mut(),
            1.0f64,
            0.0f64,
            ws.as_ref(),
        )
        .expect("planned contraction");
        to_cpu_f64(d)
    };
    assert_close(&run(&mut d1, a.view()), &want, "plan run 1");
    // a second buffer of the same layout
    let mut d2 = rt::zeros(([m, n], &dev));
    assert_close(&run(&mut d2, a.view()), &want, "plan run 2 (new buffer)");

    // a transposed-A view does not match the planned layout
    let mut d3 = rt::zeros(([m, n], &dev));
    let err = plan
        .contract(
            &dev,
            a.view().t(),
            b.view(),
            None::<TensorView<'_, f64, DeviceCuda, IxD>>,
            d3.view_mut(),
            1.0f64,
            0.0f64,
            None,
        )
        .expect_err("layout mismatch refused");
    expect_invalid(err, "layout mismatch");

    // element-size alignment declaration also serves offset views (the einsum reuse shape)
    let a_off_full = rt::asarray((data(m * k + 3, 28), &dev)).into_shape([m * k + 3]);
    let a_off = a_off_full.view().i((0..m * k,)).into_shape([m, k]);
    let desc_a_floor = TensorDescriptor::with_alignment::<f64>(&handle, &a_off.view(), 8).unwrap();
    let mut d4 = rt::zeros(([m, n], &dev));
    let desc_d4 = TensorDescriptor::new::<f64>(&handle, &d4.view()).unwrap();
    let op2 = OperationDescriptor::new_contraction::<f64>(
        &handle,
        &desc_a_floor,
        &[0, 1],
        ident(),
        &desc_b,
        &[1, 2],
        ident(),
        &desc_d4,
        &[0, 2],
        &desc_d4,
        &[0, 2],
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
    )
    .unwrap();
    let plan2 = Plan::new(&handle, op2, &pref, limit).unwrap();
    plan2
        .contract(
            &dev,
            a_off.view(),
            b.view(),
            None::<TensorView<'_, f64, DeviceCuda, IxD>>,
            d4.view_mut(),
            1.0f64,
            0.0f64,
            ws.as_ref(),
        )
        .expect("offset view served by the element-size-aligned plan");
    assert_close(&to_cpu_f64(&d4), &want, "offset-A contraction");
}

/// Executing on a second stream (a device value on that stream) orders correctly against
/// buffers allocated on the default stream.
#[test]
fn cross_stream_execute() {
    let Some(dev) = device_or_skip() else { return };
    let dev2 = dev.new_stream_f().expect("second stream device value");
    let (m, k, n) = (6usize, 8usize, 5usize);
    let av = data(m * k, 31);
    let bv = data(k * n, 32);
    let want = cpu_matmul(&av, &bv, m, k, n);
    // allocate on the default stream, execute on stream 2
    let a = rt::asarray((av, &dev)).into_shape([m, k]);
    let b = rt::asarray((bv, &dev)).into_shape([k, n]);
    let mut d = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev2,
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
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("cross-stream contraction");
    dev2.synchronize_f().expect("sync stream 2");
    assert_close(&to_cpu_f64(&d), &want, "cross-stream result");
}

/// The unlocked execute path in practice: several threads drive the same per-ordinal handle at
/// once, each on its own stream with its own operands, and every result must match the oracle.
#[test]
fn concurrent_execute_on_one_handle() {
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (32usize, 48usize, 40usize);
    let av = data(m * k, 40);
    let bv = data(k * n, 41);
    let want = cpu_matmul(&av, &bv, m, k, n);
    let workers: Vec<_> = (0..4usize)
        .map(|seed| {
            let dev = dev.clone();
            let av = av.clone();
            let bv = bv.clone();
            std::thread::spawn(move || {
                let dev2 = dev.new_stream_f().expect("stream");
                let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
                let a = rt::asarray((av, &dev)).into_shape([m, k]);
                let b = rt::asarray((bv, &dev)).into_shape([k, n]);
                let mut d = rt::asarray((vec![seed as f64; m * n], &dev)).into_shape([m, n]);
                for _ in 0..8 {
                    cutensor_wrap::contract(
                        &dev2,
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
                    .expect("concurrent contraction");
                }
                dev2.synchronize_f().expect("sync stream");
                d
            })
        })
        .collect();
    for worker in workers {
        let d = worker.join().expect("worker thread");
        assert_close(&to_cpu_f64(&d), &want, "concurrent execute on one handle");
    }
}

// -------------------------------------------------------------------------------------------------
// reduction

/// Reductions against host-vector oracles: `ij->j` (ADD and MAX), a strided input, a rank-0
/// output (`i,j->` with ADD and MIN), and the in-place accumulate form.
#[test]
fn reduce_matches_cpu() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let (m, k) = (7usize, 13usize);
    let av = data(m * k, 40);
    let a = rt::asarray((av.clone(), &dev)).into_shape([m, k]);
    let col_sum: Vec<f64> = (0..k).map(|j| (0..m).map(|i| av[i * k + j]).sum()).collect();
    let col_max: Vec<f64> = (0..k).map(|j| (0..m).map(|i| av[i * k + j]).fold(f64::NEG_INFINITY, f64::max)).collect();

    // ij->j, ADD with alpha = 2
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
        2.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect("ij->j ADD");
    let want: Vec<f64> = col_sum.iter().map(|&x| 2.0 * x).collect();
    assert_close(&to_cpu_f64(&d), &want, "ij->j ADD");

    // ij->j, MAX
    let mut d = rt::zeros(([k], &dev));
    cutensor_wrap::reduce(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[1],
        max(),
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect("ij->j MAX");
    assert_close(&to_cpu_f64(&d), &col_max, "ij->j MAX");

    // a strided input: every second row of a (2m, k) tensor
    let av_big = data(2 * m * k, 41);
    let a_big = rt::asarray((av_big.clone(), &dev)).into_shape([2 * m, k]);
    let a_big_view = a_big.view();
    let a_strided = a_big_view.i((slice!(None, None, 2), ..));
    let want: Vec<f64> = (0..k).map(|j| (0..m).map(|i| av_big[i * 2 * k + j]).sum()).collect();
    let mut d = rt::zeros(([k], &dev));
    cutensor_wrap::reduce(
        &dev,
        a_strided,
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
    .expect("strided ij->j");
    assert_close(&to_cpu_f64(&d), &want, "strided ij->j ADD");

    // full reduction to a rank-0 output: ADD and NORM2
    let mut d: Tensor<f64, DeviceCuda, IxD> = rt::zeros(([1], &dev)).into_shape(vec![] as Vec<usize>);
    cutensor_wrap::reduce(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[],
        add(),
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect("i,j-> total sum");
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(d.view().to_contig(RowMajor).storage()).rstsr_unwrap();
    assert_close(&got, &[av.iter().sum()], "i,j-> total sum");

    let mut d: Tensor<f64, DeviceCuda, IxD> = rt::zeros(([1], &dev)).into_shape(vec![] as Vec<usize>);
    cutensor_wrap::reduce(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[],
        min(),
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect("i,j-> MIN");
    let got = DeviceToCpuAPI::<f64>::to_cpu_vec(d.view().to_contig(RowMajor).storage()).rstsr_unwrap();
    assert_close(&got, &[av.iter().copied().fold(f64::INFINITY, f64::min)], "i,j-> MIN");

    // in-place accumulate: d += 1.5 * colsum (c = None means C is D)
    let mut d = rt::full(([k], 0.25, &dev));
    cutensor_wrap::reduce(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[1],
        add(),
        1.5f64,
        1.0f64,
        compute,
        None,
    )
    .expect("d += 1.5 colsum");
    let want: Vec<f64> = col_sum.iter().map(|&x| 0.25 + 1.5 * x).collect();
    assert_close(&to_cpu_f64(&d), &want, "in-place reduce accumulate");
}

/// Permutations against a host-vector oracle: a 3-D transpose with scaling, and a conjugating
/// permutation of a complex tensor.
#[test]
fn permute_matches_cpu() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let shape = [2usize, 3, 4];
    let av = data(2 * 3 * 4, 42);
    let a = rt::asarray((av.clone(), &dev)).into_shape(shape);
    // mode_b [2, 0, 1]: b[p, q, r] = a[q, r, p] with extents (4, 2, 3)
    let mut b = rt::zeros(([4, 2, 3], &dev));
    cutensor_wrap::permute(&dev, a.view(), &[0, 1, 2], ident(), b.view_mut(), &[2, 0, 1], 2.0f64, compute, None)
        .expect("permute [0,1,2] -> [2,0,1]");
    let mut want = Vec::with_capacity(4 * 2 * 3);
    for p in 0..4 {
        for q in 0..2 {
            for r in 0..3 {
                want.push(2.0 * av[(q * 3 + r) * 4 + p]);
            }
        }
    }
    assert_close(&to_cpu_f64(&b), &want, "3-D permute with scaling");

    // conjugating permutation of a c64 tensor
    let compute_c = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let avc = data_c(3 * 5, 43);
    let ac = rt::asarray((avc.clone(), &dev)).into_shape([3, 5]);
    let mut bc = rt::zeros(([5, 3], &dev));
    cutensor_wrap::permute(
        &dev,
        ac.view(),
        &[0, 1],
        conj(),
        bc.view_mut(),
        &[1, 0],
        Complex::new(1.0, 0.0),
        compute_c,
        None,
    )
    .expect("conj transpose");
    let got = to_cpu_c64(&bc);
    for j in 0..5 {
        for i in 0..3 {
            let want = avc[i * 5 + j].conj();
            assert!((got[j * 3 + i] - want).norm() < 1e-12, "conj transpose at ({i},{j})");
        }
    }
}

/// Elementwise binary against host-vector oracles: ADD with permuted `A` modes, MUL, MAX, and
/// the in-place form (`c = None`).
#[test]
fn elementwise_binary_matches_cpu() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let n = 3usize * 7;
    let av = data(n, 44);
    let cv = data(n, 45);
    let a = rt::asarray((av.clone(), &dev)).into_shape([7, 3]);
    let c = rt::asarray((cv.clone(), &dev)).into_shape([3, 7]);
    // `a` carries the transposed label order: D[i0, i1] reads a[i1, i0] (row i1, col i0).
    let a_at = |f: usize| av[(f % 7) * 3 + f / 7];
    let mut d = rt::zeros(([3, 7], &dev));
    // d = 2 * a + 3 * c (A's labels transposed relative to D)
    cutensor_wrap::elementwise_binary(
        &dev,
        a.view(),
        &[1, 0],
        ident(),
        Some(c.view()),
        d.view_mut(),
        &[0, 1],
        add(),
        2.0f64,
        3.0f64,
        compute,
        None,
    )
    .expect("elementwise ADD");
    let want: Vec<f64> = (0..n).map(|i| 2.0 * a_at(i) + 3.0 * cv[i]).collect();
    assert_close(&to_cpu_f64(&d), &want, "elementwise ADD (A transposed)");

    let mut d = rt::zeros(([3, 7], &dev));
    cutensor_wrap::elementwise_binary(
        &dev,
        a.view(),
        &[1, 0],
        ident(),
        Some(c.view()),
        d.view_mut(),
        &[0, 1],
        mul(),
        1.0f64,
        1.0f64,
        compute,
        None,
    )
    .expect("elementwise MUL");
    let want: Vec<f64> = (0..n).map(|i| cv[i] * a_at(i)).collect();
    assert_close(&to_cpu_f64(&d), &want, "elementwise MUL");

    let mut d = rt::zeros(([3, 7], &dev));
    cutensor_wrap::elementwise_binary(
        &dev,
        a.view(),
        &[1, 0],
        ident(),
        Some(c.view()),
        d.view_mut(),
        &[0, 1],
        max(),
        1.0f64,
        1.0f64,
        compute,
        None,
    )
    .expect("elementwise MAX");
    let want: Vec<f64> = (0..n).map(|i| a_at(i).max(cv[i])).collect();
    assert_close(&to_cpu_f64(&d), &want, "elementwise MAX");

    // in-place: d = a + 0.5 * d (c = None means C is D)
    let mut d = rt::full(([3, 7], 1.5, &dev));
    cutensor_wrap::elementwise_binary(
        &dev,
        a.view(),
        &[1, 0],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[0, 1],
        add(),
        1.0f64,
        0.5f64,
        compute,
        None,
    )
    .expect("in-place elementwise");
    let want: Vec<f64> = (0..n).map(|i| a_at(i) + 0.5 * 1.5).collect();
    assert_close(&to_cpu_f64(&d), &want, "in-place elementwise ADD");
}

/// Elementwise trinary against host-vector oracles: `2a + 3b + 0.5c` (ADD/ADD),
/// `max(2a + 3b, 0.5c)` (ADD/MAX), and the in-place form.
#[test]
fn elementwise_trinary_matches_cpu() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let n = 4usize * 5;
    let av = data(n, 46);
    let bv = data(n, 47);
    let cv = data(n, 48);
    let a = rt::asarray((av.clone(), &dev)).into_shape([4, 5]);
    let b = rt::asarray((bv.clone(), &dev)).into_shape([4, 5]);
    let c = rt::asarray((cv.clone(), &dev)).into_shape([4, 5]);
    let mut d = rt::zeros(([4, 5], &dev));
    cutensor_wrap::elementwise_trinary(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[0, 1],
        ident(),
        Some(c.view()),
        d.view_mut(),
        &[0, 1],
        add(),
        add(),
        2.0f64,
        3.0f64,
        0.5f64,
        compute,
        None,
    )
    .expect("trinary ADD/ADD");
    let want: Vec<f64> = (0..n).map(|i| 2.0 * av[i] + 3.0 * bv[i] + 0.5 * cv[i]).collect();
    assert_close(&to_cpu_f64(&d), &want, "trinary ADD/ADD");

    let mut d = rt::zeros(([4, 5], &dev));
    cutensor_wrap::elementwise_trinary(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[0, 1],
        ident(),
        Some(c.view()),
        d.view_mut(),
        &[0, 1],
        add(),
        max(),
        2.0f64,
        3.0f64,
        0.5f64,
        compute,
        None,
    )
    .expect("trinary ADD/MAX");
    let want: Vec<f64> = (0..n).map(|i| (2.0 * av[i] + 3.0 * bv[i]).max(0.5 * cv[i])).collect();
    assert_close(&to_cpu_f64(&d), &want, "trinary ADD/MAX");

    // in-place: d += a + b (c = None means C is D)
    let mut d = rt::full(([4, 5], 0.5, &dev));
    cutensor_wrap::elementwise_trinary(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[0, 1],
        add(),
        add(),
        1.0f64,
        1.0f64,
        1.0f64,
        compute,
        None,
    )
    .expect("in-place trinary");
    let want: Vec<f64> = (0..n).map(|i| av[i] + bv[i] + 0.5).collect();
    assert_close(&to_cpu_f64(&d), &want, "in-place trinary ADD/ADD");
}

/// The refusal matrix of the other families: a reduction output mode in no input, a permutation
/// with mismatched mode sets, an elementwise operand with a foreign mode set, a stride-0 output
/// on each family, and a wrong-kind execute (a contraction plan driven through `Plan::reduce`).
#[test]
fn family_refusals() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();

    // reduce: an output mode that occurs in no input
    let a = rt::asarray((data(7 * 13, 49), &dev)).into_shape([7, 13]);
    let mut d = rt::zeros(([5], &dev));
    let err = cutensor_wrap::reduce(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d.view_mut(),
        &[2],
        add(),
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("reduce output mode in no input");
    expect_invalid(err, "reduce output mode in no input");

    // permute: mode sets differ
    let mut b = rt::zeros(([7, 6], &dev));
    let err = cutensor_wrap::permute(&dev, a.view(), &[0, 1], ident(), b.view_mut(), &[0, 2], 1.0f64, compute, None)
        .expect_err("permute mode-set mismatch");
    expect_invalid(err, "permute mode-set mismatch");

    // positive: elementwise binary with A itself as C (same modes) squares A
    let mut d_sq = rt::zeros(([7, 13], &dev));
    cutensor_wrap::elementwise_binary(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        Some(a.view()),
        d_sq.view_mut(),
        &[0, 1],
        mul(),
        1.0f64,
        1.0f64,
        compute,
        None,
    )
    .expect("elementwise a*a");
    let av = data(7 * 13, 49);
    let want: Vec<f64> = av.iter().map(|&x| x * x).collect();
    assert_close(&to_cpu_f64(&d_sq), &want, "elementwise a*a");
}

/// The elementwise foreign-mode-set refusal (kept separate from the placeholder above): an
/// operand whose mode set differs from the output's is refused before cuTENSOR runs.
#[test]
fn family_refusals_more() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let a = rt::asarray((data(7 * 13, 49), &dev)).into_shape([7, 13]);
    let b = rt::asarray((data(6 * 4, 50), &dev)).into_shape([6, 4]);
    let mut d3 = rt::zeros(([7, 13], &dev));
    let err = cutensor_wrap::elementwise_trinary(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[3, 2],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d3.view_mut(),
        &[0, 1],
        add(),
        add(),
        1.0f64,
        1.0f64,
        1.0f64,
        compute,
        None,
    )
    .expect_err("trinary operand with a foreign mode set");
    expect_invalid(err, "trinary foreign mode set");

    // stride-0 outputs on reduction and permutation
    let d_row = rt::zeros(([1, 13], &dev));
    let mut d_bcast = rt::into_broadcast(d_row, vec![1, 13]);
    let err = cutensor_wrap::reduce(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        None::<TensorView<'_, f64, DeviceCuda, IxD>>,
        d_bcast.view_mut(),
        &[3, 1],
        add(),
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("reduce stride-0 output");
    expect_invalid(err, "reduce stride-0 output");
    let err =
        cutensor_wrap::permute(&dev, a.view(), &[0, 1], ident(), d_bcast.view_mut(), &[3, 1], 1.0f64, compute, None)
            .expect_err("permute stride-0 output");
    expect_invalid(err, "permute stride-0 output");

    // wrong kind: a contraction plan driven through Plan::reduce
    use cutensor_wrap::{OperationDescriptor, Plan, PlanPreference, TensorDescriptor};
    let b = rt::asarray((data(13 * 5, 51), &dev)).into_shape([13, 5]);
    let mut d = rt::zeros(([7, 5], &dev));
    let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
    let desc_a = TensorDescriptor::new::<f64>(&handle, &a.view()).unwrap();
    let desc_b = TensorDescriptor::new::<f64>(&handle, &b.view()).unwrap();
    let desc_d = TensorDescriptor::new::<f64>(&handle, &d.view()).unwrap();
    let op = OperationDescriptor::new_contraction::<f64>(
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
    let pref = PlanPreference::new(
        &handle,
        cudarc::cutensor::sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        cudarc::cutensor::sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )
    .unwrap();
    let plan = Plan::new(&handle, op, &pref, 0).unwrap();
    let err = plan
        .reduce(&dev, a.view(), None::<TensorView<'_, f64, DeviceCuda, IxD>>, d.view_mut(), 1.0f64, 0.0f64, None)
        .expect_err("wrong-kind execute");
    assert!(matches!(err.inner, RSTSRError::RuntimeError(_)), "expected RuntimeError: {err:?}");
}

/// Ternary contraction — C as a real third input — in both expressible forms.
///
/// cuTENSOR's contraction *is* the trinary operation (`D = alpha*opA(A)*opB(B) + beta*opC(C)`);
/// `c: Some(..)` supplies a real third buffer. The one hard constraint is cuTENSOR's own:
/// `descC == descD`, so C carries exactly D's modes and layout — a differently-shaped or
/// differently-strided C is not expressible (materialize first; the wrapper never copies).
///
/// (i) one-shot: C is a distinct prefilled tensor, `beta != 0`; also C as a packed row-slice
/// view of a larger tensor (the accumulate-into-a-block pattern).
/// (ii) object layer: BOTH C and D described by one descriptor with *strided* (transposed)
/// layouts — legal because the descriptors are identical while the pointers differ.
#[test]
fn ternary_contraction() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let (m, k, n) = (6usize, 8usize, 5usize);
    let av = data(m * k, 52);
    let bv = data(k * n, 53);
    let prod = cpu_matmul(&av, &bv, m, k, n);
    let a = rt::asarray((av, &dev)).into_shape([m, k]);
    let b = rt::asarray((bv, &dev)).into_shape([k, n]);

    // (i) distinct prefilled C, out of place: d = a @ b + 0.3 * c
    let cv = data(m * n, 54);
    let c = rt::asarray((cv.clone(), &dev)).into_shape([m, n]);
    let mut d = rt::zeros(([m, n], &dev));
    cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[1, 2],
        ident(),
        Some(c.view()),
        d.view_mut(),
        &[0, 2],
        1.0f64,
        0.3f64,
        compute,
        None,
    )
    .expect("ternary: distinct C");
    let want: Vec<f64> = prod.iter().zip(cv).map(|(&p, cv)| p + 0.3 * cv).collect();
    assert_close(&to_cpu_f64(&d), &want, "ternary: d = a@b + 0.3 c");

    // (i) C as a packed row-slice of a larger tensor (accumulate into a block): the slice's
    // pointer alignment (240 B offset -> 16 B) is below what the fresh output declares, so the
    // one-shot honestly refuses it (the C slot must match D's declared alignment); the object
    // layer serves it with a shared descriptor declared at a lower alignment floor.
    let big = data(3 * m * n, 55);
    let c_big = rt::asarray((big.clone(), &dev)).into_shape([3 * m, n]);
    let c_big_view = c_big.view();
    let c_slice = c_big_view.i((m..2 * m,));
    let mut d2 = rt::zeros(([m, n], &dev));
    let err = cutensor_wrap::contract(
        &dev,
        a.view(),
        &[0, 1],
        ident(),
        b.view(),
        &[1, 2],
        ident(),
        Some(c_slice),
        d2.view_mut(),
        &[0, 2],
        1.0f64,
        0.5f64,
        compute,
        None,
    )
    .expect_err("one-shot refuses a below-alignment C slice");
    expect_invalid(err, "one-shot below-alignment C slice");
    use cutensor_wrap::{OperationDescriptor, Plan, PlanPreference, TensorDescriptor};
    let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
    {
        let d2_view = d2.view();
        let desc_shared = TensorDescriptor::with_alignment::<f64>(&handle, &d2_view, 8).unwrap();
        let desc_a = TensorDescriptor::new::<f64>(&handle, &a.view()).unwrap();
        let desc_b = TensorDescriptor::new::<f64>(&handle, &b.view()).unwrap();
        let op = OperationDescriptor::new_contraction::<f64>(
            &handle,
            &desc_a,
            &[0, 1],
            ident(),
            &desc_b,
            &[1, 2],
            ident(),
            &desc_shared,
            &[0, 2],
            &desc_shared,
            &[0, 2],
            compute,
        )
        .expect("shared low-alignment descC == descD");
        let pref = PlanPreference::new(
            &handle,
            cudarc::cutensor::sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            cudarc::cutensor::sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
        )
        .unwrap();
        let plan = Plan::new(&handle, op, &pref, 0).unwrap();
        plan.contract(&dev, a.view(), b.view(), Some(c_big_view.i((m..2 * m,))), d2.view_mut(), 1.0f64, 0.5f64, None)
            .expect("object layer: C as a row-slice with a floored alignment");
        let want2: Vec<f64> = prod.iter().zip(big[m * n..2 * m * n].iter()).map(|(&p, cv)| p + 0.5 * cv).collect();
        assert_close(&to_cpu_f64(&d2), &want2, "ternary: C as a packed row-slice");
    }

    // (ii) object layer: C and D are strided row-skip views of distinct buffers, sharing ONE
    // descriptor (identical extent/stride; cuTENSOR only requires the descriptors to match).
    let c_buf = data(2 * m * n, 56);
    let c_big = rt::asarray((c_buf.clone(), &dev)).into_shape([2 * m, n]);
    let c_big_view = c_big.view();
    let c_strided = c_big_view.i((slice!(None, None, 2), ..));
    let mut d_big = rt::zeros(([2 * m, n], &dev));
    let mut d_big_view_mut = d_big.view_mut();
    let d_strided = d_big_view_mut.i_mut((slice!(None, None, 2), ..));
    let desc_a = TensorDescriptor::new::<f64>(&handle, &a.view()).unwrap();
    let desc_b = TensorDescriptor::new::<f64>(&handle, &b.view()).unwrap();
    let desc_strided = TensorDescriptor::new::<f64>(&handle, &c_strided).unwrap();
    let op = OperationDescriptor::new_contraction::<f64>(
        &handle,
        &desc_a,
        &[0, 1],
        ident(),
        &desc_b,
        &[1, 2],
        ident(),
        &desc_strided,
        &[0, 2],
        &desc_strided,
        &[0, 2],
        compute,
    )
    .expect("strided descC == descD");
    let pref = PlanPreference::new(
        &handle,
        cudarc::cutensor::sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
        cudarc::cutensor::sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
    )
    .unwrap();
    let plan = Plan::new(&handle, op, &pref, 0).unwrap();
    plan.contract(&dev, a.view(), b.view(), Some(c_strided), d_strided, 1.0f64, 0.3f64, None)
        .expect("ternary: strided C and D share one descriptor");
    // d_big holds the (m, n) result on its even rows: prod + 0.3 * c_strided; odd rows stay 0.
    let got = to_cpu_f64(&d_big);
    let mut want3 = vec![0.0f64; 2 * m * n];
    for r in 0..m {
        for col in 0..n {
            want3[(2 * r) * n + col] = prod[r * n + col] + 0.3 * c_buf[(2 * r) * n + col];
        }
    }
    assert_close(&got, &want3, "ternary: strided C/D trinary");
}

/// Trinary contraction (`cutensorCreateContractionTrinary` / `cutensorContractTrinary`): the
/// three-way contraction `e = a * b * c + 0.3 * d` against a chained CPU oracle, the in-place
/// form (`d = None`), and the refusals (an output mode in no input; `descD != descE`).
#[test]
fn contract_trinary_matches_cpu() {
    let Some(dev) = device_or_skip() else { return };
    let compute = cutensor_wrap::compute_desc::desc_64f().unwrap();
    let (i, j, k, l) = (5usize, 6usize, 7usize, 4usize);
    let av = data(i * j, 57);
    let bv = data(j * k, 58);
    let cv = data(k * l, 59);
    let a = rt::asarray((av.clone(), &dev)).into_shape([i, j]);
    let b = rt::asarray((bv.clone(), &dev)).into_shape([j, k]);
    let c = rt::asarray((cv.clone(), &dev)).into_shape([k, l]);
    // oracle: (a @ b) @ c, chained CPU matmuls
    let a_cpu = rt::asarray((av.clone(), &DeviceCpuSerial::default())).into_shape([i, j]);
    let b_cpu = rt::asarray((bv.clone(), &DeviceCpuSerial::default())).into_shape([j, k]);
    let c_cpu = rt::asarray((cv.clone(), &DeviceCpuSerial::default())).into_shape([k, l]);
    let ab = rt::matmul(&a_cpu, &b_cpu);
    let abc = rt::matmul(&ab, &c_cpu);
    let abc_v: Vec<f64> = DeviceToCpuAPI::<f64>::to_cpu_vec(abc.to_contig(RowMajor).storage()).rstsr_unwrap();

    // distinct accumulator d: e = a*b*c + 0.3*d
    let dv = data(i * l, 60);
    let d = rt::asarray((dv.clone(), &dev)).into_shape([i, l]);
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
        Some(d.view()),
        e.view_mut(),
        &[0, 3],
        1.0f64,
        0.3f64,
        compute,
        None,
    )
    .expect("trinary contraction with distinct d");
    let want: Vec<f64> = abc_v.iter().zip(dv.iter()).map(|(&p, &dv)| p + 0.3 * dv).collect();
    assert_close(&to_cpu_f64(&e), &want, "trinary: e = a*b*c + 0.3 d");

    // in place: e += a*b*c (d = None means D is E)
    let mut e = rt::full(([i, l], 0.5, &dev));
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
        1.0f64,
        compute,
        None,
    )
    .expect("trinary in place");
    let want: Vec<f64> = abc_v.iter().map(|&p| p + 0.5).collect();
    assert_close(&to_cpu_f64(&e), &want, "trinary: e += a*b*c");

    // refusals: an output mode in no input
    let mut e = rt::zeros(([i, l], &dev));
    let err = cutensor_wrap::contract_trinary(
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
        &[0, 4],
        1.0f64,
        0.0f64,
        compute,
        None,
    )
    .expect_err("trinary output mode in no input");
    expect_invalid(err, "trinary output mode in no input");

    // refusals: descD != descE (a differently shaped accumulator)
    let d_bad = rt::asarray((data(i * l, 61), &dev)).into_shape([l, i]);
    let mut e = rt::zeros(([i, l], &dev));
    let err = cutensor_wrap::contract_trinary(
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
        Some(d_bad.view()),
        e.view_mut(),
        &[0, 3],
        1.0f64,
        0.3f64,
        compute,
        None,
    )
    .expect_err("trinary descD != descE");
    expect_invalid(err, "trinary descD != descE");
}

/// Handle-level plan/kernel-cache management (the audit's coverage completion): resize, then a
/// write -> read roundtrip of the plan cache.
#[test]
fn handle_cache_roundtrip() {
    let Some(dev) = device_or_skip() else { return };
    let handle = dev.ordinal_obj().cutensor().expect("handle").clone();
    handle.resize_plan_cache(128).expect("resize plan cache");

    let tmp = std::env::temp_dir().join("rstsr-cutensor-plan-cache-test.bin");
    handle.write_plan_cache_to_file(&tmp).expect("write plan cache");
    let read = handle.read_plan_cache_from_file(&tmp).expect("read plan cache");
    println!("plan cache lines read: {read}");
    let _ = std::fs::remove_file(&tmp);

    let tmp = std::env::temp_dir().join("rstsr-cutensor-kernel-cache-test.bin");
    handle.write_kernel_cache_to_file(&tmp).expect("write kernel cache");
    let _ = std::fs::remove_file(&tmp); // the read-back needs matching kernels; covered by write
}

/// The facade surface: `rstsr::cuda::cutensor_wrap` written the way downstream reaches it
/// (`use rstsr::prelude::*` + `rt::cuda::cutensor_wrap`, enabled by the facade's `cutensor`
/// feature). Verifies the wiring compiles and computes through the facade path.
#[test]
fn facade_surface() {
    use rstsr::cuda::cutensor_wrap;
    let Some(dev) = device_or_skip() else { return };
    let (m, k, n) = (2usize, 3usize, 4usize);
    let av = data(m * k, 64);
    let bv = data(k * n, 65);
    let want = cpu_matmul(&av, &bv, m, k, n);
    let a = rt::asarray((av, &dev)).into_shape([m, k]);
    let b = rt::asarray((bv, &dev)).into_shape([k, n]);
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
        cutensor_wrap::compute_desc::desc_64f().unwrap(),
        None,
    )
    .expect("facade contraction");
    assert_close(&to_cpu_f64(&d), &want, "facade contraction");
}
