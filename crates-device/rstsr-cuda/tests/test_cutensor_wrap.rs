//! Contraction tests of the cuTENSOR wrapper, on hardware: every serveable dtype against a CPU
//! oracle, the view shapes of real callers, the accumulation forms, and the refusal matrix.
//!
//! The wrapper is memory-safe by construction; what the tests establish is *agreement* (the
//! wrapper computes what the CPU device computes, within the dtype's summation error) and
//! *honesty* (every case the probes showed to be accepted-with-garbage or undocumented is refused
//! by name before cuTENSOR runs). Skips say why: no CUDA device, or `libcutensor` not loadable —
//! a green run without the skip lines means everything ran. Point `LD_LIBRARY_PATH` at another
//! version's `libcutensor.so` symlink dir to run the file against that version.

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

/// `opA = CONJ` on `c64` computes `conj(A) @ B` (probe A6/D9 behaviour through the wrapper).
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
