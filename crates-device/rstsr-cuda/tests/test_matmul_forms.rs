//! The symmetric form of the matmul: `a @ a.t()` with a fresh output runs as one `?syrk` call
//! whose triangle is mirrored across the diagonal. The cases the form must not take (`beta != 0`,
//! the conjugate transpose, another buffer, a batch, a stepped operand) run the gemm chain; both
//! give the full symmetric matrix, so every test compares the whole output against a reference.

use num::Complex;
use rstsr::prelude::*;
use rstsr_cuda::dtype::CudaDType;
use rstsr_cuda::matmul::{CudaMatMulMode, MatMulDrivers, MatMulUseSyrk};
use rstsr_cuda::prelude_dev::*;
use rstsr_cuda::registry::is_available;

/// Uniform data in `[-1, 1]` with an imaginary part in `[0, 1]`.
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn data_c(n: usize, seed: u64) -> Vec<Complex<f64>> {
    let (re, im) = (data(n, seed), data(n, seed ^ 0x9e37_79b9));
    re.iter().zip(&im).map(|(&x, &y)| Complex::new(x, (y + 1.0) / 2.0)).collect()
}

macro_rules! require_cuda {
    () => {
        if !std::panic::catch_unwind(is_available).unwrap_or(false) {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

type Cpu<T> = Tensor<T, DeviceCpuSerial, IxD>;

/// A device tensor back on the host, contiguous.
fn host<T: CudaDType + Copy>(t: &Tensor<T, DeviceCuda, IxD>) -> Vec<T> {
    DeviceToCpuAPI::<T>::to_cpu_vec(t.to_contig(RowMajor).storage()).rstsr_unwrap()
}

/// A CPU tensor as one flat vector.
fn host_cpu<T: Clone>(t: &Cpu<T>) -> Vec<T> {
    let n: usize = t.shape().iter().product();
    t.to_contig(RowMajor).view().into_shape([n]).to_owned().to_vec()
}

/// `a @ a.t()` on every path: the chain (the form first), the classic driver alone, and the chain
/// without it; compared with the CPU device, in both orders. `a` stays a **view** into a larger
/// buffer in both orientations — a real row-strided operand, offset not the buffer base, leading
/// dimension the big matrix's row stride. A copy would test an ordinary contiguous gemm, and the
/// copied transpose is a look-alike the form must not take (it has its own test below).
fn syrk_views<
    S: CudaDType + CudaMatMulMode + Copy + Clone + Send + Sync + 'static + num::Zero + num::One + PartialEq,
>(
    what: &str,
    close: fn(S, S) -> bool,
    mk: fn(Vec<Complex<f64>>) -> Vec<S>,
) {
    require_cuda!();
    let (n, k) = (37usize, 19usize);
    let v = data_c((n + 2) * (k + 3), 7);
    let (zero, one, fill) = (
        mk(vec![Complex::new(0.0, 0.0)])[0],
        mk(vec![Complex::new(1.0, 0.0)])[0],
        mk(vec![Complex::new(0.5, -0.25)])[0],
    );
    for order in [RowMajor, ColMajor] {
        // the CPU reference takes the same default order as the device, so the two big matrices
        // are the same tensor laid out the same way and the views mean the same thing
        let mut dcpu = DeviceCpuSerial::default();
        dcpu.set_default_order(order);
        let cpu_big: Cpu<S> = rt::asarray((mk(v.clone()), &dcpu)).into_shape([n + 2, k + 3]);
        for drivers in [MatMulDrivers::all(), MatMulDrivers::CUBLAS, MatMulDrivers::all().minus(MatMulDrivers::CUBLAS)]
        {
            let ctx = |name: &str| format!("{what}, {order:?}, {drivers:?}, {name}");
            let mut flags = CudaRuntimeFlags::default();
            flags.matmul_drivers = drivers;
            let mut d = DeviceCuda::new(0).with_flags(flags);
            d.set_default_order(order);
            let big: Tensor<S, DeviceCuda, IxD> = rt::asarray((mk(v.clone()), &d)).into_shape([n + 2, k + 3]);
            // the two orientations of a real strided view; the roles swap with the device order
            let cpu_big_t = cpu_big.t();
            let big_t = big.t();
            let cpu_a = cpu_big.i(s![1..n + 1, 2..k + 2]);
            let cpu_a2 = cpu_big_t.i(s![2..k + 2, 1..n + 1]);
            let dev_a = big.i(s![1..n + 1, 2..k + 2]);
            let dev_a2 = big_t.i(s![2..k + 2, 1..n + 1]);
            for (name, c, a) in [("a strided view", &cpu_a, &dev_a), ("a of the other orientation", &cpu_a2, &dev_a2)] {
                let want = host_cpu(&rt::matmul(c, c.t()));
                let got = host(&rt::matmul(a, a.t()));
                for (x, y) in got.iter().zip(&want) {
                    assert!(close(*x, *y), "{}: {x:?} vs {y:?}", ctx(name));
                }
            }
            // k = 0: the product is exactly the zero matrix
            let g0 = big.i(s![1..n + 1, 2..2]);
            let got = host(&rt::matmul(&g0, g0.t()));
            assert!(got.iter().all(|&x| x == zero), "{}: {:?}", ctx("k = 0"), &got[..4]);
            // n = 1 and k = 1
            for (name, cs, gs) in [
                ("n = 1", cpu_big.i(s![1..2, 2..k + 2]), big.i(s![1..2, 2..k + 2])),
                ("k = 1", cpu_big.i(s![1..n + 1, 2..3]), big.i(s![1..n + 1, 2..3])),
            ] {
                let want = host_cpu(&rt::matmul(&cs, cs.t()));
                let got = host(&rt::matmul(&gs, gs.t()));
                for (x, y) in got.iter().zip(&want) {
                    assert!(close(*x, *y), "{}: {x:?} vs {y:?}", ctx(name));
                }
            }
            // alpha = 0 with beta = 0 into a caller's buffer: exactly the zeros it asks for
            let mut c = rt::full(([n * n], fill, &d)).into_shape([n, n]);
            rt::matmul_from(&mut c, &dev_a, dev_a.t(), zero, zero);
            let got = host(&c);
            assert!(got.iter().all(|&x| x == zero), "{}: {:?}", ctx("alpha = 0, beta = 0"), &got[..4]);
            // a strided output view inside a larger buffer: the block is written, nothing else is
            let want = host_cpu(&rt::matmul(&cpu_a, cpu_a.t()));
            let mut zbig: Tensor<S, DeviceCuda, IxD> =
                rt::asarray((mk(vec![Complex::new(0.0, 0.0); 4 * n * n]), &d)).into_shape([2 * n, 2 * n]);
            {
                let mut block = zbig.i_mut(s![..n, ..n]);
                rt::matmul_from(&mut block, &dev_a, dev_a.t(), one, zero);
            }
            let check_block = |got: &[S], name: &str| {
                for (i, x) in got.iter().enumerate() {
                    let (r, co) = (i / (2 * n), i % (2 * n));
                    let y = if r < n && co < n { want[r * n + co] } else { zero };
                    assert!(close(*x, y), "{}: [{r},{co}] {x:?} vs {y:?}", ctx(name));
                }
            };
            check_block(&host(&zbig), "c strided view");
            // the transposed output view has no column-major expression: the tiled/naive case,
            // which only the whole chain serves
            if drivers == MatMulDrivers::all() {
                let mut zbig: Tensor<S, DeviceCuda, IxD> =
                    rt::asarray((mk(vec![Complex::new(0.0, 0.0); 4 * n * n]), &d)).into_shape([2 * n, 2 * n]);
                {
                    let mut whole = zbig.view_mut().into_reverse_axes();
                    let mut block = whole.i_mut(s![..n, ..n]);
                    rt::matmul_from(&mut block, &dev_a, dev_a.t(), one, zero);
                }
                check_block(&host(&zbig), "c strided view, transposed");
            }
        }
    }
}

#[test]
fn syrk_f64() {
    syrk_views::<f64>(
        "f64",
        |x, y| (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
        |v| v.into_iter().map(|z| z.re).collect(),
    );
}

#[test]
fn syrk_c32() {
    syrk_views::<Complex<f32>>(
        "c32",
        |x, y| (x - y).norm() <= 1e-5 * (1.0 + y.norm()),
        |v| v.into_iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect(),
    );
}

#[test]
fn syrk_c64() {
    syrk_views::<Complex<f64>>("c64", |x, y| (x - y).norm() <= 1e-12 * (1.0 + y.norm()), |v| v);
}

/// The cases the form must leave alone give the same symmetric (or not) matrix through the gemm
/// chain: `beta != 0`, the conjugate transpose, a copy of the transpose, a batch, and a stepped
/// operand.
#[test]
fn syrk_look_alikes_take_the_chain() {
    require_cuda!();
    let (n, k, p) = (600usize, 20usize, 3usize);
    let v = data_c(n * k, 11);
    // the references are row-major on the CPU; the device operands take the order's own layout
    let dcpu = DeviceCpuSerial::default();
    for order in [RowMajor, ColMajor] {
        let mut d = DeviceCuda::new(0);
        d.set_default_order(order);
        // `asarray` lays data out in the device's default order: give it that flattening
        let flat = |dims: &[usize], x: &[Complex<f64>]| -> Vec<Complex<f64>> {
            if order == RowMajor {
                return x.to_vec();
            }
            let mut out = vec![Complex::new(0.0, 0.0); x.len()];
            for (pos, _) in x.iter().enumerate() {
                let (mut idx, mut rem) = (0, pos);
                for &dim in dims.iter().rev() {
                    let (q, r) = (rem / dim, rem % dim);
                    rem = q;
                    idx = idx * dim + r;
                }
                out[idx] = x[pos];
            }
            out
        };
        let cpu: Cpu<Complex<f64>> = rt::asarray((v.clone(), &dcpu)).into_shape([n, k]);
        let vc: Vec<Complex<f64>> = v.iter().map(|z| z.conj()).collect();
        let cpu_ah: Cpu<Complex<f64>> = rt::asarray((vc.clone(), &dcpu)).into_shape([n, k]);
        let ah: Tensor<Complex<f64>, DeviceCuda, IxD> = rt::asarray((flat(&[n, k], &vc), &d)).into_shape([n, k]);
        let a: Tensor<Complex<f64>, DeviceCuda, IxD> = rt::asarray((flat(&[n, k], &v), &d)).into_shape([n, k]);
        let cp = |t: &Tensor<Complex<f64>, DeviceCuda, IxD>, w: &Cpu<Complex<f64>>, what: &str| {
            let (got, want) = (host(t), host_cpu(w));
            for (x, y) in got.iter().zip(&want) {
                assert!((x - y).norm() <= 1e-12 * (1.0 + y.norm()), "{what}, {order:?}: {x} vs {y}");
            }
        };

        // beta != 0: into a caller's (non-symmetric) output
        let (w, g) = (data_c(n * n, 5), data_c(n * n, 5));
        let mut w: Cpu<Complex<f64>> = rt::asarray((w, &dcpu)).into_shape([n, n]);
        let mut g: Tensor<Complex<f64>, DeviceCuda, IxD> = rt::asarray((flat(&[n, n], &g), &d)).into_shape([n, n]);
        rt::matmul_from(&mut w, &cpu, cpu.t(), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
        rt::matmul_from(&mut g, &a, a.t(), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0));
        cp(&g, &w, "beta != 0");

        // the conjugate transpose: `conj(a)` stored as its own tensor, then transposed
        cp(&rt::matmul(&a, ah.t()), &rt::matmul(&cpu, cpu_ah.t()), "a @ a.h()");

        // a copy of the transpose in another buffer
        let (w, g) = (data_c(n * k, 9), data_c(n * k, 9));
        let w: Cpu<Complex<f64>> = rt::asarray((w, &dcpu)).into_shape([k, n]);
        let g: Tensor<Complex<f64>, DeviceCuda, IxD> = rt::asarray((flat(&[k, n], &g), &d)).into_shape([k, n]);
        cp(&rt::matmul(&a, &g), &rt::matmul(&cpu, &w), "another buffer");

        // a batch of symmetric products; the batch axis leads under row-major and trails under
        // column-major (the matmul axes are the last two resp. first two axes of an operand)
        let vb = data_c(p * n * k, 13);
        let wb: Cpu<Complex<f64>> = rt::asarray((vb.clone(), &dcpu)).into_shape([p, n, k]);
        let want = rt::matmul(&wb, wb.transpose([0, 2, 1]));
        let got: Tensor<Complex<f64>, DeviceCuda, IxD> = if order == RowMajor {
            let a = rt::asarray((vb, &d)).into_shape([p, n, k]);
            rt::matmul(&a, a.transpose([0, 2, 1]))
        } else {
            // logical (q, i, j) is stored at `i + j * n + q * n * k`
            let mut flat = vec![Complex::new(0.0, 0.0); p * n * k];
            for i in 0..n {
                for j in 0..k {
                    for q in 0..p {
                        flat[i + j * n + q * n * k] = vb[q * n * k + i * k + j];
                    }
                }
            }
            let a = rt::asarray((flat, &d)).into_shape([n, k, p]);
            let b = a.transpose([1, 0, 2]);
            rt::matmul(&a, &b).transpose([2, 0, 1]).to_owned()
        };
        cp(&got, &want, "batched");

        // a stepped operand: no unit stride anywhere, so no cuBLAS expression
        let vs = data_c(4 * n * k, 17);
        let ws: Cpu<Complex<f64>> = rt::asarray((vs.clone(), &dcpu)).into_shape([2 * n, 2 * k]);
        let gs: Tensor<Complex<f64>, DeviceCuda, IxD> =
            rt::asarray((flat(&[2 * n, 2 * k], &vs), &d)).into_shape([2 * n, 2 * k]);
        let ws = ws.i(s![slice!(None, None, 2), slice!(None, None, 2)]).to_owned();
        let gs = gs.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
        cp(&rt::matmul(&gs, gs.t()), &rt::matmul(&ws, ws.t()), "stepped");
    }
}

/// The flag chooses the output types: `f32` takes the form under `Allow` only, `f64` never under
/// `Disallow`. Every setting gives the gemm's answer (a thin operand and a square one).
#[test]
fn syrk_follows_the_flag() {
    require_cuda!();
    let cells: [(usize, usize); 2] = [(2048, 64), (512, 512)];
    for use_syrk in [MatMulUseSyrk::Allow, MatMulUseSyrk::Scientific, MatMulUseSyrk::Disallow] {
        // the reference never takes the form: the classic driver off switches it off
        for ((n, k), t) in cells.iter().flat_map(|c| [(*c, "f32"), (*c, "f64")]) {
            let mk = |flags: CudaRuntimeFlags, no_form: bool| -> Vec<f64> {
                let mut flags = flags;
                if no_form {
                    flags.matmul_drivers = flags.matmul_drivers.minus(MatMulDrivers::CUBLAS);
                }
                let d = DeviceCuda::new(0).with_flags(flags);
                if t == "f32" {
                    let v: Vec<f32> = data(n * k, 7).into_iter().map(|x| x as f32).collect();
                    let a = rt::asarray((v, &d)).into_shape([n, k]);
                    host(&rt::matmul(&a, a.t())).into_iter().map(f64::from).collect()
                } else {
                    let v = data(n * k, 7);
                    let a = rt::asarray((v, &d)).into_shape([n, k]);
                    host(&rt::matmul(&a, a.t()))
                }
            };
            let mut flags = CudaRuntimeFlags::default();
            flags.matmul_use_syrk = use_syrk;
            let (got, want) = (mk(flags, false), mk(flags, true));
            for (x, y) in got.iter().zip(&want) {
                assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()), "{use_syrk:?}, {t}, ({n}, {k}): {x} vs {y}");
            }
        }
    }
}

/// The rest of the inner dot's cases: mixed dtypes on the naive driver (`f16` operands, an `f32`
/// output), a large `k` against the CPU device, and the rank-0 output next to the `[1]` and
/// `[1, 1]` shapes of the same product.
#[test]
fn dot_mixed_large_and_shapes() {
    require_cuda!();
    // mixed dtypes: only the naive driver serves `f16 -> f32` with a rank-0 output
    {
        let k = 1000usize;
        let v = data(k, 21);
        let mut flags = CudaRuntimeFlags::default();
        flags.matmul_drivers = MatMulDrivers::NAIVE;
        let d = DeviceCuda::new(0).with_flags(flags);
        let f16v = |x: f64| half::f16::from_f64(x);
        let a: Tensor<half::f16, DeviceCuda, IxD> =
            rt::asarray((v.iter().map(|&x| f16v(x)).collect::<Vec<_>>(), &d)).into_shape(vec![k]);
        let got = a.matmul_with_mode(&a.t(), ComputeType32F::Default).out::<f32>();
        let want: f32 = v.iter().map(|&x| f16v(x).to_f64() * f16v(x).to_f64()).sum::<f64>() as f32;
        assert!((got.to_scalar() - want).abs() <= 1e-4 * want.abs(), "f16 -> f32 dot");
    }
    // a large k: many blocks on the memory-bound path
    {
        let k = 1usize << 22;
        let v = data(k, 23);
        let d = DeviceCuda::new(0);
        let a: Tensor<f64, DeviceCuda, IxD> = rt::asarray((v.clone(), &d)).into_shape(vec![k]);
        let got = rt::matmul(&a, a.t()).to_scalar();
        let want: f64 = v.iter().map(|&x| x * x).sum();
        assert!((got - want).abs() <= 1e-12 * want, "large dot");
    }
    // the shapes around the rank-0 one give the same single number
    {
        let k = 40usize;
        let v = data(k, 25);
        let d = DeviceCuda::new(0);
        let dot: Tensor<f64, DeviceCuda, IxD> = rt::asarray((v.clone(), &d)).into_shape(vec![k]);
        let want = rt::matmul(&dot, dot.t()).to_scalar();
        let row: Tensor<f64, DeviceCuda, IxD> = rt::asarray((v.clone(), &d)).into_shape([1, k]);
        let col: Tensor<f64, DeviceCuda, IxD> = rt::asarray((v, &d)).into_shape([k, 1]);
        for (what, got) in [
            ("[1, k] @ [k]", rt::matmul(&row, dot.t()).to_scalar()),
            ("[k] @ [k, 1]", rt::matmul(&dot, &col).to_scalar()),
            ("[1, k] @ [k, 1]", rt::matmul(&row, &col).to_scalar()),
        ] {
            assert!((got - want).abs() <= 1e-13 * want.abs(), "{what}: {got} vs {want}");
        }
    }
}
