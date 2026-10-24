//! Elementwise benchmark cells: contiguous / strided / broadcast / assign
//! scenarios over f64, median-of-R wall timing per cell (sync once per
//! repeat; each op enqueued once per iteration).
//!
//! Output: one `cell,median_ms` CSV line per scenario. `RSTSR_BENCH_ITER` /
//! `RSTSR_BENCH_REPEAT` override iteration and repeat counts (default 200/5).
//!
//! Run: cargo bench --bench bench_elementwise
//!
//! Cells (2-D operands 2048x2048 f64 = 32 MiB; 3-D operands 512^3 = 1 GiB):
//!   fill          C = 2.5
//!   add           C = A + B                    (contig)
//!   add_t         C = A + B.T                  (strided B)
//!   add_t_self    C = A + A.T
//!   add_swap3d    C = A + swapaxes(A,0,1)      ([512,512,512])
//!   add_outer     C = a[None,:] + b[:,None]
//!   add_bcast_r   C = A + B[:,None,:]          ([512,1,512] B)
//!   add_bcast_c   C = A + B[:,:,None]          ([512,512,1] B)
//!   add_slice     C = A[:,10:1010] + B[:,50:1050]
//!   add_step      C = A[:, :1024] + B[:, ::2]
//!   add_flip      C = A[:, :1024] + flip(B,1)[:, :1024]
//!   where_tt      C = where(mask, A, B)
//!   where_ts      C = where(mask, A, 0.5)
//!   clip          C = clip(A, 1000, 3e6)       (scalar bounds)
//!   rem           C = A % B
//!   sin           C = sin(A)
//!   pow           C = A ** 2.5
//!   copy_T        C.T = A                      (strided output assign)
//!   to_contig     C = B.T.to_contig()          (copy-on-write path)

use std::time::Instant;

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

fn bench_cell<F: FnMut()>(d: &DeviceCuda, name: &str, n_iter: usize, repeat: usize, mut f: F) -> f64 {
    // warmup: NVRTC compile + driver JIT on first call
    for _ in 0..30 {
        f();
    }
    funcs::synchronize(d).unwrap();
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        for _ in 0..n_iter {
            f();
        }
        funcs::synchronize(d).unwrap();
        samples.push(t0.elapsed().as_secs_f64() / n_iter as f64);
    }
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = samples[samples.len() / 2];
    println!("{name},{:.6}", med * 1e3);
    med
}

fn main() {
    if !funcs::is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let d = DeviceCuda::new(0);
    let n_iter: usize = std::env::var("RSTSR_BENCH_ITER").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let repeat: usize = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let n = 2048usize; // 2-D operand: 32 MiB f64

    println!("cell,median_ms");

    // power-heavy warm-up: long transcendental chains spin the clocks up
    // from idle (short elementwise kernels never trigger the boost)
    {
        let w = rt::arange((0., (n * n) as f64, 1., &d)).into_shape([n, n]);
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs_f64() < 5.0 {
            let m = rt::pow(&rt::sin(&w), 2.5);
            std::hint::black_box(&m);
        }
        funcs::synchronize(&d).unwrap();
    }

    // fill
    let dt_fill = bench_cell(&d, "fill", n_iter, repeat, || {
        let c: Tensor<f64, _, _> = rt::full(([n, n], 2.5, &d));
        std::hint::black_box(&c);
    });

    // add contiguous
    let a = rt::arange((0., (n * n) as f64, 1., &d)).into_shape([n, n]);
    let b = rt::arange((0., (n * n) as f64, 1., &d)).into_shape([n, n]);
    let dt_add = bench_cell(&d, "add", n_iter, repeat, || {
        let s = rt::add(&a, &b);
        std::hint::black_box(&s);
    });

    // add B.T
    let bv = b.view();
    let bt = bv.swapaxes(0, 1);
    let dt_add_t = bench_cell(&d, "add_t", n_iter, repeat, || {
        let s = rt::add(&a, &bt);
        std::hint::black_box(&s);
    });

    // add A.T (same buffer both sides)
    let av = a.view();
    let at = av.swapaxes(0, 1);
    bench_cell(&d, "add_t_self", n_iter, repeat, || {
        let s = rt::add(&a, &at);
        std::hint::black_box(&s);
    });

    // 3-D swapaxes(0,1) and broadcasts share the [512,512,512] arena
    let m = 512usize; // 1 GiB f64 per 3-D operand
    {
        let a3 = rt::arange((0., (m * m * m) as f64, 1., &d)).into_shape([m, m, m]);
        let a3v = a3.view();
        let a3t = a3v.swapaxes(0, 1);
        bench_cell(&d, "add_swap3d", n_iter, repeat, || {
            let s = rt::add(&a3, &a3t);
            std::hint::black_box(&s);
        });

        // broadcast B[:,None,:]: contiguous [512,1,512] operand
        let brow = rt::arange((0., (m * m) as f64, 1., &d)).into_shape([m, 1, m]);
        bench_cell(&d, "add_bcast_r", n_iter, repeat, || {
            let s = rt::add(&a3, &brow);
            std::hint::black_box(&s);
        });

        // broadcast B[:,:,None]: contiguous [512,512,1] operand
        let bcol = rt::arange((0., (m * m) as f64, 1., &d)).into_shape([m, m, 1]);
        bench_cell(&d, "add_bcast_c", n_iter, repeat, || {
            let s = rt::add(&a3, &bcol);
            std::hint::black_box(&s);
        });
    }

    // outer a[None,:] + b[:,None] -> [2048,2048]
    {
        let va = rt::arange((0., n as f64, 1., &d));
        let vb = rt::arange((0., n as f64, 1., &d));
        let var = va.view().into_shape([1, n]);
        let vbc = vb.view().into_shape([n, 1]);
        bench_cell(&d, "add_outer", n_iter, repeat, || {
            let s = rt::add(&var, &vbc);
            std::hint::black_box(&s);
        });
    }

    // sliced windows on [2048,2048]
    let avw = a.view();
    let aw = avw.i((.., 10..1010));
    let bvw = b.view();
    let bw = bvw.i((.., 50..1050));
    bench_cell(&d, "add_slice", n_iter, repeat, || {
        let s = rt::add(&aw, &bw);
        std::hint::black_box(&s);
    });

    // stepped B[:, ::2] against A[:, :1024]
    let avh = a.view();
    let ash = avh.i((.., ..(n / 2)));
    let bvs = b.view();
    let bs = bvs.i((.., slice!(None, None, 2)));
    bench_cell(&d, "add_step", n_iter, repeat, || {
        let s = rt::add(&ash, &bs);
        std::hint::black_box(&s);
    });

    // flipped B[:, :1024]
    let fl_all = rt::flip(&b, 1);
    let flh = fl_all.i((.., ..(n / 2)));
    bench_cell(&d, "add_flip", n_iter, repeat, || {
        let s = rt::add(&ash, &flh);
        std::hint::black_box(&s);
    });

    // where
    let mask = rt::arange((0, (n * n) as i32, 1, &d)).into_shape([n, n]);
    let maskb = rt::ge(&mask, &rt::full(([n, n], (n * n / 2) as i32, &d)));
    bench_cell(&d, "where_tt", n_iter, repeat, || {
        let s = rt::r#where(&maskb, &a, &b);
        std::hint::black_box(&s);
    });
    bench_cell(&d, "where_ts", n_iter, repeat, || {
        let s = rt::r#where(&maskb, &a, 0.5);
        std::hint::black_box(&s);
    });

    // clip scalar bounds
    bench_cell(&d, "clip", n_iter, repeat, || {
        let s = rt::clip(&a, (1000.0, 3.0e6));
        std::hint::black_box(&s);
    });

    // rem (per-element branch)
    let dt_rem = bench_cell(&d, "rem", n_iter, repeat, || {
        let s = rt::rem(&a, &b);
        std::hint::black_box(&s);
    });

    // transcendental
    let dt_sin = bench_cell(&d, "sin", n_iter, repeat, || {
        let s = rt::sin(&a);
        std::hint::black_box(&s);
    });

    // pow
    bench_cell(&d, "pow", n_iter, repeat, || {
        let s = rt::pow(&a, 2.5);
        std::hint::black_box(&s);
    });

    // strided-output assign: C.T = A
    {
        let mut dst = rt::arange((0., (n * n) as f64, 1., &d)).into_shape([n, n]);
        let dstm = dst.view_mut();
        let mut dtv = dstm.into_permute_dims((1, 0));
        bench_cell(&d, "copy_T", n_iter, repeat, || {
            dtv.assign(&a);
        });
    }

    // to_contig of a transposed view (copy-on-write path)
    {
        let av2 = a.view();
        let src = av2.swapaxes(0, 1);
        bench_cell(&d, "to_contig", n_iter, repeat, || {
            let c = src.to_contig(RowMajor);
            std::hint::black_box(&c);
        });
    }

    let _ = (dt_fill, dt_add, dt_add_t, dt_rem, dt_sin);
    funcs::synchronize(&d).unwrap();
}
