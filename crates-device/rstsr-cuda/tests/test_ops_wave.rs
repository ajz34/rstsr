//! New elementwise families (2026-10-23 wave): rem, bitwise, compound
//! assigns, promoting binaries, non-transcendental unaries, complex
//! transcendentals, where/clip — all against DeviceCpuSerial parity.

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

/// Layout-ordered host values of a tensor (D2H through the trusted CPU
/// device, so strided views gather correctly).
fn host<R, T, B, D>(t: &TensorAny<R, T, B, D>) -> Vec<T>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    T: Clone + Send + Sync + 'static,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToCpuAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    let raw = t.to_raw_f().rstsr_unwrap();
    DeviceToCpuAPI::<T>::to_cpu_vec::<DataOwned<<B as DeviceRawAPI<T>>::Raw>>(&Storage::new(
        DataOwned::from(raw),
        t.device().clone(),
    ))
    .rstsr_unwrap()
}

/// Layout-ordered host values of an f64 tensor (N-d safe).
fn host_nd<'a, Tr, B, D>(t: &TensorAny<Tr, f64, B, D>) -> Vec<f64>
where
    Tr: DataAPI<Data = <B as DeviceRawAPI<f64>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceToCpuAPI<f64> + DeviceCreationAnyAPI<f64> + OpAssignArbitaryAPI<f64, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<f64>::to_cpu_vec(contig.storage()).rstsr_unwrap()
}

macro_rules! require_cuda {
    () => {
        if !funcs::is_available() {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

#[test]
fn new_ops_rem_bitwise_assign() {
    require_cuda!();
    let dc = DeviceCpuSerial::default();
    let dg = DeviceCuda::new(0);
    let n = 1000usize;

    // rem f64 incl. negative dividends (NumPy sign-fixing remainder)
    let ag = rt::linspace((-10.0, 10.0, n, &dg));
    let bg = rt::linspace((0.5, 3.5, n, &dg));
    let ac = rt::linspace((-10.0, 10.0, n, &dc));
    let bc = rt::linspace((0.5, 3.5, n, &dc));
    let rg = host(&rt::rem(&ag, &bg));
    let rc = host(&rt::rem(&ac, &bc));
    let md = rg.iter().zip(&rc).map(|(x, y): (&f64, &f64)| (x - y).abs()).fold(0.0f64, f64::max);
    assert!(md < 1e-12, "rem f64 max_diff = {md:e}");

    // rem i32 with negative dividends
    let ig = rt::arange((-20i32, 20, 1, &dg)).into_shape([8, 5]);
    let ig2 = rt::full(([8, 5], 3i32, &dg));
    let ic = rt::arange((-20i32, 20, 1, &dc)).into_shape([8, 5]);
    let ic2 = rt::full(([8, 5], 3i32, &dc));
    let hg: Vec<i32> = host(&rt::rem(&ig, &ig2).into_shape([40]));
    let hc: Vec<i32> = host(&rt::rem(&ic, &ic2).into_shape([40]));
    assert_eq!(hg, hc);

    // bitwise families on i64
    let xg = rt::arange((0i64, 1000, 1, &dg));
    let yg = rt::arange((0i64, 1000, 1, &dg)).mul(3);
    let xc = rt::arange((0i64, 1000, 1, &dc));
    let yc = rt::arange((0i64, 1000, 1, &dc)).mul(3);
    assert_eq!(host(&rt::bitxor(&xg, &yg)), host(&rt::bitxor(&xc, &yc)));
    assert_eq!(host(&rt::bitand(&xg, &yg)), host(&rt::bitand(&xc, &yc)));
    assert_eq!(host(&rt::bitor(&xg, &yg)), host(&rt::bitor(&xc, &yc)));
    let sh2g: Tensor<i64, _, _> = rt::full(([1000], 2i64, &dg));
    let sh2c: Tensor<i64, _, _> = rt::full(([1000], 2i64, &dc));
    assert_eq!(host(&rt::shl(&xg, &sh2g)), host(&rt::shl(&xc, &sh2c)));
    assert_eq!(host(&rt::shr(&xg, &sh2g)), host(&rt::shr(&xc, &sh2c)));

    // compound assign with transposed RHS (write-ordering discipline)
    let owng = rt::arange((0.0f64, 4096.0, 1.0, &dg)).into_shape([64, 64]);
    let viewg = owng.view();
    let srcg = viewg.swapaxes(0, 1);
    let mut mg = rt::arange((0.0f64, 4096.0, 1.0, &dg)).into_shape([64, 64]);
    rt::add_assign(&mut mg, &srcg);
    let ownc = rt::arange((0.0f64, 4096.0, 1.0, &dc)).into_shape([64, 64]);
    let viewc = ownc.view();
    let srcc = viewc.swapaxes(0, 1);
    let mut mc = rt::arange((0.0f64, 4096.0, 1.0, &dc)).into_shape([64, 64]);
    rt::add_assign(&mut mc, &srcc);
    assert_eq!(host_nd(&mg), host_nd(&mc));

    // scalar compound: m *= 2.5 (owned scalar form); the strided-RHS
    // tensor form is covered by the transposed add_assign above
    let mut mg2 = rt::arange((0.0f64, 100.0, 1.0, &dg)).into_shape([10, 10]);
    rt::mul_assign(&mut mg2, 2.5);
    let mut mc2 = rt::arange((0.0f64, 100.0, 1.0, &dc)).into_shape([10, 10]);
    rt::mul_assign(&mut mc2, 2.5);
    assert_eq!(host_nd(&mg2), host_nd(&mc2));
}

#[test]
fn new_ops_promoting_binaries() {
    require_cuda!();
    let dc = DeviceCpuSerial::default();
    let dg = DeviceCuda::new(0);
    let n = 1000usize;

    // pow
    let pg = rt::linspace((0.0, 4.0, n, &dg));
    let eg = rt::linspace((0.0, 2.0, n, &dg));
    let pc = rt::linspace((0.0, 4.0, n, &dc));
    let ec = rt::linspace((0.0, 2.0, n, &dc));
    let pgv: Vec<f64> = host(&rt::pow(&pg, &eg));
    let pcv: Vec<f64> = host(&rt::pow(&pc, &ec));
    let md = pgv.iter().zip(&pcv).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
    assert!(md < 1e-12, "pow max_diff = {md:e}");

    // atan2/hypot/floor_divide/max/min
    let v1g = rt::linspace((-3.0, 3.0, n, &dg));
    let v2g = rt::linspace((0.5, 2.0, n, &dg));
    let v1c = rt::linspace((-3.0, 3.0, n, &dc));
    let v2c = rt::linspace((0.5, 2.0, n, &dc));
    let pairs = [
        ("atan2", host(&rt::atan2(&v1g, &v2g)), host(&rt::atan2(&v1c, &v2c))),
        ("hypot", host(&rt::hypot(&v1g, &v2g)), host(&rt::hypot(&v1c, &v2c))),
        // floordiv is discontinuous at integer ratios and the two linspace
        // evaluations may differ by 1 ulp — compare on dyadic inputs instead
        (
            "floordiv",
            host(&rt::floor_divide(
                &rt::asarray((vec![-3.0f64, -1.5, 0.5, 2.0], &dg)),
                &rt::asarray((vec![0.5f64, 2.0, -1.5, 0.5], &dg)),
            )),
            host(&rt::floor_divide(
                &rt::asarray((vec![-3.0f64, -1.5, 0.5, 2.0], &dc)),
                &rt::asarray((vec![0.5f64, 2.0, -1.5, 0.5], &dc)),
            )),
        ),
        ("max", host(&rt::maximum(&v1g, &v2g)), host(&rt::maximum(&v1c, &v2c))),
        ("min", host(&rt::minimum(&v1g, &v2g)), host(&rt::minimum(&v1c, &v2c))),
    ];
    for (name, hg, hc) in pairs {
        let md = hg.iter().zip(&hc).map(|(x, y): (&f64, &f64)| (x - y).abs()).fold(0.0f64, f64::max);
        assert!(md < 1e-12, "{name} max_diff = {md:e}");
    }

    // NaN propagation in maximum/minimum (CPU ext_max semantics)
    let nang = rt::asarray((vec![1.0f64, f64::NAN, 3.0], &dg));
    let other = rt::asarray((vec![2.0f64, 4.0, f64::NAN], &dg));
    let mg = host(&rt::maximum(&nang, &other));
    assert!(mg[1].is_nan() && mg[2].is_nan() && mg[0] == 2.0);

    // floor_divide int signs
    let ig = rt::arange((-7i64, 8, 1, &dg));
    let ig2: Tensor<i64, _, _> = rt::full(([15], 3i64, &dg));
    let ic = rt::arange((-7i64, 8, 1, &dc));
    let ic2: Tensor<i64, _, _> = rt::full(([15], 3i64, &dc));
    assert_eq!(host(&rt::floor_divide(&ig, &ig2)), host(&rt::floor_divide(&ic, &ic2)));
}

#[test]
fn new_ops_unary_semantics() {
    require_cuda!();
    let dc = DeviceCpuSerial::default();
    let dg = DeviceCuda::new(0);

    let data = vec![1.0f64, -2.0, f64::NAN, 3.5, -0.0, 2.5, f64::INFINITY, -f64::INFINITY];
    let vg = rt::asarray((data.clone(), &dg));
    let vc = rt::asarray((data.clone(), &dc));

    // sign: NaN->NaN, ±inf->±1, ±0->0
    let sg = host(&rt::sign(&vg));
    let sc = host(&rt::sign(&vc));
    for (x, y) in sg.iter().zip(&sc) {
        assert!((x.is_nan() && y.is_nan()) || x == y, "sign {x} vs {y}");
    }

    let absg = host(&rt::abs(&vg));
    let absc = host(&rt::abs(&vc));
    for (x, y) in absg.iter().zip(&absc) {
        assert!((x.is_nan() && y.is_nan()) || x == y, "abs {x} vs {y}");
    }
    for (xa, ya, name) in [
        (host(&rt::round(&vg)), host(&rt::round(&vc)), "round"),
        (host(&rt::floor(&vg)), host(&rt::floor(&vc)), "floor"),
        (host(&rt::ceil(&vg)), host(&rt::ceil(&vc)), "ceil"),
        (host(&rt::trunc(&vg)), host(&rt::trunc(&vc)), "trunc"),
        (host(&rt::square(&vg)), host(&rt::square(&vc)), "square"),
    ] {
        for (x, y) in xa.iter().zip(&ya) {
            assert!((x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits(), "{name} {x} vs {y}");
        }
    }
    assert_eq!(host(&rt::is_nan(&vg)), host(&rt::is_nan(&vc)));
    assert_eq!(host(&rt::is_inf(&vg)), host(&rt::is_inf(&vc)));
    assert_eq!(host(&rt::is_finite(&vg)), host(&rt::is_finite(&vc)));
    assert_eq!(host(&rt::signbit(&vg)), host(&rt::signbit(&vc)));

    // ext_neg on unsigned (two's complement wrap)
    let ug = rt::asarray((vec![3u32, 0, 5], &dg));
    let uc = rt::asarray((vec![3u32, 0, 5], &dc));
    assert_eq!(host(&rt::ext_neg(&ug)), host(&rt::ext_neg(&uc)));

    // not on bool and i32
    let bg = rt::asarray((vec![true, false, true], &dg));
    let bc = rt::asarray((vec![true, false, true], &dc));
    assert_eq!(host(&rt::not(&bg)), host(&rt::not(&bc)));
    let ig = rt::asarray((vec![5i32, -1, 0], &dg));
    let ic = rt::asarray((vec![5i32, -1, 0], &dc));
    assert_eq!(host(&rt::not(&ig)), host(&rt::not(&ic)));

    // complex component ops
    use num::Complex;
    let cz: Vec<Complex<f64>> = (0..6).map(|i| Complex::new(i as f64 - 2.5, (i as f64 - 2.0) * 0.5)).collect();
    let zg = rt::asarray((cz.clone(), &dg));
    let zc = rt::asarray((cz.clone(), &dc));
    assert_eq!(host(&rt::real(&zg)), host(&rt::real(&zc)));
    assert_eq!(host(&rt::imag(&zg)), host(&rt::imag(&zc)));
    assert_eq!(host(&rt::conj(&zg)), host(&rt::conj(&zc)));
    let ag = host(&rt::abs(&zg));
    let ac = host(&rt::abs(&zc));
    for (x, y) in ag.iter().zip(&ac) {
        assert!((x - y).abs() < 1e-12, "cabs {x} vs {y}");
    }

    // complex sign (z/|z|, 0 -> 0) and complex pow: the template-header
    // definitions (previously wired but undefined — latent NVRTC failure)
    let sg = host(&rt::sign(&zg));
    let sc = host(&rt::sign(&zc));
    for (x, y) in sg.iter().zip(&sc) {
        let rel = (x - y).norm();
        assert!(rel < 1e-12, "csign {x} vs {y}");
    }
    let pw_g = host(&rt::pow(&zg, &zg));
    let pw_c = host(&rt::pow(&zc, &zc));
    for (x, y) in pw_g.iter().zip(&pw_c) {
        let scale = if x.norm() > 1.0 { x.norm() } else { 1.0 };
        assert!((x - y).norm() / scale < 1e-10, "cpow {x} vs {y}");
    }
    // complex round: declined at the tensor tier on CPU (ExtReal-only);
    // the device round_even path supports it but rt::round does not bind
    // for complex — nothing to compare against, skip.
}

#[test]
fn new_ops_complex_transcendentals() {
    require_cuda!();
    let dc = DeviceCpuSerial::default();
    let dg = DeviceCuda::new(0);
    use num::Complex;

    // quadrant grid + near-branch-cut points
    let cz: Vec<Complex<f64>> = (-6..6).map(|i| Complex::new(i as f64 * 0.7 - 0.3, i as f64 * -0.5 + 0.2)).collect();
    let zg = rt::asarray((cz.clone(), &dg));
    let zc = rt::asarray((cz.clone(), &dc));

    type CPairs = Vec<(&'static str, Vec<Complex<f64>>, Vec<Complex<f64>>)>;
    let pairs: CPairs = vec![
        ("sin", host(&rt::sin(&zg)), host(&rt::sin(&zc))),
        ("cos", host(&rt::cos(&zg)), host(&rt::cos(&zc))),
        ("tan", host(&rt::tan(&zg)), host(&rt::tan(&zc))),
        ("sinh", host(&rt::sinh(&zg)), host(&rt::sinh(&zc))),
        ("cosh", host(&rt::cosh(&zg)), host(&rt::cosh(&zc))),
        ("tanh", host(&rt::tanh(&zg)), host(&rt::tanh(&zc))),
        ("asinh", host(&rt::asinh(&zg)), host(&rt::asinh(&zc))),
        ("acosh", host(&rt::acosh(&zg)), host(&rt::acosh(&zc))),
        ("exp", host(&rt::exp(&zg)), host(&rt::exp(&zc))),
        ("expm1", host(&rt::expm1(&zg)), host(&rt::expm1(&zc))),
        ("log", host(&rt::log(&zg)), host(&rt::log(&zc))),
        ("log1p", host(&rt::log1p(&zg)), host(&rt::log1p(&zc))),
        ("log2", host(&rt::log2(&zg)), host(&rt::log2(&zc))),
        ("log10", host(&rt::log10(&zg)), host(&rt::log10(&zc))),
        ("sqrt", host(&rt::sqrt(&zg)), host(&rt::sqrt(&zc))),
        ("inv", host(&rt::inv(&zg)), host(&rt::inv(&zc))),
        ("atan", host(&rt::atan(&zg)), host(&rt::atan(&zc))),
    ];
    for (name, hg, hc) in pairs {
        // real transcendentals differ CPU(Rust libm) vs GPU(libdevice) at
        // ~1 ulp; complex formulas amplify to ~1e-12 relative on this grid
        let md: f64 = hg
            .iter()
            .zip(&hc)
            .map(|(x, y)| if x.norm() > 1.0 { (x - y).norm() / x.norm() } else { (x - y).norm() })
            .fold(0.0f64, f64::max);
        assert!(md < 1e-10, "c{name} max_rel_diff = {md:e}");
    }
}

#[test]
fn new_ops_layout_matrix() {
    require_cuda!();
    let dc = DeviceCpuSerial::default();
    let dg = DeviceCuda::new(0);

    // operand layout classes for binary ops (same logical shape [10,10,30]):
    // contiguous, transposed, axis-sliced (offset + non-full extent),
    // stepped slice, reversed (negative stride), broadcast
    let base = rt::arange((0.0f64, 3000.0, 1.0, &dg)).into_shape([10, 10, 30]);
    let basec = rt::arange((0.0f64, 3000.0, 1.0, &dc)).into_shape([10, 10, 30]);

    let va = base.view();
    let vac = basec.view();

    // contig + transposed: buffer [30,10,10] swapped on axes (0,2) views as
    // [10,10,30] - same shape as va, fully different strides
    let cube = rt::arange((0.0f64, 3000.0, 1.0, &dg)).into_shape([30, 10, 10]);
    let cubec = rt::arange((0.0f64, 3000.0, 1.0, &dc)).into_shape([30, 10, 10]);
    let cv = cube.view();
    let cvc = cubec.view();
    let cb = cv.swapaxes(0, 2);
    let cbc = cvc.swapaxes(0, 2);
    assert_eq!(host_nd(&rt::add(&va, &cb)), host_nd(&rt::add(&vac, &cbc)));

    // sliced operands: axis-2 windows with offset + partial extent
    // ([10,10,24] outputs, base strides differ between va and cb)
    let sa = va.i((.., .., 3..27));
    let sb = cb.i((.., .., 3..27));
    let sac = vac.i((.., .., 3..27));
    let sbc = cbc.i((.., .., 3..27));
    assert_eq!(host_nd(&rt::add(&sa, &sb)), host_nd(&rt::add(&sac, &sbc)));

    // stepped first operand + flipped (negative-stride) second operand,
    // both views of the same [10,10,30] shape
    let st = va.i((.., slice!(None, None, 2), ..));
    let stc = vac.i((.., slice!(None, None, 2), ..));
    let fl = rt::flip(&st, 2);
    let flc = rt::flip(&stc, 2);
    assert_eq!(host_nd(&rt::add(&st, &fl)), host_nd(&rt::add(&stc, &flc)));

    // broadcast row pattern: A[1,10,30-slice] + B[10,1]
    let row = va.i((5, .., 7..25));
    let pat = vac.i((5, .., 7..25));
    let bg = rt::asarray((vec![1.5f64, -2.0, 3.25, 0.5], &dg)).into_shape([4, 1]);
    let bcst = rt::asarray((vec![1.5f64, -2.0, 3.25, 0.5], &dc)).into_shape([4, 1]);
    // row is [10, 18]; broadcast bg [4,1] over trailing-slice rows
    let row4 = row.i((..4, ..));
    let pat4 = pat.i((..4, ..));
    let r = rt::add(&row4, &bg);
    let rc = rt::add(&pat4, &bcst);
    assert_eq!(host_nd(&r), host_nd(&rc));

    // where with tensor arms on strided layouts (all share [10,10,30];
    // the last operand is a transposed view of a [30,10,10] buffer)
    let cond = rt::arange((0, 3000, 1, &dg)).into_shape([10, 10, 30]);
    let condc = rt::arange((0, 3000, 1, &dc)).into_shape([10, 10, 30]);
    let cg = rt::ge(&cond, &rt::full(([10, 10, 30], 1500.0f64, &dg)));
    let cc = rt::ge(&condc, &rt::full(([10, 10, 30], 1500.0f64, &dc)));
    let posg: Tensor<f64, _, _> = rt::full(([30, 10, 10], 2.5f64, &dg));
    let posc: Tensor<f64, _, _> = rt::full(([30, 10, 10], 2.5f64, &dc));
    let negg: Tensor<f64, _, _> = rt::full(([30, 10, 10], -1.0f64, &dg));
    let negc: Tensor<f64, _, _> = rt::full(([30, 10, 10], -1.0f64, &dc));
    let pgv = posg.view();
    let pcv = posc.view();
    let ngv = negg.view();
    let ncv = negc.view();
    let pg = pgv.swapaxes(0, 2);
    let pc = pcv.swapaxes(0, 2);
    let ng = ngv.swapaxes(0, 2);
    let nc = ncv.swapaxes(0, 2);
    let wg = rt::r#where(&cg, &pg, &ng);
    let wc = rt::r#where(&cc, &pc, &nc);
    assert_eq!(host_nd(&wg), host_nd(&wc));

    // clip with scalar bounds on a transposed view
    let clipped = rt::clip(&cb, (-5.0, 12.5));
    let clippedc = rt::clip(&cbc, (-5.0, 12.5));
    assert_eq!(host_nd(&clipped), host_nd(&clippedc));
}

#[test]
fn new_ops_transpose_copy_tiled() {
    // the tiled 2-D transpose fast path (to_contig of an axis-swapped view,
    // and assign into a transposed view) vs CPU parity, above and below the
    // 4096-element kernel threshold, incl. non-multiple-of-32 margins
    let d = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    for (m, n) in [(64usize, 64usize), (63, 67), (128, 100), (2048, 3500)] {
        let base = rt::arange((0.0f64, (m * n) as f64, 1.0, &d)).into_shape([m, n]);
        let basec = rt::arange((0.0f64, (m * n) as f64, 1.0, &dc)).into_shape([m, n]);
        let bv = base.view();
        let bvc = basec.view();
        let bt = bv.swapaxes(0, 1);
        let btc = bvc.swapaxes(0, 1);
        // transposed-read direction (to_contig of a swapped view)
        let got = bt.to_contig(RowMajor).into_owned();
        let want = btc.to_contig(RowMajor).into_owned();
        assert_eq!(host_nd(&got), host_nd(&want), "T-read {m}x{n}");

        // transposed-write direction (assign into a swapped view)
        let mut dst: Tensor<f64, _, _> = rt::zeros(([n, m], &d));
        let dstv = dst.view_mut();
        let mut dtv = dstv.into_permute_dims((1, 0));
        dtv.assign(&base);
        let mut dstc: Tensor<f64, _, _> = rt::zeros(([n, m], &dc));
        let dstcv = dstc.view_mut();
        let mut dtvc = dstcv.into_permute_dims((1, 0));
        dtvc.assign(&basec);
        assert_eq!(host_nd(&dst), host_nd(&dstc), "T-write {m}x{n}");
    }

    // integer dtype rides the same tile kernel
    let mi = rt::arange((0i64, 128 * 128, 1, &d)).into_shape([128, 128]);
    let mic = rt::arange((0i64, 128 * 128, 1, &dc)).into_shape([128, 128]);
    let miv = mi.view();
    let micv = mic.view();
    let mit = miv.swapaxes(0, 1);
    let mitc = micv.swapaxes(0, 1);
    let got = mit.to_contig(RowMajor).into_owned();
    let want = mitc.to_contig(RowMajor).into_owned();
    let g = host_nd_i64(&got);
    let wnt = host_nd_i64(&want);
    assert_eq!(g, wnt, "i64 T-read");
}

fn host_nd_i64<D, B>(t: &Tensor<i64, B, D>) -> Vec<i64>
where
    D: DimAPI,
    B: DeviceAPI<i64> + DeviceToCpuAPI<i64> + DeviceCreationAnyAPI<i64> + OpAssignArbitaryAPI<i64, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<i64>::to_cpu_vec(contig.storage()).rstsr_unwrap()
}
