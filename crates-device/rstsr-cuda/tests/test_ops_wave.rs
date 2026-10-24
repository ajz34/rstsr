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
    // and assign into a transposed view) vs CPU parity at and above the
    // 4096-element threshold, incl. non-multiple-of-32 margins. The read
    // direction reaches the tile when the swapped shape has cols >= rows;
    // the write direction reaches it for every shape.
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    for (m, n) in [(64usize, 64usize), (65, 63), (63, 67), (128, 100), (2048, 3500)] {
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

/// Component-wise complex comparison: both NaN, exact equality (covers
/// infinities), or relative closeness at `tol`.
fn assert_c64(name: &str, got: &[num::Complex<f64>], want: &[num::Complex<f64>], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let close = |x: f64, y: f64| (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= tol * y.abs().max(1.0);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(close(a.re, b.re) && close(a.im, b.im), "{name}[{i}]: {a} vs {b}");
    }
}

/// Contiguous scalar/tensor compound assigns and both scalar `where` arms
/// exercise the ILP-4 looping entries (n spans several grid-stride groups).
#[test]
fn regression_launcher_geometry() {
    require_cuda!();
    let dc = DeviceCpuSerial::default();
    let dg = DeviceCuda::new(0);
    let n = 4097usize;

    macro_rules! check_assign {
        ($op:ident) => {{
            let mut g = rt::linspace((0.25, 9.75, n, &dg));
            let mut c = rt::linspace((0.25, 9.75, n, &dc));
            rt::$op(&mut g, 1.5f64);
            rt::$op(&mut c, 1.5f64);
            assert_eq!(host(&g), host(&c), stringify!($op));
        }};
    }
    check_assign!(add_assign);
    check_assign!(sub_assign);
    check_assign!(mul_assign);
    check_assign!(div_assign);
    check_assign!(rem_assign);

    macro_rules! check_iassign {
        ($op:ident) => {{
            let mut g = rt::arange((0i64, n as i64, 1, &dg));
            let mut c = rt::arange((0i64, n as i64, 1, &dc));
            rt::$op(&mut g, 3i64);
            rt::$op(&mut c, 3i64);
            assert_eq!(host(&g), host(&c), stringify!($op));
        }};
    }
    check_iassign!(bitand_assign);
    check_iassign!(bitor_assign);
    check_iassign!(bitxor_assign);
    check_iassign!(shl_assign);
    check_iassign!(shr_assign);

    // tensor-RHS compound assign, contiguous
    let rg = rt::linspace((0.5, 3.5, n, &dg));
    let rc = rt::linspace((0.5, 3.5, n, &dc));
    let mut g = rt::linspace((0.25, 9.75, n, &dg));
    let mut c = rt::linspace((0.25, 9.75, n, &dc));
    rt::add_assign(&mut g, &rg);
    rt::add_assign(&mut c, &rc);
    assert_eq!(host(&g), host(&c), "add_assign tensor");

    // where with a scalar arm on either side, contiguous
    let cond = rt::ge(&rt::arange((0i32, n as i32, 1, &dg)), &rt::full(([n], (n / 2) as i32, &dg)));
    let condc = rt::ge(&rt::arange((0i32, n as i32, 1, &dc)), &rt::full(([n], (n / 2) as i32, &dc)));
    let xg = rt::linspace((0.0, 1.0, n, &dg));
    let xc = rt::linspace((0.0, 1.0, n, &dc));
    assert_eq!(host(&rt::r#where(&cond, &xg, 2.5f64)), host(&rt::r#where(&condc, &xc, 2.5f64)), "where scalar-c");
    assert_eq!(host(&rt::r#where(&cond, 2.5f64, &xg)), host(&rt::r#where(&condc, 2.5f64, &xc)), "where scalar-b");
}

/// Rank-5 (Generic-plan) fill must reach the rank-generic kernel instead of
/// recursing through the planned path.
#[test]
fn regression_fill_generic_rank5() {
    require_cuda!();
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    let mut g = rt::arange((0.0f64, 32.0, 1.0, &dg)).into_shape([2, 2, 2, 2, 2]).into_permute_dims((4, 3, 2, 1, 0));
    let mut c = rt::arange((0.0f64, 32.0, 1.0, &dc)).into_shape([2, 2, 2, 2, 2]).into_permute_dims((4, 3, 2, 1, 0));
    g.fill(2.5);
    c.fill(2.5);
    assert_eq!(host_nd(&g), host_nd(&c), "rank-5 permuted fill");
    let g5: Tensor<f64, _, _> = rt::full(([2, 2, 2, 2, 2], 1.25, &dg));
    let c5: Tensor<f64, _, _> = rt::full(([2, 2, 2, 2, 2], 1.25, &dc));
    assert_eq!(host_nd(&g5), host_nd(&c5), "rank-5 full");
}

/// `where(cond, scalar, tensor)` with sliced / stepped / flipped conditions:
/// the inversion is in-kernel, so no offset-less scratch is written.
#[test]
fn regression_where_strided_condition() {
    require_cuda!();
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    let base_g = rt::arange((0i32, 4096i32, 1, &dg)).into_shape([64, 64]);
    let base_c = rt::arange((0i32, 4096i32, 1, &dc)).into_shape([64, 64]);
    let cond_g = rt::ge(&base_g, &rt::full(([64, 64], 2048i32, &dg)));
    let cond_c = rt::ge(&base_c, &rt::full(([64, 64], 2048i32, &dc)));

    // offset window [64, 61]
    let y_g = rt::linspace((0.0f64, 1.0, 64 * 61, &dg)).into_shape([64, 61]);
    let y_c = rt::linspace((0.0f64, 1.0, 64 * 61, &dc)).into_shape([64, 61]);
    let cgv = cond_g.view();
    let ccv = cond_c.view();
    let cgs = cgv.i((.., 3..));
    let ccs = ccv.i((.., 3..));
    assert_eq!(host_nd(&rt::r#where(&cgs, 7.5f64, &y_g)), host_nd(&rt::r#where(&ccs, 7.5f64, &y_c)), "sliced cond");

    // stepped condition [64, 32]
    let y2_g = rt::linspace((0.0f64, 1.0, 64 * 32, &dg)).into_shape([64, 32]);
    let y2_c = rt::linspace((0.0f64, 1.0, 64 * 32, &dc)).into_shape([64, 32]);
    let cg_step = cgv.i((.., slice!(None, None, 2)));
    let cc_step = ccv.i((.., slice!(None, None, 2)));
    assert_eq!(
        host_nd(&rt::r#where(&cg_step, 7.5f64, &y2_g)),
        host_nd(&rt::r#where(&cc_step, 7.5f64, &y2_c)),
        "stepped cond"
    );

    // negative-stride condition [64, 64]
    let fl_g = rt::flip(&cond_g, 1);
    let fl_c = rt::flip(&cond_c, 1);
    let y3_g = rt::linspace((0.0f64, 1.0, 64 * 64, &dg)).into_shape([64, 64]);
    let y3_c = rt::linspace((0.0f64, 1.0, 64 * 64, &dc)).into_shape([64, 64]);
    assert_eq!(
        host_nd(&rt::r#where(&fl_g, 7.5f64, &y3_g)),
        host_nd(&rt::r#where(&fl_c, 7.5f64, &y3_c)),
        "flipped cond"
    );

    // scalar on the other arm (inverted condition) over the flipped condition
    assert_eq!(
        host_nd(&rt::r#where(&fl_g, &y3_g, 7.5f64)),
        host_nd(&rt::r#where(&fl_c, &y3_c, 7.5f64)),
        "flipped cond, scalar b"
    );
}

/// Clip NaN propagation, integer bounds, tensor bounds, `None`, and f32
/// literal inference.
#[test]
fn regression_clip_paths() {
    require_cuda!();
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();

    // an input NaN survives non-NaN scalar bounds
    let data = vec![f64::NAN, -1.0, 0.5, 2.0];
    let g = rt::asarray((data.clone(), &dg));
    let c = rt::asarray((data.clone(), &dc));
    let hg: Vec<f64> = host(&rt::clip(&g, (0.0, 1.0)));
    let hc: Vec<f64> = host(&rt::clip(&c, (0.0, 1.0)));
    for (x, y) in hg.iter().zip(&hc) {
        assert!((x.is_nan() && y.is_nan()) || x == y, "clip nan: {x} vs {y}");
    }

    // integer scalar bounds
    let ig = rt::asarray((vec![1i64, 5, 9, -4], &dg));
    let ic = rt::asarray((vec![1i64, 5, 9, -4], &dc));
    assert_eq!(host(&rt::clip(&ig, (2i64, 5i64))), host(&rt::clip(&ic, (2i64, 5i64))), "int scalar clip");

    // integer tensor bound + `None`
    let lo = rt::asarray((vec![0i32, 3, 8, 0], &dg));
    let loc = rt::asarray((vec![0i32, 3, 8, 0], &dc));
    let xg = rt::asarray((vec![5i32, 1, 10, -2], &dg));
    let xc = rt::asarray((vec![5i32, 1, 10, -2], &dc));
    assert_eq!(host(&rt::clip(&xg, (&lo, None))), host(&rt::clip(&xc, (&loc, None))), "int tensor clip");

    // f32 with unsuffixed float literals picks up the tensor dtype (the CPU
    // device would promote the literal to f64 and return f64)
    let fg: Tensor<f32, _, _> = rt::asarray((vec![1.0f32, 2.0, 3.0], &dg));
    let fc: Tensor<f32, _, _> = rt::asarray((vec![1.0f32, 2.0, 3.0], &dc));
    assert_eq!(host(&rt::clip(&fg, (0.0, 1.0))), host(&rt::clip(&fc, (0.0f32, 1.0f32))), "f32 clip");
}

/// f32/c32 query unaries and f32 `sign`/`abs` (the f-suffixed NVRTC names do
/// not exist).
#[test]
fn regression_query_and_unary_f32() {
    require_cuda!();
    use num::Complex;
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    let data = vec![1.0f32, -2.0, f32::NAN, f32::INFINITY, -f32::INFINITY, -0.0];
    let g = rt::asarray((data.clone(), &dg));
    let c = rt::asarray((data.clone(), &dc));
    assert_eq!(host(&rt::is_nan(&g)), host(&rt::is_nan(&c)), "f32 is_nan");
    assert_eq!(host(&rt::is_inf(&g)), host(&rt::is_inf(&c)), "f32 is_inf");
    assert_eq!(host(&rt::is_finite(&g)), host(&rt::is_finite(&c)), "f32 is_finite");
    for (x, y) in host(&rt::sign(&g)).iter().zip(&host(&rt::sign(&c))) {
        assert!((x.is_nan() && y.is_nan()) || x == y, "f32 sign {x} vs {y}");
    }
    for (x, y) in host(&rt::abs(&g)).iter().zip(&host(&rt::abs(&c))) {
        assert!((x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits(), "f32 abs {x} vs {y}");
    }

    let z: Vec<Complex<f32>> =
        vec![Complex::new(1.0, 2.0), Complex::new(f32::NAN, 0.0), Complex::new(f32::INFINITY, 1.0)];
    let zg = rt::asarray((z.clone(), &dg));
    let zc = rt::asarray((z.clone(), &dc));
    assert_eq!(host(&rt::is_nan(&zg)), host(&rt::is_nan(&zc)), "c32 is_nan");
    assert_eq!(host(&rt::is_inf(&zg)), host(&rt::is_inf(&zc)), "c32 is_inf");
    assert_eq!(host(&rt::is_finite(&zg)), host(&rt::is_finite(&zc)), "c32 is_finite");
}

/// `expm1`/`log1p` at small arguments (real and complex) must not cancel.
#[test]
fn regression_expm1_log1p_small_args() {
    require_cuda!();
    use num::Complex;
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    let x = vec![1e-16f64, -1e-16, 1e-8, -1e-8, 0.0];
    let g = rt::asarray((x.clone(), &dg));
    let c = rt::asarray((x.clone(), &dc));
    let close = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-15 * b.abs().max(1e-300);
    for (a, b) in host(&rt::expm1(&g)).iter().zip(&host(&rt::expm1(&c))) {
        assert!(close(*a, *b), "expm1 {a} vs {b}");
    }
    for (a, b) in host(&rt::log1p(&g)).iter().zip(&host(&rt::log1p(&c))) {
        assert!(close(*a, *b), "log1p {a} vs {b}");
    }

    let z: Vec<Complex<f64>> =
        vec![Complex::new(1e-16, 0.0), Complex::new(0.0, 1e-16), Complex::new(1e-8, -1e-8), Complex::new(-1.0, 0.0)];
    let zg = rt::asarray((z.clone(), &dg));
    let zc = rt::asarray((z.clone(), &dc));
    let eg = host(&rt::expm1(&zg));
    let ec = host(&rt::expm1(&zc));
    assert_c64("complex expm1", &eg, &ec, 1e-12);
    let lg = host(&rt::log1p(&zg));
    let lc = host(&rt::log1p(&zc));
    assert_c64("complex log1p", &lg, &lc, 1e-12);
}

/// Complex arithmetic (the C++ operator surface), large-magnitude math, the
/// C99 hyperbolic/trig branches, and `pow` special values.
#[test]
fn regression_complex_arithmetic_and_edges() {
    require_cuda!();
    use num::Complex;
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();

    let z1: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(-3.0, 0.5), Complex::new(0.0, -1.5)];
    let z2: Vec<Complex<f64>> = vec![Complex::new(0.5, -1.0), Complex::new(2.0, 0.0), Complex::new(-1.0, 4.0)];
    let (ag, bg) = (rt::asarray((z1.clone(), &dg)), rt::asarray((z2.clone(), &dg)));
    let (ac, bc) = (rt::asarray((z1.clone(), &dc)), rt::asarray((z2.clone(), &dc)));
    assert_c64("c add", &host(&rt::add(&ag, &bg)), &host(&rt::add(&ac, &bc)), 1e-12);
    assert_c64("c sub", &host(&rt::sub(&ag, &bg)), &host(&rt::sub(&ac, &bc)), 1e-12);
    assert_c64("c mul", &host(&rt::mul(&ag, &bg)), &host(&rt::mul(&ac, &bc)), 1e-12);
    assert_c64("c div", &host(&rt::div(&ag, &bg)), &host(&rt::div(&ac, &bc)), 1e-12);
    assert_c64("c square", &host(&rt::square(&ag)), &host(&rt::square(&ac)), 1e-12);
    assert_c64("c ext_neg", &host(&rt::ext_neg(&ag)), &host(&rt::ext_neg(&ac)), 1e-12);

    // compound assigns
    let mut mg = rt::asarray((z1.clone(), &dg));
    let mut mc = rt::asarray((z1.clone(), &dc));
    rt::add_assign(&mut mg, &bg);
    rt::add_assign(&mut mc, &bc);
    assert_c64("c add_assign", &host(&mg), &host(&mc), 1e-12);
    let mut mg = rt::asarray((z1.clone(), &dg));
    let mut mc = rt::asarray((z1.clone(), &dc));
    rt::mul_assign(&mut mg, &bg);
    rt::mul_assign(&mut mc, &bc);
    assert_c64("c mul_assign", &host(&mg), &host(&mc), 1e-12);

    // large magnitudes: overflow-free abs/log/sqrt/sign
    let big: Vec<Complex<f64>> =
        vec![Complex::new(1e200, 1e200), Complex::new(-1e200, 1e200), Complex::new(3e150, -4e150)];
    let (gg, gc) = (rt::asarray((big.clone(), &dg)), rt::asarray((big.clone(), &dc)));
    assert_c64(
        "big abs",
        &host(&rt::abs(&gg)).iter().map(|x| Complex::new(*x, 0.0)).collect::<Vec<_>>(),
        &host(&rt::abs(&gc)).iter().map(|x| Complex::new(*x, 0.0)).collect::<Vec<_>>(),
        1e-12,
    );
    assert_c64("big log", &host(&rt::log(&gg)), &host(&rt::log(&gc)), 1e-12);
    assert_c64("big sqrt", &host(&rt::sqrt(&gg)), &host(&rt::sqrt(&gc)), 1e-12);
    assert_c64("big sign", &host(&rt::sign(&gg)), &host(&rt::sign(&gc)), 1e-12);

    // C99 hyperbolic/trig special values
    let ex: Vec<Complex<f64>> =
        vec![Complex::new(-800.0, 0.0), Complex::new(0.0, 800.0), Complex::new(800.0, 0.0), Complex::new(30.0, 1.0)];
    let (xg, xc) = (rt::asarray((ex.clone(), &dg)), rt::asarray((ex.clone(), &dc)));
    assert_c64("c sinh", &host(&rt::sinh(&xg)), &host(&rt::sinh(&xc)), 1e-12);
    assert_c64("c cosh", &host(&rt::cosh(&xg)), &host(&rt::cosh(&xc)), 1e-12);
    assert_c64("c tanh", &host(&rt::tanh(&xg)), &host(&rt::tanh(&xc)), 1e-12);
    assert_c64("c tan", &host(&rt::tan(&xg)), &host(&rt::tan(&xc)), 1e-12);

    // pow special values: 0^0 = 1, 0^1 = 0, 0^-1 = NaN (not 0/inf)
    let zero = Complex::new(0.0, 0.0);
    let z0 = rt::asarray((vec![zero], &dg));
    let z0c = rt::asarray((vec![zero], &dc));
    let y0 = rt::asarray((vec![zero], &dg));
    let y0c = rt::asarray((vec![zero], &dc));
    assert_c64("0^0", &host(&rt::pow(&z0, &y0)), &host(&rt::pow(&z0c, &y0c)), 1e-12);
    let y1 = rt::asarray((vec![Complex::new(1.0, 0.0)], &dg));
    let y1c = rt::asarray((vec![Complex::new(1.0, 0.0)], &dc));
    assert_c64("0^1", &host(&rt::pow(&z0, &y1)), &host(&rt::pow(&z0c, &y1c)), 1e-12);
    let ym1 = rt::asarray((vec![Complex::new(-1.0, 0.0)], &dg));
    let ym1c = rt::asarray((vec![Complex::new(-1.0, 0.0)], &dc));
    assert_c64("0^-1", &host(&rt::pow(&z0, &ym1)), &host(&rt::pow(&z0c, &ym1c)), 1e-12);
}

/// In-place `not` (owned path) compiles and matches the CPU.
#[test]
fn regression_inplace_not() {
    require_cuda!();
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    let bg = rt::asarray((vec![true, false, true, false], &dg));
    let bc = rt::asarray((vec![true, false, true, false], &dc));
    assert_eq!(host(&rt::not(bg)), host(&rt::not(bc)), "bool not in place");
    let ig = rt::asarray((vec![5i32, -1, 0, 7], &dg));
    let ic = rt::asarray((vec![5i32, -1, 0, 7], &dc));
    assert_eq!(host(&rt::not(ig)), host(&rt::not(ic)), "int not in place");
}

/// A negative scalar integer exponent is rejected on both devices; a tensor
/// exponent is a documented divergence (not exercised here).
#[test]
fn regression_integer_pow_negative_scalar_exponent() {
    require_cuda!();
    let dg = DeviceCuda::new(0);
    let dc = DeviceCpuSerial::default();
    let g = rt::asarray((vec![2i64, 3], &dg));
    let c = rt::asarray((vec![2i64, 3], &dc));
    assert!(rt::pow_f(&g, -1i64).is_err(), "cuda must reject a negative scalar exponent");
    assert!(rt::pow_f(&c, -1i64).is_err(), "cpu rejects a negative scalar exponent");
}

/// Column-major devices: the tiled transpose and elementwise paths agree
/// with the CPU.
#[test]
fn regression_col_major_transpose() {
    require_cuda!();
    let mut dg = DeviceCuda::new(0);
    dg.set_default_order(ColMajor);
    let mut dc = DeviceCpuSerial::default();
    dc.set_default_order(ColMajor);
    for (m, n) in [(64usize, 64usize), (128, 100), (100, 128)] {
        let base = rt::arange((0.0f64, (m * n) as f64, 1.0, &dg)).into_shape([m, n]);
        let basec = rt::arange((0.0f64, (m * n) as f64, 1.0, &dc)).into_shape([m, n]);
        let bv = base.view();
        let bvc = basec.view();
        let bt = bv.swapaxes(0, 1);
        let btc = bvc.swapaxes(0, 1);
        let got = bt.to_contig(ColMajor).into_owned();
        let want = btc.to_contig(ColMajor).into_owned();
        assert_eq!(host_nd(&got), host_nd(&want), "col-major T-read {m}x{n}");

        let mut dst: Tensor<f64, _, _> = rt::zeros(([n, m], &dg));
        let dstv = dst.view_mut();
        let mut dtv = dstv.into_permute_dims((1, 0));
        dtv.assign(&base);
        let mut dstc: Tensor<f64, _, _> = rt::zeros(([n, m], &dc));
        let dstcv = dstc.view_mut();
        let mut dtvc = dstcv.into_permute_dims((1, 0));
        dtvc.assign(&basec);
        assert_eq!(host_nd(&dst), host_nd(&dstc), "col-major T-write {m}x{n}");
    }
}
