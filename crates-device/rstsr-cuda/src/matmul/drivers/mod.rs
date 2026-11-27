//! The matmul drivers: the candidate table and the ordered dispatch.
//!
//! A driver takes a resolved [`Problem`] and answers with a [`Verdict`]: it ran, or it declined
//! with the reason. Selection is the candidate table of the type triple ([`candidates`]), in the
//! chain order, filtered by the enabled set ([`MatMulDrivers`]); every driver that can serve the
//! triple is tried in turn until one runs. An error is returned only when nothing ran.
//!
//! A driver that declines never enqueues anything, so fall-through cannot change the answer.
//!
//! The three cuBLAS drivers share the argument derivation ([`cublas::prepare`]): which layouts
//! they can express is one rule, whatever entry is called.

use std::any::TypeId;

use super::flags::{MatMulDrivers, MatMulUseSyrk};
use super::geom::GemmGeom;
use super::mode::{ComputeType16F, ComputeType32F, ComputeType64F, CudaMatMulMode};
use crate::dtype::CudaDType;
use crate::prelude_dev::*;
use cudarc::cublas::sys;
pub(crate) use naive::NaiveScalar;

pub(crate) mod cublas;
pub(crate) mod cublas_ex;
pub(crate) mod cublas_lt;
pub(crate) mod naive;
pub(crate) mod syrk;
pub(crate) mod tiled;

/// One matmul driver of the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GemmDriver {
    /// cuBLASLt.
    CublasLt,
    /// `cublasGemmEx` / `cublasGemmStridedBatchedEx`.
    CublasEx,
    /// The typed classic entries (`cublasSgemm`, `cublasDgemm`, ...).
    Cublas,
    /// The tiled SIMT kernel (`drivers/tiled.rs`).
    Tiled,
    /// The universal elementwise kernel (`drivers/naive.rs`).
    Naive,
}

impl GemmDriver {
    /// Every driver, in the chain order.
    pub const ALL: [Self; 5] = [Self::CublasLt, Self::CublasEx, Self::Cublas, Self::Tiled, Self::Naive];

    /// Lower-case name, for messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::CublasLt => "cublasLt",
            Self::CublasEx => "cublasEx",
            Self::Cublas => "cublas",
            Self::Tiled => "tiled",
            Self::Naive => "naive",
        }
    }

    /// The enable-set bit of this driver.
    pub const fn flag(self) -> MatMulDrivers {
        let bit = match self {
            Self::CublasLt => 0,
            Self::CublasEx => 1,
            Self::Cublas => 2,
            Self::Tiled => 3,
            Self::Naive => 4,
        };
        MatMulDrivers::from_bits(1 << bit)
    }
}

/// Why a driver did not take the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The layout or the type triple is outside what the driver can express.
    Cannot(&'static str),
    /// cuBLAS answered with this status; nothing was enqueued.
    CublasStatus(cudarc::cublas::sys::cublasStatus_t),
    /// cuBLASLt answered with this status; nothing was enqueued.
    CublasLtStatus(cudarc::cublaslt::sys::cublasStatus_t),
    /// `libcublas` is not loadable on this machine.
    NoCublas,
    /// `libcublasLt` is not loadable on this machine.
    NoCublasLt,
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cannot(what) => write!(f, "cannot serve: {what}"),
            Self::CublasStatus(status) => write!(f, "cuBLAS status {status:?}"),
            Self::CublasLtStatus(status) => write!(f, "cuBLASLt status {status:?}"),
            Self::NoCublas => write!(f, "libcublas is not loadable"),
            Self::NoCublasLt => write!(f, "libcublasLt is not loadable"),
        }
    }
}

/// What a driver did with a problem.
pub(crate) enum Verdict {
    /// The product is enqueued.
    Ran,
    /// Nothing was enqueued; the next driver is asked.
    Declined(Refusal),
}

/// One resolved GEMM problem, ready for a driver.
pub(crate) struct Problem<'a, TA, TB, TC, TBuf> {
    pub device: &'a DeviceCuda,
    pub geom: GemmGeom,
    /// The output: `TC` or `MaybeUninit<TC>`. Never read when `read_c` is false.
    pub c: &'a CudaBuffer<TBuf>,
    pub a: &'a CudaBuffer<TA>,
    pub b: &'a CudaBuffer<TB>,
    pub alpha: TC,
    pub beta: TC,
    /// Whether the driver must add `beta * c`; false means `c` holds nothing to read.
    pub read_c: bool,
}

impl<TA, TB, TC, TBuf> Problem<'_, TA, TB, TC, TBuf> {
    /// The type names of the triple, for messages.
    pub fn triple(&self) -> String
    where
        TA: CudaDType,
        TB: CudaDType,
        TC: CudaDType,
    {
        format!("{} x {} -> {}", TA::CNAME, TB::CNAME, TC::CNAME)
    }
}

/// The problem's three buffers read as `A`, `B`, `C`.
pub(crate) struct Buffers<'a, A, B, C> {
    pub c: &'a CudaBuffer<C>,
    pub a: &'a CudaBuffer<A>,
    pub b: &'a CudaBuffer<B>,
}

impl<TA, TB, TC, TBuf> Problem<'_, TA, TB, TC, TBuf> {
    /// The buffers read as `A`, `B`, `C`, or `None` unless `TA == A`, `TB == B` and `TC == C`.
    ///
    /// The cuBLAS drivers check the triple at run time and re-type the buffers, instead of
    /// widening their bounds to every dtype.
    pub fn retyped<'a, A: 'static, B: 'static, C: 'static>(&'a self) -> Option<Buffers<'a, A, B, C>>
    where
        TA: 'static,
        TB: 'static,
        TC: 'static,
        TBuf: 'static,
    {
        if !(same::<TA, A>() && same::<TB, B>() && same::<TC, C>()) {
            return None;
        }
        Some(Buffers { c: self.c.view_as::<C>()?, a: self.a.view_as::<A>()?, b: self.b.view_as::<B>()? })
    }

    /// [`Self::retyped`] with one type for the three buffers.
    pub fn uniform<'a, T: Copy + 'static>(&'a self) -> Option<Buffers<'a, T, T, T>>
    where
        TA: 'static,
        TB: 'static,
        TC: 'static,
        TBuf: 'static,
    {
        self.retyped::<T, T, T>()
    }
}

/// `x` of the output type `TC`, as the scale type `S` of its family (the type itself, or `f32`
/// from a half output); `None` if `TC` is not of `S`'s family.
pub(crate) fn scale<S: Copy + 'static, TC: Copy + 'static>(x: TC) -> Option<S> {
    /// `x`, read as `U`: valid only when `TC` is known to be `U` (checked by `TypeId`).
    macro_rules! read {
        ($u:ty) => {
            unsafe { *(&x as *const TC as *const $u) }
        };
    }
    if same::<S, TC>() {
        return Some(read!(S));
    }
    // a half output takes `f32` scales: convert the value, then name it `S`
    let f = if same::<TC, half::f16>() {
        f32::from(read!(half::f16))
    } else if same::<TC, half::bf16>() {
        f32::from(read!(half::bf16))
    } else {
        return None;
    };
    if !same::<S, f32>() {
        return None;
    }
    Some(unsafe { *(&f as *const f32 as *const S) })
}

/// A scale handed to cuBLAS by pointer, in storage aligned to 16 bytes.
///
/// cuBLASLt reads the 16-byte `Complex<f64>` scales with aligned SIMD loads, while the type's own
/// alignment is 8: a plain local is 16-aligned only by chance (it was not in a debug build).
#[repr(C, align(16))]
pub(crate) struct AlignedScale<T>(pub T);

fn same<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// The `cudaDataType_t` of the three matrices, named after the **geom** operands (cuBLAS's slot
/// `A` holds `geom.b`; see [`cublas::prepare`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CublasGemmTypes {
    pub a: sys::cudaDataType_t,
    pub b: sys::cudaDataType_t,
    pub c: sys::cudaDataType_t,
}

/// The problem of one call as the cuBLAS drivers see it: the types of its three matrices and the
/// compute mode.
///
/// The mode is the one **valid for the triple**: one the triple cannot serve was resolved to the
/// family default before ([`plan`]), so a driver only maps, never downgrades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CublasPlan {
    /// `f32 x f32 -> f32`: the 32F mode.
    F32(ComputeType32F),
    /// `f64 x f64 -> f64`: the 64F mode.
    F64(ComputeType64F),
    /// `Complex<f32> x Complex<f32> -> Complex<f32>`: the 32F mode.
    C32(ComputeType32F),
    /// `Complex<f64>` throughout: the 64F mode.
    C64(ComputeType64F),
    /// `f16 x f16 -> f16`: the 16F mode.
    F16(ComputeType16F),
    /// `f16 x f16 -> f32`: the 32F mode.
    F16F32(ComputeType32F),
    /// `bf16 x bf16 -> bf16`: the 16F mode.
    BF16(ComputeType16F),
    /// `bf16 x bf16 -> f32`: the 32F mode.
    BF16F32(ComputeType32F),
    /// No cuBLAS problem: integers, or operands of different dtypes.
    None,
}

impl CublasPlan {
    /// The `cudaDataType_t` of the three matrices, in **geom** operand order (`a`, `b`, output);
    /// `None` without a cuBLAS problem.
    pub fn gemm_types(self) -> Option<CublasGemmTypes> {
        use sys::cudaDataType_t as D;
        let (a, b, c) = match self {
            Self::F32(_) => (D::CUDA_R_32F, D::CUDA_R_32F, D::CUDA_R_32F),
            Self::F64(_) => (D::CUDA_R_64F, D::CUDA_R_64F, D::CUDA_R_64F),
            Self::C32(_) => (D::CUDA_C_32F, D::CUDA_C_32F, D::CUDA_C_32F),
            Self::C64(_) => (D::CUDA_C_64F, D::CUDA_C_64F, D::CUDA_C_64F),
            Self::F16(_) => (D::CUDA_R_16F, D::CUDA_R_16F, D::CUDA_R_16F),
            Self::F16F32(_) => (D::CUDA_R_16F, D::CUDA_R_16F, D::CUDA_R_32F),
            Self::BF16(_) => (D::CUDA_R_16BF, D::CUDA_R_16BF, D::CUDA_R_16BF),
            Self::BF16F32(_) => (D::CUDA_R_16BF, D::CUDA_R_16BF, D::CUDA_R_32F),
            Self::None => return None,
        };
        Some(CublasGemmTypes { a, b, c })
    }

    /// The `cublasGemmEx`/cuBLASLt compute type; `None` where the plan has none (the fixed-point
    /// emulation is a math mode of the classic handle).
    pub fn compute(self) -> Option<sys::cublasComputeType_t> {
        match self {
            Self::F32(mode) | Self::C32(mode) | Self::F16F32(mode) | Self::BF16F32(mode) => Some(mode.gemm_ex()),
            Self::F64(mode) | Self::C64(mode) => mode.gemm_ex(),
            Self::F16(mode) | Self::BF16(mode) => Some(mode.gemm_ex()),
            Self::None => None,
        }
    }
}

/// The plan of one call: the type triple with its mode, the mode resolved to one the triple can
/// serve. Every `CudaMatMulMode` impl pairs an output type with the mode enum of its family, so
/// the mode is re-typed by the output type's `TypeId` (the shape `rules/code-concepts.md` names).
///
/// A mode the triple cannot serve resolves to the family default silently: the `Fast*` modes of
/// the 32F family down-convert `f32` inputs, so they are not for complex inputs nor for half
/// inputs with an `f32` output; `Acc16F` accumulates `f16` only, so a `bf16` output keeps the
/// default. `Pedantic` is served by every triple, so it is never downgraded.
pub(crate) fn plan<TA: 'static, TB: 'static, TC: CudaMatMulMode>(mode: &TC::Mode) -> CublasPlan {
    use num::Complex;
    if same::<TC, f32>() {
        // SAFETY: these output types have `Mode` of the family named here.
        let mode32 = unsafe { *(mode as *const TC::Mode as *const ComputeType32F) };
        if same::<TA, f32>() && same::<TB, f32>() {
            CublasPlan::F32(mode32)
        } else if same::<TA, half::f16>() && same::<TB, half::f16>() {
            CublasPlan::F16F32(resolve_32f(mode32))
        } else if same::<TA, half::bf16>() && same::<TB, half::bf16>() {
            CublasPlan::BF16F32(resolve_32f(mode32))
        } else {
            CublasPlan::None
        }
    } else if same::<TC, f64>() {
        // SAFETY: as above.
        CublasPlan::F64(if same::<TA, f64>() && same::<TB, f64>() {
            unsafe { *(mode as *const TC::Mode as *const ComputeType64F) }
        } else {
            return CublasPlan::None;
        })
    } else if same::<TC, Complex<f32>>() {
        // SAFETY: as above.
        let mode32 = unsafe { *(mode as *const TC::Mode as *const ComputeType32F) };
        if same::<TA, Complex<f32>>() && same::<TB, Complex<f32>>() {
            CublasPlan::C32(resolve_32f(mode32))
        } else {
            CublasPlan::None
        }
    } else if same::<TC, Complex<f64>>() {
        // SAFETY: as above.
        let mode64 = unsafe { *(mode as *const TC::Mode as *const ComputeType64F) };
        if same::<TA, Complex<f64>>() && same::<TB, Complex<f64>>() {
            // the fixed-point emulation is a real-`f64` math mode
            CublasPlan::C64(if mode64 == ComputeType64F::EmulatedFixedPoint {
                ComputeType64F::default()
            } else {
                mode64
            })
        } else {
            CublasPlan::None
        }
    } else if same::<TC, half::f16>() {
        // SAFETY: as above.
        let mode16 = unsafe { *(mode as *const TC::Mode as *const ComputeType16F) };
        if same::<TA, half::f16>() && same::<TB, half::f16>() {
            CublasPlan::F16(mode16)
        } else {
            CublasPlan::None
        }
    } else if same::<TC, half::bf16>() {
        // SAFETY: as above.
        let mode16 = unsafe { *(mode as *const TC::Mode as *const ComputeType16F) };
        if same::<TA, half::bf16>() && same::<TB, half::bf16>() {
            CublasPlan::BF16(if mode16 == ComputeType16F::Acc16F { ComputeType16F::default() } else { mode16 })
        } else {
            CublasPlan::None
        }
    } else {
        CublasPlan::None
    }
}

/// A 32F mode for inputs that are not `f32`: the `Fast*` conversions do not apply.
fn resolve_32f(mode: ComputeType32F) -> ComputeType32F {
    match mode {
        ComputeType32F::Default | ComputeType32F::Pedantic => mode,
        _ => ComputeType32F::Default,
    }
}

/// An ordered candidate list: at most one entry per driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Chain {
    list: [GemmDriver; N],
    len: u8,
}

/// The number of drivers.
const N: usize = GemmDriver::ALL.len();

impl Chain {
    const EMPTY: [GemmDriver; N] = [GemmDriver::Naive; N];

    fn new(list: &[GemmDriver]) -> Self {
        let mut out = Self { list: Self::EMPTY, len: list.len() as u8 };
        out.list[..list.len()].copy_from_slice(list);
        out
    }

    pub fn iter(&self) -> impl Iterator<Item = GemmDriver> + '_ {
        self.list[..self.len as usize].iter().copied()
    }
}

/// The rows of the table with the tiled kernel: every driver, in the chain order.
const ACCELERATED: [GemmDriver; 5] = GemmDriver::ALL;

/// The drivers that can serve `(TA, TB) -> TC` under `mode`, in the chain order.
///
/// Pure: a table lookup on the type triple and the mode, with no layout input. (Layout decides
/// whether a listed driver *can*; that is the run-time fall-through.)
///
/// | call | mode | candidates in order |
/// |---|---|---|
/// | `f32 -> f32` | any 32F mode | cuBLASLt, `cublasEx`, cuBLAS, tiled, naive |
/// | `f64 -> f64` | `Default` / `Pedantic` | cuBLASLt, `cublasEx`, cuBLAS, tiled, naive |
/// | `f64 -> f64` | `EmulatedFixedPoint` | cuBLAS, tiled, naive |
/// | `c32`, `c64` | `Default` / `Pedantic` | cuBLASLt, `cublasEx`, cuBLAS, naive |
/// | `f16 -> f16` | `Default` / `Pedantic` | cuBLASLt, `cublasEx`, naive |
/// | `f16 -> f16` | `Acc16F` | cuBLASLt, `cublasEx`, naive |
/// | `f16 -> f32`, `bf16 -> bf16`, `bf16 -> f32` | `Default` / `Pedantic` | cuBLASLt, `cublasEx`, naive |
/// | anything else | — | naive |
///
/// (Modes are resolved by [`plan`] first, so `f16 -> f32` under `FastTF32` arrives here as
/// `Default`.) The tiled and naive kernels are mode-agnostic: they compute in the output precision
/// (or better), which every mode allows, so only the cuBLAS drivers read the plan.
pub(crate) fn candidates<TA: 'static, TB: 'static, TC: CudaMatMulMode>(mode: &TC::Mode) -> Chain {
    match plan::<TA, TB, TC>(mode) {
        CublasPlan::F32(_) => Chain::new(&ACCELERATED),
        CublasPlan::F64(ComputeType64F::EmulatedFixedPoint) => {
            Chain::new(&[GemmDriver::Cublas, GemmDriver::Tiled, GemmDriver::Naive])
        },
        CublasPlan::F64(_) => Chain::new(&ACCELERATED),
        CublasPlan::C32(_) | CublasPlan::C64(_) => {
            Chain::new(&[GemmDriver::CublasLt, GemmDriver::CublasEx, GemmDriver::Cublas, GemmDriver::Naive])
        },
        CublasPlan::F16(_) | CublasPlan::F16F32(_) | CublasPlan::BF16(_) | CublasPlan::BF16F32(_) => {
            Chain::new(&[GemmDriver::CublasLt, GemmDriver::CublasEx, GemmDriver::Naive])
        },
        CublasPlan::None => Chain::new(&[GemmDriver::Naive]),
    }
}

/// Whether the symmetric form is tried before the chain: the layout condition of the BLAS devices
/// ([`CallerGeom::is_syrk`]), the same buffer for both operands, `beta` zero, one type, the output
/// type allowed by [`MatMulUseSyrk`], and the classic driver enabled (which owns the routine).
fn syrk_applies<TA: 'static, TB: 'static, TC: 'static, TBuf>(
    p: &Problem<'_, TA, TB, TC, TBuf>,
    enabled: MatMulDrivers,
    use_syrk: MatMulUseSyrk,
) -> bool {
    same::<TA, TB>()
        && same::<TB, TC>()
        && !p.read_c
        && p.geom.n_batch == 1
        && p.geom.caller().is_syrk()
        && std::ptr::eq(p.a.device_ptr() as *const u8, p.b.device_ptr() as *const u8)
        && use_syrk.takes::<TC>()
        && enabled.contains(GemmDriver::Cublas.flag())
}

/// Run the problem on the first enabled driver that takes it.
pub(crate) fn dispatch<TA, TB, TC, TBuf>(
    p: &Problem<'_, TA, TB, TC, TBuf>,
    flags: &crate::flags::CudaRuntimeFlags,
    mode: &TC::Mode,
) -> Result<()>
where
    TA: NaiveScalar,
    TB: NaiveScalar,
    TC: NaiveScalar + CudaMatMulMode,
    TBuf: 'static,
{
    let plan = plan::<TA, TB, TC>(mode);
    let enabled = flags.matmul_drivers;
    let trace = |what: &str, verdict: &Verdict| {
        if flags.trace_matmul {
            let g = &p.geom;
            eprintln!(
                "[rstsr-cuda matmul] {} {}x{}x{} batch {} mode {:?}: {} -> {}",
                p.triple(),
                g.m,
                g.k,
                g.n,
                g.n_batch,
                mode,
                what,
                match verdict {
                    Verdict::Ran => "ran".to_string(),
                    Verdict::Declined(r) => format!("declined ({r})"),
                }
            );
        }
    };
    // the symmetric form first: `a @ a^T` with a fresh output is half the work of the gemm
    if syrk_applies::<TA, TB, TC, TBuf>(p, enabled, flags.matmul_use_syrk) {
        let verdict = syrk::run(p, plan, flags.fixedpoint_mantissa_bits)?;
        trace("cublas (syrk)", &verdict);
        if let Verdict::Ran = verdict {
            return Ok(());
        }
    }
    let mut declined: Vec<(&'static str, Refusal)> = Vec::new();
    for driver in candidates::<TA, TB, TC>(mode).iter() {
        if !enabled.contains(driver.flag()) {
            continue;
        }
        let verdict = match driver {
            GemmDriver::CublasLt => cublas_lt::run(p, plan)?,
            GemmDriver::CublasEx => cublas_ex::run(p, plan)?,
            GemmDriver::Cublas => cublas::run(p, plan, flags.fixedpoint_mantissa_bits)?,
            GemmDriver::Tiled => tiled::run(p)?,
            GemmDriver::Naive => naive::run(p)?,
        };
        trace(driver.name(), &verdict);
        match verdict {
            Verdict::Ran => return Ok(()),
            Verdict::Declined(r) => declined.push((driver.name(), r)),
        }
    }
    let g = &p.geom;
    let why: Vec<String> = declined.iter().map(|(name, r)| format!("{name}: {r}")).collect();
    rstsr_raise!(
        UnImplemented,
        "cuda matmul: no enabled driver can serve {} ({}x{} @ {}x{}, batch {}); enabled {:?}; {}",
        p.triple(),
        g.m,
        g.k,
        g.k,
        g.n,
        g.n_batch,
        enabled,
        if why.is_empty() {
            "no enabled driver is a candidate for this type triple".to_string()
        } else {
            format!("declined: {}", why.join("; "))
        }
    )
}

/// The on-device bytes of a scalar argument (at most 16 bytes).
pub(crate) fn scalar_bytes<T>(x: &T) -> [u8; 16] {
    let nb = core::mem::size_of::<T>();
    let mut bytes = [0u8; 16];
    debug_assert!(nb <= 16, "a scalar argument wider than 16 bytes is not a scalar");
    // SAFETY: `T: CudaDType` is a plain scalar, so its bytes are its on-device representation.
    unsafe { core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*x) as *const u8, bytes.as_mut_ptr(), nb) };
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::mode::{ComputeType16F, ComputeType32F, ComputeType64F};

    fn names(c: &Chain) -> Vec<&'static str> {
        c.iter().map(|d| d.name()).collect()
    }

    /// Every 32F mode gives the full chain to `f32 -> f32`.
    #[test]
    fn table_32f() {
        let chain = ["cublasLt", "cublasEx", "cublas", "tiled", "naive"];
        let modes = [
            ComputeType32F::default(),
            ComputeType32F::Pedantic,
            ComputeType32F::Fast16F,
            ComputeType32F::Fast16BF,
            ComputeType32F::FastTF32,
        ];
        for mode in modes {
            assert_eq!(names(&candidates::<f32, f32, f32>(&mode)), chain, "{mode:?}");
        }
    }

    /// `f64 -> f64`: the full chain, except that the fixed-point emulation is served by the
    /// classic handle only.
    #[test]
    fn table_64f() {
        let chain = ["cublasLt", "cublasEx", "cublas", "tiled", "naive"];
        assert_eq!(names(&candidates::<f64, f64, f64>(&ComputeType64F::default())), chain);
        assert_eq!(names(&candidates::<f64, f64, f64>(&ComputeType64F::Pedantic)), chain);
        assert_eq!(names(&candidates::<f64, f64, f64>(&ComputeType64F::EmulatedFixedPoint)), [
            "cublas", "tiled", "naive"
        ]);
    }

    /// The plan is keyed by the type triple: the family mode for the accelerated triples, none for
    /// the others.
    #[test]
    fn plan_by_type_triple() {
        use num::Complex;
        let p32 = CublasPlan::F32(ComputeType32F::FastTF32);
        assert_eq!(plan::<f32, f32, f32>(&ComputeType32F::FastTF32), p32);
        assert_eq!(plan::<f64, f64, f64>(&ComputeType64F::Pedantic), CublasPlan::F64(ComputeType64F::Pedantic));
        assert_eq!(
            plan::<Complex<f64>, Complex<f64>, Complex<f64>>(&ComputeType64F::default()),
            CublasPlan::C64(ComputeType64F::default())
        );
        assert_eq!(plan::<i32, i32, i32>(&()), CublasPlan::None);
        assert_eq!(
            plan::<half::f16, half::f16, half::f16>(&ComputeType16F::default()),
            CublasPlan::F16(ComputeType16F::default())
        );
        assert_eq!(
            plan::<half::f16, half::f16, f32>(&ComputeType32F::default()),
            CublasPlan::F16F32(ComputeType32F::Default)
        );
        assert_eq!(
            plan::<half::bf16, half::bf16, half::bf16>(&ComputeType16F::default()),
            CublasPlan::BF16(ComputeType16F::Default)
        );
        assert_eq!(
            plan::<half::bf16, half::bf16, f32>(&ComputeType32F::default()),
            CublasPlan::BF16F32(ComputeType32F::Default)
        );
        // operands of another dtype than the output names
        assert_eq!(plan::<half::f16, half::bf16, f32>(&ComputeType32F::default()), CublasPlan::None);
        assert_eq!(plan::<f32, f64, f64>(&ComputeType64F::default()), CublasPlan::None);
    }

    /// A mode the triple cannot serve is the family default by the time a driver sees it;
    /// `Pedantic` is served everywhere and never downgraded.
    #[test]
    fn modes_the_triple_cannot_serve_resolve_to_default() {
        use num::Complex;
        let fast = [ComputeType32F::Fast16F, ComputeType32F::Fast16BF, ComputeType32F::FastTF32];
        for mode in fast {
            assert_eq!(
                plan::<num::Complex<f32>, Complex<f32>, Complex<f32>>(&mode),
                CublasPlan::C32(ComputeType32F::Default),
                "{mode:?}"
            );
            assert_eq!(
                plan::<half::f16, half::f16, f32>(&mode),
                CublasPlan::F16F32(ComputeType32F::Default),
                "{mode:?}"
            );
            assert_eq!(
                plan::<half::bf16, half::bf16, f32>(&mode),
                CublasPlan::BF16F32(ComputeType32F::Default),
                "{mode:?}"
            );
        }
        assert_eq!(
            plan::<Complex<f32>, Complex<f32>, Complex<f32>>(&ComputeType32F::Pedantic),
            CublasPlan::C32(ComputeType32F::Pedantic)
        );
        assert_eq!(
            plan::<half::f16, half::f16, f32>(&ComputeType32F::Pedantic),
            CublasPlan::F16F32(ComputeType32F::Pedantic)
        );
        // no bf16 accumulation exists; the fixed-point emulation is real only
        assert_eq!(
            plan::<Complex<f64>, Complex<f64>, Complex<f64>>(&ComputeType64F::EmulatedFixedPoint),
            CublasPlan::C64(ComputeType64F::Default)
        );
        assert_eq!(
            plan::<half::bf16, half::bf16, half::bf16>(&ComputeType16F::Acc16F),
            CublasPlan::BF16(ComputeType16F::Default)
        );
        assert_eq!(
            plan::<half::f16, half::f16, half::f16>(&ComputeType16F::Acc16F),
            CublasPlan::F16(ComputeType16F::Acc16F)
        );
    }

    /// Every plan names the storage of its three matrices and a compute type, except the
    /// fixed-point emulation.
    #[test]
    fn plan_gemm_types_and_compute() {
        use num::Complex;
        use sys::cudaDataType_t as D;
        let gemm_types = |p: CublasPlan| p.gemm_types().map(|s| (s.a, s.b, s.c));
        assert_eq!(
            gemm_types(CublasPlan::F32(ComputeType32F::default())),
            Some((D::CUDA_R_32F, D::CUDA_R_32F, D::CUDA_R_32F))
        );
        assert_eq!(
            gemm_types(CublasPlan::C64(ComputeType64F::default())),
            Some((D::CUDA_C_64F, D::CUDA_C_64F, D::CUDA_C_64F))
        );
        assert_eq!(
            gemm_types(CublasPlan::F16F32(ComputeType32F::default())),
            Some((D::CUDA_R_16F, D::CUDA_R_16F, D::CUDA_R_32F))
        );
        assert_eq!(
            gemm_types(CublasPlan::BF16F32(ComputeType32F::default())),
            Some((D::CUDA_R_16BF, D::CUDA_R_16BF, D::CUDA_R_32F))
        );
        assert_eq!(
            gemm_types(CublasPlan::BF16(ComputeType16F::default())),
            Some((D::CUDA_R_16BF, D::CUDA_R_16BF, D::CUDA_R_16BF))
        );
        assert_eq!(gemm_types(CublasPlan::None), None);
        assert_eq!(
            plan::<Complex<f32>, Complex<f32>, Complex<f32>>(&ComputeType32F::Pedantic).compute(),
            Some(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC)
        );
        assert_eq!(
            plan::<half::f16, half::f16, half::f16>(&ComputeType16F::Acc16F).compute(),
            Some(sys::cublasComputeType_t::CUBLAS_COMPUTE_16F)
        );
        assert_eq!(plan::<f64, f64, f64>(&ComputeType64F::EmulatedFixedPoint).compute(), None);
    }

    /// The scale of a half output is `f32`; a type of another family has none.
    #[test]
    fn scales_of_the_output_type() {
        use num::Complex;
        assert_eq!(scale::<f32, f32>(1.5), Some(1.5));
        assert_eq!(scale::<f32, half::f16>(half::f16::from_f32(1.5)), Some(1.5));
        assert_eq!(scale::<f32, half::bf16>(half::bf16::from_f32(1.5)), Some(1.5));
        assert_eq!(scale::<f64, f64>(1.5), Some(1.5));
        assert_eq!(scale::<Complex<f32>, Complex<f32>>(Complex::new(1.0, 2.0)), Some(Complex::new(1.0, 2.0)));
        assert_eq!(scale::<f32, f64>(1.5), None);
        assert_eq!(scale::<f64, f32>(1.5), None);
    }

    /// Every other triple starts and ends at naive, whatever the mode.
    #[test]
    fn table_by_type_triple() {
        let naive = ["naive"];
        let blas = ["cublasLt", "cublasEx", "cublas", "naive"];
        let half = ["cublasLt", "cublasEx", "naive"];
        assert_eq!(names(&candidates::<i32, i32, i32>(&())), naive);
        assert_eq!(names(&candidates::<f32, f64, f64>(&ComputeType64F::default())), naive, "mixed operands");
        assert_eq!(names(&candidates::<f32, f64, f32>(&ComputeType32F::default())), naive, "mixed operands, f32 out");
        assert_eq!(
            names(&candidates::<half::f16, half::bf16, f32>(&ComputeType32F::default())),
            naive,
            "mixed operands, f32 out"
        );
        assert_eq!(names(&candidates::<half::f16, half::f16, f32>(&ComputeType32F::default())), half);
        assert_eq!(names(&candidates::<half::bf16, half::bf16, f32>(&ComputeType32F::default())), half);
        assert_eq!(names(&candidates::<half::f16, half::f16, half::f16>(&ComputeType16F::default())), half);
        assert_eq!(names(&candidates::<half::f16, half::f16, half::f16>(&ComputeType16F::Acc16F)), half);
        assert_eq!(names(&candidates::<half::bf16, half::bf16, half::bf16>(&ComputeType16F::default())), half);
        assert_eq!(
            names(&candidates::<num::Complex<f64>, num::Complex<f64>, num::Complex<f64>>(&ComputeType64F::default())),
            blas
        );
        // the emulation of the 64F family is real `f64` only: a `c64` output keeps the default
        assert_eq!(
            names(&candidates::<num::Complex<f64>, num::Complex<f64>, num::Complex<f64>>(
                &ComputeType64F::EmulatedFixedPoint
            )),
            blas
        );
    }

    /// The candidate list holds every driver at most once and never overflows.
    #[test]
    fn chain_bounds() {
        assert_eq!(Chain::new(&GemmDriver::ALL).iter().collect::<Vec<_>>(), GemmDriver::ALL.to_vec());
        assert_eq!(Chain::new(&[]).iter().count(), 0);
    }

    /// Each driver owns one enable-set bit, and the default set enables all of them.
    #[test]
    fn driver_flags() {
        for d in GemmDriver::ALL {
            assert!(MatMulDrivers::all().contains(d.flag()), "{}", d.name());
            assert!(!MatMulDrivers::none().contains(d.flag()));
        }
    }

    /// The scales handed to cuBLAS sit in 16-byte-aligned storage, whatever their own alignment:
    /// cuBLASLt loads a `Complex<f64>` scale with an aligned SIMD load.
    #[test]
    fn scales_are_aligned_for_cublas() {
        let a = AlignedScale(num::Complex::new(1.0f64, 0.0));
        assert_eq!(core::mem::align_of::<num::Complex<f64>>(), 8);
        assert_eq!(core::mem::align_of_val(&a), 16);
        assert_eq!(&a.0 as *const _ as usize, &a as *const _ as usize);
    }
}
