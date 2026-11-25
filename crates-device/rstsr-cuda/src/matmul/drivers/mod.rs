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

use super::flags::MatMulDrivers;
use super::geom::GemmGeom;
use super::mode::{ComputeType32F, ComputeType64F, CudaMatMulMode};
use crate::dtype::CudaDType;
use crate::prelude_dev::*;
pub(crate) use naive::NaiveScalar;

pub(crate) mod cublas;
pub(crate) mod cublas_ex;
pub(crate) mod cublas_lt;
pub(crate) mod naive;
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

/// The problem with all three buffers read as one type `T`.
pub(crate) struct Uniform<'a, T> {
    pub c: &'a CudaBuffer<T>,
    pub a: &'a CudaBuffer<T>,
    pub b: &'a CudaBuffer<T>,
    pub alpha: T,
    pub beta: T,
}

/// `Some` iff `TA == TB == TC == T`, with the buffers re-typed accordingly.
///
/// The single-type drivers (cuBLAS, tiled) check the triple at run time and re-type the buffers,
/// instead of widening their bound to every dtype.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uniform<'a, T: Copy + 'static, TA: 'static, TB: 'static, TC: 'static, TBuf: 'static>(
    c: &'a CudaBuffer<TBuf>,
    a: &'a CudaBuffer<TA>,
    b: &'a CudaBuffer<TB>,
    alpha: TC,
    beta: TC,
) -> Option<Uniform<'a, T>> {
    if TypeId::of::<TC>() != TypeId::of::<T>() {
        return None;
    }
    Some(Uniform {
        c: c.view_as::<T>()?,
        a: a.view_as::<T>()?,
        b: b.view_as::<T>()?,
        // SAFETY: `TC` is `T` (checked above), so this is a plain copy of a `T`.
        alpha: unsafe { *(&alpha as *const TC as *const T) },
        beta: unsafe { *(&beta as *const TC as *const T) },
    })
}

impl<'a, TA, TB, TC, TBuf> Problem<'a, TA, TB, TC, TBuf> {
    /// [`uniform`] over this problem.
    pub fn uniform<T: Copy + 'static>(&self) -> Option<Uniform<'a, T>>
    where
        TA: 'static,
        TB: 'static,
        TC: Copy + 'static,
        TBuf: 'static,
    {
        uniform(self.c, self.a, self.b, self.alpha, self.beta)
    }
}

fn same<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// The compute mode of one call, resolved for its output type: what the cuBLAS drivers are asked
/// for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CublasPlan {
    /// An `f32` output: the 32F mode.
    F32(ComputeType32F),
    /// An `f64` output: the 64F mode.
    F64(ComputeType64F),
    /// No cuBLAS compute mode: integers, half, or a type the accelerated drivers do not serve yet.
    None,
}

/// The compute mode of one call, resolved for its output type: what the cuBLAS drivers are asked
/// for. Every `CudaMatMulMode` impl pairs an output type with the mode enum of its family, so the
/// mode is re-typed by the output type's `TypeId` (the shape `rules/code-concepts.md` names).
///
/// Complex and half outputs have no accelerated driver yet, so they have no plan; with those
/// drivers, this is where a mode of the right family that the triple cannot serve resolves to the
/// family default (never [`CublasPlan::None`], which would hand an asked-for mode to a kernel that
/// never heard of it).
pub(crate) fn plan<TC: CudaMatMulMode>(mode: &TC::Mode) -> CublasPlan {
    if same::<TC, f32>() {
        // SAFETY: these output types have `Mode = ComputeType32F`.
        CublasPlan::F32(unsafe { *(mode as *const TC::Mode as *const ComputeType32F) })
    } else if same::<TC, f64>() {
        // SAFETY: these output types have `Mode = ComputeType64F`.
        CublasPlan::F64(unsafe { *(mode as *const TC::Mode as *const ComputeType64F) })
    } else {
        CublasPlan::None
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

/// The accelerated rows of the table: every driver, in the chain order.
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
/// | anything else | — | naive |
///
/// The tiled and naive kernels are mode-agnostic: they compute in the output precision (or
/// better), which every mode allows, so only the cuBLAS drivers read the plan.
pub(crate) fn candidates<TA: 'static, TB: 'static, TC: CudaMatMulMode>(mode: &TC::Mode) -> Chain {
    let same_type = same::<TA, TC>() && same::<TB, TC>();
    match plan::<TC>(mode) {
        CublasPlan::F32(_) if same_type => Chain::new(&ACCELERATED),
        CublasPlan::F64(ComputeType64F::EmulatedFixedPoint) if same_type => {
            Chain::new(&[GemmDriver::Cublas, GemmDriver::Tiled, GemmDriver::Naive])
        },
        CublasPlan::F64(_) if same_type => Chain::new(&ACCELERATED),
        CublasPlan::F32(_) | CublasPlan::F64(_) | CublasPlan::None => Chain::new(&[GemmDriver::Naive]),
    }
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
    let plan = plan::<TC>(mode);
    let enabled = flags.matmul_drivers;
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
                driver.name(),
                match &verdict {
                    Verdict::Ran => "ran".to_string(),
                    Verdict::Declined(r) => format!("declined ({r})"),
                }
            );
        }
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

    /// The plan is keyed by the output type: the family mode for `f32`/`f64`, none for the types
    /// without an accelerated driver yet.
    #[test]
    fn plan_by_output_type() {
        assert_eq!(plan::<f32>(&ComputeType32F::FastTF32), CublasPlan::F32(ComputeType32F::FastTF32));
        assert_eq!(plan::<f64>(&ComputeType64F::Pedantic), CublasPlan::F64(ComputeType64F::Pedantic));
        assert_eq!(plan::<num::Complex<f64>>(&ComputeType64F::default()), CublasPlan::None);
        assert_eq!(plan::<i32>(&()), CublasPlan::None);
        assert_eq!(plan::<half::bf16>(&ComputeType16F::default()), CublasPlan::None);
    }

    /// Every other triple starts and ends at naive, whatever the mode.
    #[test]
    fn table_by_type_triple() {
        let naive = ["naive"];
        assert_eq!(names(&candidates::<i32, i32, i32>(&())), naive);
        assert_eq!(names(&candidates::<f32, f64, f64>(&ComputeType64F::default())), naive, "mixed operands");
        assert_eq!(names(&candidates::<f32, f64, f32>(&ComputeType32F::default())), naive, "mixed operands, f32 out");
        assert_eq!(names(&candidates::<half::f16, half::f16, f32>(&ComputeType32F::default())), naive);
        assert_eq!(names(&candidates::<half::bf16, half::bf16, half::bf16>(&ComputeType16F::default())), naive);
        assert_eq!(
            names(&candidates::<num::Complex<f64>, num::Complex<f64>, num::Complex<f64>>(&ComputeType64F::default())),
            naive
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
}
