//! Per-ordinal device object and its process-wide registry.

use crate::error::{map_driver_error, map_driver_result};
use crate::flags;
use crate::matmul::drivers::cublas_lt::CublasLtHandle;
use crate::prelude_dev::*;
use cudarc::driver::CudaContext;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Heavyweight per-GPU state, canonical per ordinal via the registry.
///
/// Holds everything that is *identity* of one physical GPU: the primary
/// context, the default (non-blocking) stream, and later heavyweight
/// resources (BLAS handles, kernel cache, allocator config). The registry
/// canonicalizes it per ordinal so two `DeviceCuda::new(0)` calls share one
/// pool and one set of handles instead of forking state.
pub struct CudaOrdinalObj {
    pub(crate) ordinal: usize,
    pub(crate) ctx: Arc<CudaContext>,
    /// The default execution stream of this ordinal (non-blocking; never the
    /// legacy null stream). New devices start on it.
    pub(crate) default_stream: Arc<CudaStreamObj>,
    /// Per-ordinal compiled-kernel cache (lazily initialized).
    pub(crate) kernel_cache_slot: OnceLock<crate::kernel::KernelCachePublic>,
    /// Per-ordinal intern map of planned elementwise specializations.
    pub(crate) planned_cache_slot: OnceLock<crate::kernel::PlannedCache>,
    /// Cached arch string (one driver query per ordinal).
    pub(crate) arch_slot: OnceLock<core::result::Result<String, rstsr_core::prelude_dev::Error>>,
    /// The cuBLASLt handle and heuristic cache, or a cached "not available".
    pub(crate) cublaslt_slot: OnceLock<Option<Arc<CublasLtHandle>>>,
    /// The cuTENSOR handle, or a cached "not available".
    #[cfg(cutensor_bindings)]
    pub(crate) cutensor_slot: OnceLock<Option<Arc<crate::cutensor_wrap::handle::CutensorHandle>>>,
}

impl CudaOrdinalObj {
    /// Create the object for `ordinal` on its primary context.
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(map_driver_error)?;
        let default_stream = CudaStreamObj::new(&ctx)?;
        Ok(Self {
            ordinal,
            ctx,
            default_stream: Arc::new(default_stream),
            kernel_cache_slot: OnceLock::new(),
            planned_cache_slot: OnceLock::new(),
            arch_slot: OnceLock::new(),
            cublaslt_slot: OnceLock::new(),
            #[cfg(cutensor_bindings)]
            cutensor_slot: OnceLock::new(),
        })
    }

    /// The CUDA ordinal this object belongs to.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The shared primary context of this ordinal.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Compute capability `(major, minor)` of this ordinal's device. Panics
    /// through the error mapping; use [`CudaOrdinalObj::compute_capability_f`]
    /// for the fallible form.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability_f().rstsr_unwrap()
    }

    /// Fallible form of [`CudaOrdinalObj::compute_capability`].
    pub fn compute_capability_f(&self) -> Result<(i32, i32)> {
        map_driver_result(self.ctx.compute_capability())
    }

    /// The cuBLASLt state of this ordinal (see [`CublasLtHandle`]), or `None` when `libcublasLt` is
    /// not loadable (cached: the probe either works once or never).
    pub(crate) fn cublaslt(&self) -> Option<&Arc<CublasLtHandle>> {
        self.cublaslt_slot
            .get_or_init(|| {
                // cudarc resolves the symbols lazily and panics when one is missing.
                let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    CublasLtHandle::new(&self.default_stream)
                }));
                made.ok().and_then(|r| r.ok()).map(Arc::new)
            })
            .as_ref()
    }

    /// The cuTENSOR handle of this ordinal (see
    /// [`crate::cutensor_wrap::handle::CutensorHandle`]), or `None` when `libcutensor` is not
    /// loadable (cached: the probe either works once or never).
    #[cfg(cutensor_bindings)]
    #[allow(dead_code)] // wired by the B2 one-shot
    pub(crate) fn cutensor(&self) -> Option<&Arc<crate::cutensor_wrap::handle::CutensorHandle>> {
        self.cutensor_slot
            .get_or_init(|| {
                // cudarc resolves the symbols lazily and panics when one is missing.
                let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::cutensor_wrap::handle::CutensorHandle::new(&self.default_stream)
                }));
                made.ok().and_then(|r| r.ok()).map(Arc::new)
            })
            .as_ref()
    }

    /// `(free, total)` device memory in bytes. Panics through the error
    /// mapping; use [`CudaOrdinalObj::memory_info_f`] for the fallible form.
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info_f().rstsr_unwrap()
    }

    /// Fallible form of [`CudaOrdinalObj::memory_info`].
    pub fn memory_info_f(&self) -> Result<(usize, usize)> {
        map_driver_result(self.ctx.mem_get_info())
    }
}

/// Number of visible CUDA devices (0 if CUDA is unavailable).
pub fn device_count() -> usize {
    CudaContext::device_count().unwrap_or(0).max(0) as usize
}

/// Whether any CUDA device is available.
pub fn is_available() -> bool {
    device_count() > 0
}

impl Debug for CudaOrdinalObj {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaOrdinalObj").field("ordinal", &self.ordinal).finish()
    }
}

static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<CudaOrdinalObj>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<usize, Weak<CudaOrdinalObj>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Fetch or create the canonical per-ordinal object. Also seeds the
/// process-global flags default from the environment on first use.
pub(crate) fn ordinal_obj(ordinal: usize) -> Result<Arc<CudaOrdinalObj>> {
    flags::default_flags();
    let mut map = registry().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(obj) = map.get(&ordinal).and_then(|w| w.upgrade()) {
        return Ok(obj);
    }
    let obj = Arc::new(CudaOrdinalObj::new(ordinal)?);
    map.insert(ordinal, Arc::downgrade(&obj));
    Ok(obj)
}
