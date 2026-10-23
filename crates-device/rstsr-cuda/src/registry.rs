//! Per-ordinal device object and its process-wide registry.

use crate::error::{map_driver_error, map_driver_result};
use crate::flags;
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
    /// Cached arch string (one driver query per ordinal).
    pub(crate) arch_slot:
        OnceLock<core::result::Result<String, rstsr_core::prelude_dev::Error>>,
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
            arch_slot: OnceLock::new(),
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

    /// Compute capability `(major, minor)` of this ordinal's device.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        map_driver_result(self.ctx.compute_capability())
    }

    /// `(free, total)` device memory in bytes.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        map_driver_result(self.ctx.mem_get_info())
    }
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
