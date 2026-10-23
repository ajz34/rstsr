//! Kernel compilation, caching, and launch (B1 infrastructure, D2–D4).
//!
//! - Sources are Rust-generated CUDA C per kernel family with a shared
//!   prelude; one NVRTC program per specialization, loaded lazily.
//! - Compilation: PTX via the safe layer, `compute_<cc>` arch derived at
//!   runtime from the device; strict library-fixed options.
//! - Cache: per-ordinal in-memory `KernelKey -> Arc<CudaFunction>` plus an
//!   on-disk cache of the **PTX text** (what `load_module` consumes).
//! - Launch: raw `result::launch_kernel` with a hand-built `void*` param
//!   array — marshaling is plain bytes (device pointers as u64, lengths/
//!   strides as u64, fill scalars as little-endian bytes), so no cudarc
//!   dtype trait is ever on our path.
//!
//! Layout convention (D9 T0 floor): kernels are rank-generic, taking
//! runtime shape/stride arrays (element units, pre-multiplied host-side)
//! plus the base pointer and offset. Copy/fill-bytes/gather kernels work
//! in bytes on `char*` and are dtype-free (one entry per arch); typed
//! kernels generate from `CudaDType::CNAME`.

use crate::buffer::CudaBuffer;
use crate::error::{map_driver_error, map_driver_result};
use crate::registry::CudaOrdinalObj;
use crate::stream::CudaStreamObj;
use rstsr_core::prelude_dev::*;

use cudarc::driver::sys::{CUevent, CUevent_flags, CUevent_wait_flags};
use cudarc::driver::{result as dr, CudaFunction};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Shared CUDA C prelude: fixed-width names, complex pairs, IEEE helpers.
pub(crate) const CUDA_PRELUDE: &str = r#"
typedef signed char rstsr_i8; typedef short rstsr_i16; typedef int rstsr_i32;
typedef long long rstsr_i64; typedef long long rstsr_isz;
typedef unsigned char rstsr_u8; typedef unsigned short rstsr_u16; typedef unsigned int rstsr_u32;
typedef unsigned long long rstsr_u64; typedef unsigned long long rstsr_usz;
struct rstsr_c32 { float re, im; };
struct rstsr_c64 { double re, im; };
__device__ rstsr_c32 make_c32(float re, float im) { rstsr_c32 v; v.re = re; v.im = im; return v; }
__device__ rstsr_c64 make_c64(double re, double im) { rstsr_c64 v; v.re = re; v.im = im; return v; }
__device__ int rstsr_isnan_f(float x) { return x != x; }
__device__ int rstsr_isnan_d(double x) { return x != x; }
#define rstsr_isnan(x) _Generic((x), float: rstsr_isnan_f, double: rstsr_isnan_d, default: (int)(x != x))
"#;

/// Bump when generated-source semantics or the disk payload change.
const FORMAT_VERSION: u32 = 3;

/// Cache key of one compiled kernel specialization.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct KernelKey {
    /// Kernel entry-point name.
    pub name: String,
    /// Full generated source (prelude + kernel).
    pub source: String,
    /// `compute_<major><minor>` arch string.
    pub arch: String,
    /// Compile options as passed to NVRTC (joined, order-fixed).
    pub options: String,
}

impl KernelKey {
    fn disk_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        FORMAT_VERSION.hash(&mut h);
        self.source.hash(&mut h);
        self.arch.hash(&mut h);
        self.options.hash(&mut h);
        h.finish()
    }
}

/// Library-fixed compile options (D3). `arch` enters separately.
fn compile_options(extra: &[&str]) -> Vec<String> {
    let mut opts: Vec<String> = vec!["--std=c++17".into()];
    opts.extend(extra.iter().map(|s| s.to_string()));
    #[cfg(debug_assertions)]
    opts.push("-lineinfo".into());
    opts
}

fn cache_dir() -> Option<PathBuf> {
    if let Some(v) = std::env::var_os("RSTSR_CUDA_CACHE") {
        if v == "off" {
            return None;
        }
    }
    if let Some(d) = std::env::var_os("RSTSR_CUDA_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("rstsr").join("cuda"))
}

/// Per-ordinal compiled-kernel cache (D4). Lives in `CudaOrdinalObj`'s
/// `OnceLock` slot; `pub(crate)` for the field type only.
#[derive(Default)]
pub(crate) struct KernelCachePublic {
    pub(crate) entries: Mutex<HashMap<KernelKey, Arc<CudaFunction>>>,
}
type KernelCache = KernelCachePublic;

impl CudaOrdinalObj {
    fn kernel_cache(&self) -> &KernelCache {
        self.kernel_cache_slot.get_or_init(KernelCache::default)
    }

    /// Compile (or fetch) the function for `key` on this ordinal.
    pub(crate) fn kernel_function(&self, key: &KernelKey) -> Result<Arc<CudaFunction>> {
        if let Some(f) = {
            let cache = self.kernel_cache();
            let entries = cache.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.get(key).cloned()
        } {
            return Ok(f);
        }
        let ptx_src = self.load_or_compile_ptx(key)?;
        let module = self
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_src))
            .map_err(map_driver_error)?;
        let function = Arc::new(module.load_function(&key.name).map_err(map_driver_error)?);
        let cache = self.kernel_cache();
        let mut entries = cache.entries.lock().unwrap_or_else(|e| e.into_inner());
        // prefer a racing thread's entry over ours
        Ok(entries.entry(key.clone()).or_insert(function).clone())
    }

    /// Disk-cache lookup (PTX text), else NVRTC compile; stores on miss.
    fn load_or_compile_ptx(&self, key: &KernelKey) -> Result<String> {
        let hash = key.disk_hash();
        if let Some(dir) = cache_dir() {
            let path = dir.join(format!("k_{hash:016x}.ptx"));
            if let Ok(src) = std::fs::read_to_string(&path) {
                return Ok(src);
            }
            let src = self.compile_now(key)?;
            let _ = std::fs::create_dir_all(&dir);
            let tmp = dir.join(format!(".k_{hash:016x}.tmp{}", std::process::id()));
            if std::fs::write(&tmp, &src).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
            return Ok(src);
        }
        self.compile_now(key)
    }

    fn compile_now(&self, key: &KernelKey) -> Result<String> {
        let options: Vec<String> = key.options.split('\u{1}').map(|s| s.to_string()).collect();
        let arch: &'static str = Box::leak(key.arch.clone().into_boxed_str());
        let opts = cudarc::nvrtc::CompileOptions { arch: Some(arch), options, ..Default::default() };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&key.source, opts)
            .map_err(|e| rstsr_error!(DeviceError, "NVRTC compile failed for kernel `{}`: {e:?}", key.name))?;
        Ok(ptx.to_src())
    }

    /// `compute_<major><minor>` arch string of this ordinal (D3); cached
    /// after the first driver query (launch-path hot).
    pub(crate) fn arch_string(&self) -> Result<String> {
        let cached = self.arch_slot.get_or_init(|| {
            self.compute_capability()
                .map(|(maj, min)| format!("compute_{maj}{min}"))
        });
        match cached {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(rstsr_error!(DeviceError, "arch query failed: {e}")),
        }
    }
}

/// Build a key for `name` compiled from `source` on `ordinal`. Memoized by
/// hashing (source, arch, options) — sources are static strings per
/// specialization, so repeat calls skip the String rebuild via a cheap hash
/// lookup (launch-path hot). The full source still lives in the key.
pub(crate) fn make_key(ordinal: &CudaOrdinalObj, name: &str, source: &str, extra: &[&str]) -> Result<KernelKey> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Mutex;
    thread_local! {
        static KEY_CACHE: Mutex<HashMap<(u64, KernelKey), ()>> = Mutex::new(HashMap::new());
    }
    let arch = ordinal.arch_string()?;
    let options = compile_options(extra).join("\u{1}");
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    arch.hash(&mut h);
    options.hash(&mut h);
    let digest = h.finish();
    let key = KernelKey {
        name: name.to_string(),
        source: source.to_string(),
        arch,
        options,
    };
    // memo: skip the clone if this exact key was built before (rare re-use
    // of an Arc'd CudaFunction still happens through the ordinal cache; this
    // memo only avoids re-allocating key strings)
    KEY_CACHE.with(|m| {
        let mut guard = m.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert((digest, key.clone()), ());
    });
    Ok(key)
}

/// Grid-stride launch geometry over `n` elements (256-thread blocks).
const BLOCK: u32 = 256;

/// Raw byte-wise launcher (D2): owns the param storage for one launch.
pub(crate) struct Launch<'a> {
    stream: &'a Arc<CudaStreamObj>,
    function: &'a CudaFunction,
    n: usize,
    params: Vec<Vec<u8>>,
}

impl<'a> Launch<'a> {
    pub fn new(stream: &'a Arc<CudaStreamObj>, function: &'a CudaFunction, n: usize) -> Self {
        Self { stream, function, n, params: Vec::with_capacity(8) }
    }

    /// Push a raw byte scalar (fixed-size `Pod`-like value).
    pub fn arg_bytes(mut self, bytes: &[u8]) -> Self {
        self.params.push(bytes.to_vec());
        self
    }

    pub fn arg_u64(self, v: u64) -> Self {
        self.arg_bytes(&v.to_ne_bytes())
    }

    pub fn arg_usize(self, v: usize) -> Self {
        self.arg_u64(v as u64)
    }

    /// Base device pointer of `buf` (null for empty buffers).
    pub fn arg_buf<T>(self, buf: &CudaBuffer<T>) -> Self {
        let ptr = if buf.is_empty() { 0 } else { buf.device_ptr() as u64 };
        self.arg_u64(ptr)
    }

    /// Enqueue the kernel (async; host sync happens only at host reads).
    pub fn launch(mut self) -> Result<()> {
        let grid = self.n.div_ceil(BLOCK as usize) as u32;
        if grid == 0 {
            return Ok(());
        }
        let mut ptrs: Vec<*mut core::ffi::c_void> =
            self.params.iter_mut().map(|p| p.as_mut_ptr() as *mut core::ffi::c_void).collect();
        // SAFETY: live function/stream handles; params outlive the enqueue;
        // same-context launch (kernels are looked up through this ordinal).
        unsafe {
            self.stream.inner().context().bind_to_thread().map_err(map_driver_error)?;
            map_driver_result(dr::launch_kernel(
                self.function.cu_function(),
                (grid, 1, 1),
                (BLOCK, 1, 1),
                0,
                self.stream.inner().cu_stream(),
                &mut ptrs,
            ))?;
        }
        Ok(())
    }
}

/// Blocking-sync event wrapper for cross-stream ordering (D8).
pub(crate) struct Event(CUevent);

impl Event {
    pub fn create() -> Result<Self> {
        // DISABLE_TIMING (2) | BLOCKING_SYNC (1); the enum has no BitOr.
        let flags = 2u32 | 1u32;
        let e = dr::event::create(unsafe { core::mem::transmute::<u32, CUevent_flags>(flags) })
            .map_err(map_driver_error)?;
        Ok(Self(e))
    }

    /// Record on `stream`, then make `wait_stream` wait for it.
    ///
    /// # Safety
    /// Both streams must belong to the same context.
    pub unsafe fn order(&self, stream: cudarc::driver::sys::CUstream, wait_stream: cudarc::driver::sys::CUstream) -> Result<()> {
        map_driver_result(dr::event::record(self.0, stream))?;
        map_driver_result(dr::stream::wait_event(wait_stream, self.0, CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // a leaked event is harmless; destruction failure unrecoverable
        unsafe { dr::event::destroy(self.0).ok() };
    }
}

/// Order `buf`'s allocation stream before execution on `exec` (no-op when
/// they match). Returns the event guard; drop after the launch is enqueued.
pub(crate) fn order_input<T>(buf: &CudaBuffer<T>, exec: &Arc<CudaStreamObj>) -> Result<Option<Event>> {
    let alloc_stream = buf.as_bytes().stream();
    if alloc_stream.cu_stream() == exec.inner().cu_stream() {
        return Ok(None);
    }
    let event = Event::create()?;
    // SAFETY: both streams belong to this ordinal's context (buffers and
    // device values of one ordinal never mix contexts).
    unsafe { event.order(alloc_stream.cu_stream(), exec.inner().cu_stream())? };
    Ok(Some(event))
}

/// The shared T0 index-decomposition fragment (rank-generic), part of
/// every kernel source.
/// `lin` is enumerated over the OUTPUT's row-major order: decompose by
/// `oshape`, scale by the operand's strides (0 = broadcast axis).
const INDEX_HELPER: &str = r#"
__device__ rstsr_usz rstsr_index_to_offset(rstsr_usz lin, int rank,
    const rstsr_usz* oshape, const rstsr_usz* strides, rstsr_usz base_offset) {
    rstsr_usz off = base_offset;
    for (int ax = rank - 1; ax >= 0; --ax) {
        off += (lin % oshape[ax]) * strides[ax];
        lin /= oshape[ax];
    }
    return off;
}
"#;

/// Build the full source: prelude + dtype includes + index helper + kernel
/// body.
pub(crate) fn assemble_source(includes: &[&str], body: &str) -> String {
    let mut s = String::from(CUDA_PRELUDE);
    for inc in includes {
        let _ = writeln!(s, "{inc}");
    }
    let _ = write!(s, "\n{INDEX_HELPER}");
    let _ = write!(s, "\n{body}");
    s
}
