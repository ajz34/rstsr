//! Kernel compilation, caching, and launch.
//!
//! Sources are generated CUDA C; compilation goes through NVRTC (PTX)
//! with a runtime-derived arch, and compiled functions are cached per
//! ordinal in memory and on disk. `FlatLaunch` is the flat
//! one-element-per-thread launcher for elementwise/copy/gather/sequence
//! kernels; other kernel classes (reductions, GEMM, scans) get their own
//! geometry. Marshaling is plain bytes, so no cudarc dtype trait is on
//! our path.

use crate::buffer::CudaBuffer;
use crate::error::{map_driver_error, map_driver_result};
use crate::registry::CudaOrdinalObj;
use crate::stream::CudaStreamObj;
use rstsr_core::prelude_dev::*;

use cudarc::driver::sys::{CUevent, CUevent_flags, CUevent_wait_flags};
use cudarc::driver::{result as dr, CudaFunction, CudaStream};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// Shared CUDA C prelude: fixed-width names, complex pairs, IEEE helpers.
pub(crate) const CUDA_PRELUDE: &str = r#"
typedef signed char rstsr_i8; typedef short rstsr_i16; typedef int rstsr_i32;
typedef long long rstsr_i64; typedef long long rstsr_isz;
typedef unsigned char rstsr_u8; typedef unsigned short rstsr_u16; typedef unsigned int rstsr_u32;
typedef unsigned long long rstsr_u64; typedef unsigned long long rstsr_usz;
// One template serves both complex widths; the aliases keep the CNAME
// strings (and cache keys) stable. Layout {re, im} is byte-identical to
// the previous plain structs.
template <typename T>
struct rstsr_complex {
    T re, im;
    __device__ rstsr_complex() {}
    __device__ rstsr_complex(T r, T i) : re(r), im(i) {}
};
template <typename T>
__device__ rstsr_complex<T> rstsr_make_complex(T re, T im) { return rstsr_complex<T>(re, im); }
using rstsr_c32 = rstsr_complex<float>;
using rstsr_c64 = rstsr_complex<double>;
"#;

/// Bump when generated-source semantics or the disk payload change.
const FORMAT_VERSION: u32 = 4;

/// Math helpers shared by elementwise kernels: NumPy-parity remainder
/// (CPU `ExtReal::ext_rem` semantics) and shift-square integer pow
/// (CPU `ExtNum::ext_pow` for ints; the device op rejects negative
/// exponents before the launch).
pub(crate) const MATH_HELPER: &str = r#"
// NVRTC provides the isnan/isinf/signbit/copysign overloads implicitly
// (cuda_fp16.h-style device intrinsics); no host math headers are on the
// include path, so NaN is a bit-pattern constant (Rust's canonical quiet
// NaN, matching the CPU devices).
__device__ float rstsr_nanf() { return __int_as_float(0x7fc00000); }

// Either-operand NaN propagation and the zero tie-break of the CPU
// ext_max/ext_min (and NumPy): the second operand is returned when the
// values compare equal, which fixes the sign of a 0.0/-0.0 tie. The plain
// select chain (no PTX max.NaN, which needs sm_80 and canonicalizes the
// payload) keeps NaN payloads and runs on every arch.
__device__ double rstsr_maxnand(double x, double y) {
    return (isnan(x) ? x : (isnan(y) ? y : (x > y ? x : y)));
}
__device__ double rstsr_minnand(double x, double y) {
    return (isnan(x) ? x : (isnan(y) ? y : (x < y ? x : y)));
}
__device__ float rstsr_maxnanf(float x, float y) {
    return (isnan(x) ? x : (isnan(y) ? y : (x > y ? x : y)));
}
__device__ float rstsr_minnanf(float x, float y) {
    return (isnan(x) ? x : (isnan(y) ? y : (x < y ? x : y)));
}
__device__ double rstsr_nan() { return __longlong_as_double(0x7ff8000000000000LL); }

__device__ float rstsr_fremf(float x, float y) {
    // CPU ExtReal::ext_rem: NaN propagation, inf-dividend NaN, inf-divisor
    // sign rule, divisor-signed zero on exact multiples, else fmod
    // adjusted toward the divisor's sign.
    if (isnan(x) || isnan(y)) return rstsr_nanf();
    if (isinf(x)) return rstsr_nanf();
    if (isinf(y)) {
        if (x == 0.0f) return copysignf(0.0f, y);
        return (signbit(x) == signbit(y)) ? x : y;
    }
    if (y == 0.0f) return rstsr_nanf();
    float r = fmodf(x, y);
    if (r == 0.0f) return copysignf(0.0f, y);
    if (signbit(r) != signbit(y)) r += y;
    return r;
}

__device__ double rstsr_frem(double x, double y) {
    if (isnan(x) || isnan(y)) return rstsr_nan();
    if (isinf(x)) return rstsr_nan();
    if (isinf(y)) {
        if (x == 0.0) return copysign(0.0, y);
        return (signbit(x) == signbit(y)) ? x : y;
    }
    if (y == 0.0) return rstsr_nan();
    double r = fmod(x, y);
    if (r == 0.0) return copysign(0.0, y);
    if (signbit(r) != signbit(y)) r += y;
    return r;
}

__device__ long long rstsr_ipow(long long base, long long exp_in) {
    // Exponent width follows the CPU ext_pow (`other as u32`); the device
    // op rejects negative exponents on the host. Unsigned accumulation
    // makes the wrapping multiplication well defined (Rust release-mode
    // wrap; the store truncates to the output width).
    unsigned int exp = (unsigned int)exp_in;
    unsigned long long b = (unsigned long long)base;
    unsigned long long result = 1ULL;
    while (exp > 0) {
        if (exp & 1U) result *= b;
        b *= b;
        exp >>= 1;
    }
    return (long long)result;
}
"#;

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

/// NVRTC search path for the toolkit headers our dtype includes need
/// (`cuda_fp16.h` / `cuda_bf16.h`; NVRTC has no builtin copy). Resolved
/// once: the paths enter the cache key, so the lookup must also be cheap.
fn nvrtc_include_dirs() -> &'static [String] {
    static DIRS: OnceLock<Vec<String>> = OnceLock::new();
    DIRS.get_or_init(|| {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(p) = std::env::var_os("RSTSR_CUDA_INCLUDE") {
            candidates.push(PathBuf::from(p));
        }
        for key in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"] {
            if let Some(p) = std::env::var_os(key) {
                candidates.push(PathBuf::from(p).join("include"));
            }
        }
        candidates.push(PathBuf::from("/usr/local/cuda/include"));
        candidates.push(PathBuf::from("/usr/include"));
        candidates
            .into_iter()
            .filter(|d| d.join("cuda_fp16.h").is_file())
            .map(|d| d.to_string_lossy().into_owned())
            .collect()
    })
}

/// `RSTSR_CUDA_STRICT_FP=1`: forbid FMA contraction (`-fmad=true` is NVRTC's **default**).
///
/// The default is what makes `l2_norm`/`var`/`vecdot` differ from the CPU in the last bits: the
/// CPU computes `fl(x*x)` and then `fl(a + that)`, a contracted kernel computes `fma(x, x, a)`, and
/// one rounding is not two. That is a legitimate speed/parity trade — FMA is free throughput on
/// every NVIDIA part — but it should be a decision the caller can reverse rather than a property
/// discovered by comparing results.
///
/// Note what this cannot fix: the reduction *tree* still adds in a different order than a
/// sequential loop, and transcendentals are a different implementation (`sinf` is NVIDIA's, not
/// Rust's). This knob removes exactly one of the three differences — see the crate docs.
fn strict_fp() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| matches!(std::env::var("RSTSR_CUDA_STRICT_FP").as_deref(), Ok("1") | Ok("true")))
}

/// Library-fixed compile options. `arch` enters separately.
fn compile_options(extra: &[&str]) -> Vec<String> {
    let mut opts: Vec<String> = vec!["--std=c++17".into()];
    opts.extend(nvrtc_include_dirs().iter().map(|d| format!("--include-path={d}")));
    opts.extend(extra.iter().map(|s| s.to_string()));
    if strict_fp() {
        opts.push("--fmad=false".into());
    }
    #[cfg(debug_assertions)]
    opts.push("-lineinfo".into());
    opts
}

/// `RSTSR_CUDA_ARCH=compute_80`: compile for an arch other than the card's own.
///
/// Diagnostic only — it does not make a card run code it cannot. What it does is let **one** GPU
/// stand in for the older ones, because PTX is forward-compatible: an sm_86 driver will JIT
/// `compute_75`/`compute_80`/`compute_86` PTX, so a single A100 can be made to accept every arch at
/// or below its own. That turns "we validated one arch" into "we validated the ones one card can
/// host", which is the difference between an unknown and a known.
///
/// It must be read before anything is compiled, so it is a `OnceLock`: a process compiles for one
/// arch. A malformed value is ignored with a warning rather than silently forcing something odd —
/// a typo here would otherwise look like a driver bug.
fn arch_override() -> Option<&'static str> {
    static OV: OnceLock<Option<&'static str>> = OnceLock::new();
    *OV.get_or_init(|| {
        let raw = std::env::var("RSTSR_CUDA_ARCH").ok()?;
        match parse_arch_override(&raw) {
            // `Box::leak` because the value outlives this closure (an `OnceLock` payload); it is a
            // handful of bytes, allocated at most once per process.
            Some(_) => Some(Box::leak(raw.into_boxed_str())),
            None => {
                eprintln!(
                    "rstsr-cuda: ignoring RSTSR_CUDA_ARCH={raw:?}; expected e.g. \"compute_80\" \
                     (this knob exists so one GPU can be made to compile for older arches)"
                );
                None
            },
        }
    })
}

/// `"compute_80"` -> `Some("compute_80")`; anything else -> `None`.
///
/// Split out from the env read so the rule is testable: the read happens once per process, the
/// validation should not be untested because of that.
pub(crate) fn parse_arch_override(raw: &str) -> Option<&str> {
    let digits = raw.strip_prefix("compute_")?;
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(raw)
}

/// `compute_80` -> `80`, for the "is this arch even hostable here" check.
fn arch_number(arch: &str) -> Option<u32> {
    arch.strip_prefix("compute_")?.parse().ok()
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

/// Per-ordinal compiled-kernel cache. Lives in `CudaOrdinalObj`'s
/// `OnceLock` slot; `pub(crate)` for the field type only.
#[derive(Default)]
pub(crate) struct KernelCachePublic {
    pub(crate) entries: Mutex<HashMap<KernelKey, Arc<CudaFunction>>>,
}
type KernelCache = KernelCachePublic;

/// Per-ordinal intern map of the *planned* elementwise functions, keyed by
/// the specialization discriminator (entry + includes + substitutions).
/// Lives on the ordinal object so it shares the kernel cache's lifetime
/// (no process-global pinning of a retired context's modules).
#[derive(Default)]
pub(crate) struct PlannedCache {
    pub(crate) entries: Mutex<HashMap<String, Arc<CudaFunction>>>,
}

impl CudaOrdinalObj {
    fn kernel_cache(&self) -> &KernelCache {
        self.kernel_cache_slot.get_or_init(KernelCache::default)
    }

    pub(crate) fn planned_cache(&self) -> &PlannedCache {
        self.planned_cache_slot.get_or_init(PlannedCache::default)
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
        let module = self.context().load_module(cudarc::nvrtc::Ptx::from_src(ptx_src)).map_err(map_driver_error)?;
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
        let arch = arch_static(&key.arch);
        let opts = cudarc::nvrtc::CompileOptions { arch: Some(arch), options, ..Default::default() };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&key.source, opts)
            .map_err(|e| rstsr_error!(DeviceError, "NVRTC compile failed for kernel `{}`: {e:?}", key.name))?;
        Ok(ptx.to_src())
    }

    /// `compute_<major><minor>` arch string of this ordinal; cached
    /// after the first driver query (launch-path hot).
    pub(crate) fn arch_string(&self) -> Result<String> {
        // One choke point: the returned string becomes the NVRTC flag, part of the key, and part of
        // the disk-cache hash, so overriding it here cannot leave the three out of step.
        if let Some(a) = arch_override() {
            if let (Some(want), Ok((maj, min))) = (arch_number(a), self.compute_capability()) {
                if want > (maj as u32) * 10 + (min as u32) {
                    eprintln!(
                        "rstsr-cuda: compiling for {a} on a compute_{maj}{min} device; PTX is only \
                         forward-compatible, so loading it will fail. Use an arch at or below the card's."
                    );
                }
            }
            return Ok(a.to_string());
        }
        let cached =
            self.arch_slot.get_or_init(|| self.compute_capability().map(|(maj, min)| format!("compute_{maj}{min}")));
        match cached {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(rstsr_error!(DeviceError, "arch query failed: {e}")),
        }
    }
}

/// Build a key for `name` compiled from `source` on `ordinal`. Callers
/// with static source pass a cached assembled string (see
/// `kernels_ops::*_assembled`); dtype-specialized callers build it per call.
pub(crate) fn make_key(ordinal: &CudaOrdinalObj, name: &str, source: &str, extra: &[&str]) -> Result<KernelKey> {
    // `name` is the symbol `kernel_function` will look up in the module, so it has to appear in the
    // source as a definition. Checked here because the alternative is discovering the mismatch as
    // `CUDA_ERROR_NOT_FOUND, "named symbol not found"` on hardware, which names neither the kernel
    // nor the caller: it cost a GPU round trip when the scan kernel was registered as `cumsum`
    // instead of `rstsr_cumsum`. The source is right here, so the check is local and permanent
    // rather than a `debug_assert!` that release builds skip.
    //
    // This is also the answer to "is the cache-key namespace complete?" for the *name* half: a key
    // can no longer disagree with the entry it will load.
    match source.find(&format!("void {name}(")) {
        Some(_) => {}
        None => rstsr_raise!(
            DeviceError,
            "kernel `{name}` is not defined as `void {name}(...)` in the source it was keyed with; \
             the launch would fail with `named symbol not found`"
        )?,
    }
    let arch = ordinal.arch_string()?;
    let options = compile_options(extra).join("\u{1}");
    Ok(KernelKey { name: name.to_string(), source: source.to_string(), arch, options })
}

/// Intern one `&'static str` per distinct arch (the safe NVRTC layer
/// takes `&'static str`; the set is bounded by the visible devices).
fn arch_static(arch: &str) -> &'static str {
    static ARCHES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut guard = ARCHES.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = guard.get(arch) {
        return s;
    }
    let leaked: &'static str = Box::leak(arch.to_string().into_boxed_str());
    guard.insert(arch.to_string(), leaked);
    leaked
}

/// Flat launch geometry: one element per thread, 256-thread blocks (the
/// kernel guards the tail). 256 divides every SM's thread capacity evenly
/// and is supported by all CUDA GPUs; per-arch tuning is deferred to the
/// future vectorized/multi-element kernels.
const BLOCK: u32 = 256;

/// Flat 1-D launcher: one element per thread, `ceil(n / BLOCK)`
/// blocks, no shared memory; owns the param storage for one launch.
/// Elementwise/gather/sequence kernels use this; other kernel classes
/// (reductions, GEMM, scans) get their own launch geometry.
/// Explicit grid/block dims for a launch (tiled 2-D kernels).
pub(crate) type LaunchShape = ((u32, u32, u32), (u32, u32, u32));

pub(crate) struct FlatLaunch<'a> {
    stream: &'a Arc<CudaStreamObj>,
    function: &'a CudaFunction,
    n: usize,
    elems_per_thread: u32,
    /// Explicit `(grid, block)` overrides for non-1-D geometries (tiled
    /// kernels); when set, `n`/`elems_per_thread` are ignored.
    explicit_shape: Option<LaunchShape>,
    params: Vec<Vec<u8>>,
    /// Read-event guards of every buffer argument; dropped after the launch,
    /// so each read event covers the consuming kernel and the allocation
    /// stream's free waits for it (foreign-stream read lifetime).
    reads: Vec<cudarc::driver::SyncOnDrop<'a>>,
}

impl<'a> FlatLaunch<'a> {
    pub fn new(stream: &'a Arc<CudaStreamObj>, function: &'a CudaFunction, n: usize) -> Self {
        Self { stream, function, n, elems_per_thread: 1, explicit_shape: None, params: Vec::with_capacity(8), reads: Vec::new() }
    }

    /// Grid covers `elems_per_thread` consecutive elements per thread
    /// (flat ILP entries); `n` stays the launch element count.
    pub fn elems_per_thread(mut self, ept: u32) -> Self {
        self.elems_per_thread = ept;
        self
    }

    /// Launch with an explicit grid/block shape (tiled 2-D kernels).
    pub fn explicit_shape(mut self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        self.explicit_shape = Some((grid, block));
        self
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

    /// Push a 32-bit `int`/`unsigned` kernel parameter (exact width).
    pub fn arg_i32(self, v: i32) -> Self {
        self.arg_bytes(&v.to_ne_bytes())
    }

    /// Base device pointer of `buf` (null for empty buffers). The read
    /// event is recorded after [`Self::launch`] enqueues the kernel, so the
    /// buffer's allocation stream waits for this launch before its free.
    pub fn arg_buf<'b, T>(mut self, buf: &'b CudaBuffer<T>) -> Self
    where
        'b: 'a,
    {
        if buf.is_empty() {
            return self.arg_u64(0);
        }
        let (ptr, guard) = buf.read_ptr_on(self.stream);
        self.reads.push(guard);
        self.arg_u64(ptr as u64)
    }

    /// `arg_buf` folded with an element-unit offset (the common planned-kernel
    /// pointer form).
    pub fn arg_ptr<'b, T>(self, buf: &'b CudaBuffer<T>, offset_elems: usize) -> Self
    where
        'b: 'a,
    {
        if buf.is_empty() {
            return self.arg_u64(0);
        }
        let (ptr, guard) = buf.read_ptr_on(self.stream);
        let mut this = self;
        this.reads.push(guard);
        this.arg_u64((ptr as u64).wrapping_add(offset_elems as u64 * core::mem::size_of::<T>() as u64))
    }

    /// `arg_ptr` for an optional buffer operand (null when absent).
    pub fn arg_ptr_opt<'b, T>(self, buf: Option<&'b CudaBuffer<T>>, offset_elems: usize) -> Self
    where
        'b: 'a,
    {
        match buf {
            Some(buf) => self.arg_ptr(buf, offset_elems),
            None => self.arg_u64(0),
        }
    }

    /// Enqueue the kernel (async; host sync happens only at host reads).
    pub fn launch(mut self) -> Result<()> {
        let (grid, block): LaunchShape = if let Some(sh) = self.explicit_shape {
            sh
        } else {
            let work = (BLOCK as usize) * self.elems_per_thread as usize;
            ((self.n.div_ceil(work) as u32, 1, 1), (BLOCK, 1, 1))
        };
        if grid.0 == 0 {
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
                grid,
                block,
                0,
                self.stream.inner().cu_stream(),
                &mut ptrs,
            ))?;
        }
        // the enqueued kernel is covered by every read event from here on
        drop(self.reads);
        Ok(())
    }
}

/// Event wrapper for cross-stream ordering.
pub(crate) struct Event(CUevent);

impl Event {
    pub fn create() -> Result<Self> {
        // no host-side synchronize on this event, so BLOCKING_SYNC is not
        // needed; DISABLE_TIMING alone is a valid enum variant
        let e = dr::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING).map_err(map_driver_error)?;
        Ok(Self(e))
    }

    /// Record on `stream`, then make `wait_stream` wait for it.
    ///
    /// # Safety
    /// Both streams must belong to the same context.
    pub unsafe fn order(
        &self,
        stream: cudarc::driver::sys::CUstream,
        wait_stream: cudarc::driver::sys::CUstream,
    ) -> Result<()> {
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
    order_input_raw(buf, exec.inner())
}

/// `order_input` over a raw cudarc stream (for paths that only hold the
/// buffer's own stream, e.g. the memcpy fast path).
pub(crate) fn order_input_raw<T>(buf: &CudaBuffer<T>, exec: &Arc<CudaStream>) -> Result<Option<Event>> {
    let alloc_stream = buf.as_bytes().stream();
    if alloc_stream.cu_stream() == exec.cu_stream() {
        return Ok(None);
    }
    let event = Event::create()?;
    // SAFETY: both streams belong to this ordinal's context (buffers and
    // device values of one ordinal never mix contexts).
    unsafe { event.order(alloc_stream.cu_stream(), exec.cu_stream())? };
    Ok(Some(event))
}

/// Order execution on `exec` before later work on `buf`'s allocation stream
/// (no-op when they match), the mirror of [`order_input`]. Required after an
/// op writes into an existing buffer: host reads only synchronize that
/// buffer's allocation stream. Call it after the op is enqueued: the event
/// captures everything enqueued so far.
pub(crate) fn order_output<T>(buf: &CudaBuffer<T>, exec: &Arc<CudaStreamObj>) -> Result<Option<Event>> {
    order_output_raw(buf, exec.inner())
}

/// `order_output` over a raw cudarc stream.
pub(crate) fn order_output_raw<T>(buf: &CudaBuffer<T>, exec: &Arc<CudaStream>) -> Result<Option<Event>> {
    let alloc_stream = buf.as_bytes().stream();
    if alloc_stream.cu_stream() == exec.cu_stream() {
        return Ok(None);
    }
    let event = Event::create()?;
    // SAFETY: both streams belong to this ordinal's context.
    unsafe { event.order(exec.cu_stream(), alloc_stream.cu_stream())? };
    Ok(Some(event))
}

/// The shared T0 index-decomposition fragments (rank-generic), part of
/// every kernel source.
///
/// `rstsr_index_to_offset` decomposes `lin` by one operand's shape in
/// row-major order (elementwise kernels: both operands share the output's
/// logical shape, strides 0 = broadcast axis).
/// `rstsr_iter_offset` decomposes `lin` by an operand's *own* shape in the
/// device's order; assign kernels use it for each side separately, which
/// also covers rank/shape-changing assigns (positional zip in iteration
/// order, matching the CPU devices).
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

__device__ rstsr_usz rstsr_iter_offset(rstsr_usz lin, int rank,
    const rstsr_usz* shape, const rstsr_usz* strides, rstsr_usz base_offset,
    int col_major) {
    rstsr_usz off = base_offset;
    if (col_major) {
        for (int ax = 0; ax < rank; ++ax) {
            off += (lin % shape[ax]) * strides[ax];
            lin /= shape[ax];
        }
    } else {
        for (int ax = rank - 1; ax >= 0; --ax) {
            off += (lin % shape[ax]) * strides[ax];
            lin /= shape[ax];
        }
    }
    return off;
}
"#;

/// Build the full source: prelude + dtype includes + index/math helpers +
/// kernel body.
pub(crate) fn assemble_source(includes: &[&str], body: &str) -> String {
    let mut s = String::from(CUDA_PRELUDE);
    s.push_str(&crate::kernels::splice_includes(includes));
    let _ = write!(s, "\n{INDEX_HELPER}");
    let _ = write!(s, "\n{MATH_HELPER}");
    let _ = write!(s, "\n{body}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The env read happens once per process, so the *rule* is what gets tested; without this split
    /// the only untested part of the arch override would be the part that decides what it means.
    #[test]
    fn arch_override_accepts_only_compute_strings() {
        assert_eq!(parse_arch_override("compute_80"), Some("compute_80"));
        assert_eq!(parse_arch_override("compute_90"), Some("compute_90"));
        assert_eq!(parse_arch_override("compute_120"), Some("compute_120"));
        // Anything else must be refused, because a silently-forced bogus arch looks like a driver bug.
        for bad in ["", "compute_", "compute_x", "sm_80", "compute_8o", " compute_80", "compute_-1"] {
            assert_eq!(parse_arch_override(bad), None, "{bad:?} should be refused");
        }
    }

    /// The parse that decides whether an override is even hostable: `compute_86` is `86`, and the
    /// comparison against `maj * 10 + min` is what produces the "will fail to load" warning.
    #[test]
    fn arch_numbers_compare_like_capabilities() {
        assert_eq!(arch_number("compute_80"), Some(80));
        assert_eq!(arch_number("compute_120"), Some(120));
        assert!(arch_number("compute_86") > arch_number("compute_80"));
        assert!(arch_number("compute_86") < arch_number("compute_90"));
        assert_eq!(arch_number("compute_x"), None);
    }
}
