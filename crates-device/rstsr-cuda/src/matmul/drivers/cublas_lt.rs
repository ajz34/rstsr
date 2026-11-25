//! The cuBLASLt driver: `cublasLtMatmul`.
//!
//! The same column-major mapping as the other cuBLAS drivers (it shares [`cublas::prepare`]), but
//! the problem is described by descriptors: a matmul descriptor, four matrix layouts and a
//! heuristic preference, and the kernel is chosen by `cublasLtMatmulAlgoGetHeuristic`.
//!
//! # Resources
//!
//! [`CublasLtHandle`] is the per-ordinal state: one `cublasLtHandle_t`, the lock held across every
//! call on it, and the heuristic cache. A problem is built and asked for its algorithm once per
//! cache key, and the key holds every descriptor field plus the pointer alignment classes (an
//! algorithm states a required alignment per matrix, which `cublasLtMatmul` checks). The workspace
//! is one 32 MiB buffer per **stream**
//! ([`CudaStreamObj::cublaslt_workspace`][crate::stream::CudaStreamObj]): it is the
//! kernels' scratch memory, so two streams of one ordinal never share it.
//!
//! Every 32F mode is served, `Fast16BF` included (a compute type, where the classic entries only
//! have a handle math mode); of the 64F family `Default` and `Pedantic`, the fixed-point emulation
//! being left to the classic driver.

use core::ffi::c_void;
use cudarc::cublas::sys as cb;
use cudarc::cublaslt::sys as lt;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::cublas::{GemmCall, Served, prepare};
use super::cublas_ex::ExScalar;
use super::{CublasPlan, Problem, Refusal, Verdict};
use crate::matmul::cublas_args::CublasArgs;
use crate::prelude_dev::*;

/// The cuBLASLt workspace, per stream.
pub(crate) const WORKSPACE_BYTES: usize = 32 << 20;

/// Entries kept before the heuristic cache is dropped whole.
const CACHE_MAX: usize = 1024;

/// The same enum in the `cublaslt` bindings: the `cublas` and `cublaslt` modules are generated from
/// one header and differ only in name.
///
/// # Safety
/// `A` and `B` must be the same `#[repr(u32)]` enum.
unsafe fn retyped<A: Copy, B: Copy>(x: A) -> B {
    assert_eq!(core::mem::size_of::<A>(), core::mem::size_of::<B>());
    unsafe { core::mem::transmute_copy(&x) }
}

fn succeeded(status: lt::cublasStatus_t) -> bool {
    status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

/// One matrix layout as the cache sees it: every field of the descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayoutKey {
    dtype: u32,
    rows: u64,
    cols: u64,
    ld: i64,
    batch: i32,
    stride: i64,
}

/// The stored (column-major) matrix of one cuBLAS slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stored {
    pub rows: u64,
    pub cols: u64,
    pub ld: i64,
}

/// The stored matrices of cuBLAS's slots `A`, `B` and `C`/`D` for these arguments.
///
/// `op(A)` is `m x k`: the stored matrix is `m x k` under `OP_N` and `k x m` under `OP_T`; `op(B)`
/// is `k x n`: stored `k x n` or `n x k`; `C` and `D` are `m x n`.
pub(crate) fn stored_matrices(a: &CublasArgs) -> [Stored; 3] {
    let (m, n, k) = (a.mo as u64, a.no as u64, a.ko as u64);
    let (lda, ldb, ldc) = (a.lda as i64, a.ldb as i64, a.ldc as i64);
    let a_stored = if a.transa { Stored { rows: k, cols: m, ld: lda } } else { Stored { rows: m, cols: k, ld: lda } };
    let b_stored = if a.transb { Stored { rows: n, cols: k, ld: ldb } } else { Stored { rows: k, cols: n, ld: ldb } };
    [a_stored, b_stored, Stored { rows: m, cols: n, ld: ldc }]
}

/// The full problem as the heuristic sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    compute: u32,
    scale: u32,
    trans: [bool; 2],
    /// The layouts of the slots `A`, `B`, `C`, `D` (`D` equals `C`).
    layouts: [LayoutKey; 4],
    workspace: u64,
    /// The low address bits of the pointers `A`, `B`, `C`, `D`: an algorithm's alignment
    /// requirement is per matrix, so it is chosen and cached per alignment class.
    align: [u8; 4],
}

/// The low 8 address bits of a pointer: the alignment class the algorithm is checked for.
fn align(ptr: u64) -> u8 {
    (ptr & 0xff) as u8
}

/// The layouts of the three stored matrices; batch count and batch stride (in **elements**, like
/// the strided-batched entries) follow [`crate::matmul::cublas_args::batch_strides`], slot by slot.
fn layout_keys(a: &CublasArgs, dtype: u32) -> [LayoutKey; 3] {
    let (stride_a, stride_b, stride_c) = crate::matmul::cublas_args::batch_strides(a);
    let mut out = [LayoutKey { dtype, rows: 0, cols: 0, ld: 0, batch: 0, stride: 0 }; 3];
    for (o, (s, stride)) in out.iter_mut().zip(stored_matrices(a).into_iter().zip([stride_a, stride_b, stride_c])) {
        *o = LayoutKey { dtype, rows: s.rows, cols: s.cols, ld: s.ld, batch: a.batch, stride };
    }
    out
}

/// The cache key of one call: its cuBLAS arguments, the three pointers (named after the **geom**
/// operands) and the compute type.
pub(crate) fn key<T: ExScalar>(args: &CublasArgs, ptrs: [u64; 3], compute: lt::cublasComputeType_t) -> Key {
    // SAFETY: `cudaDataType_t` is the same enum in both modules.
    let scale: lt::cudaDataType_t = unsafe { retyped(T::DATA) };
    let [a_l, b_l, c_l] = layout_keys(args, scale as u32);
    let [ptr_a, ptr_b, ptr_c] = slot_ptrs(ptrs);
    Key {
        compute: compute as u32,
        scale: scale as u32,
        trans: [args.transa, args.transb],
        layouts: [a_l, b_l, c_l, c_l],
        workspace: WORKSPACE_BYTES as u64,
        align: [align(ptr_a), align(ptr_b), align(ptr_c), align(ptr_c)],
    }
}

/// The pointers of cuBLAS's slots `A`, `B`, `C`: `CublasArgs` describes the slots, and slot `A`
/// holds `geom.b` (the operand swap of `cublas_args`), slot `B` `geom.a`.
pub(crate) fn slot_ptrs(ptrs: [u64; 3]) -> [u64; 3] {
    [ptrs[1], ptrs[0], ptrs[2]]
}

/// Everything a cached problem holds: the descriptors and the chosen algorithm. A descriptor
/// that is null was never created (a build that failed half-way).
struct Entry {
    desc: lt::cublasLtMatmulDesc_t,
    layouts: [lt::cublasLtMatrixLayout_t; 4],
    pref: lt::cublasLtMatmulPreference_t,
    algo: lt::cublasLtMatmulAlgo_t,
    /// The workspace this algorithm needs, in bytes (at most the preference's maximum).
    workspace: usize,
}

impl Drop for Entry {
    fn drop(&mut self) {
        // Either the ordinal is done with the descriptors, or the cache is cleared under its
        // lock: nothing is running against them.
        unsafe {
            if !self.pref.is_null() {
                lt::cublasLtMatmulPreferenceDestroy(self.pref);
            }
            for l in self.layouts {
                if !l.is_null() {
                    lt::cublasLtMatrixLayoutDestroy(l);
                }
            }
            if !self.desc.is_null() {
                lt::cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}

fn set_desc(
    desc: lt::cublasLtMatmulDesc_t,
    attr: lt::cublasLtMatmulDescAttributes_t,
    value: &impl Sized,
) -> lt::cublasStatus_t {
    unsafe {
        lt::cublasLtMatmulDescSetAttribute(
            desc,
            attr,
            value as *const _ as *const c_void,
            core::mem::size_of_val(value),
        )
    }
}

fn set_layout(
    layout: lt::cublasLtMatrixLayout_t,
    attr: lt::cublasLtMatrixLayoutAttribute_t,
    value: &impl Sized,
) -> lt::cublasStatus_t {
    unsafe {
        lt::cublasLtMatrixLayoutSetAttribute(
            layout,
            attr,
            value as *const _ as *const c_void,
            core::mem::size_of_val(value),
        )
    }
}

fn set_pref(
    pref: lt::cublasLtMatmulPreference_t,
    attr: lt::cublasLtMatmulPreferenceAttributes_t,
    value: &impl Sized,
) -> lt::cublasStatus_t {
    unsafe {
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref,
            attr,
            value as *const _ as *const c_void,
            core::mem::size_of_val(value),
        )
    }
}

/// The status as a refusal; `None` on success.
fn refused(status: lt::cublasStatus_t) -> Option<Refusal> {
    if succeeded(status) { None } else { Some(Refusal::CublasLtStatus(status)) }
}

/// Build the descriptors of `call` and ask the heuristic for its algorithm.
///
/// Every failure is a refusal: nothing has been enqueued at this point.
///
/// # Safety
/// The handle must be live and its lock held.
unsafe fn build<T: ExScalar>(
    handle: lt::cublasLtHandle_t,
    call: &GemmCall<'_>,
    compute: lt::cublasComputeType_t,
) -> Served<Entry> {
    // SAFETY: `cudaDataType_t` is the same enum in both modules.
    let scale: lt::cudaDataType_t = unsafe { retyped(T::DATA) };
    let a = &call.args;
    let mut desc = core::mem::MaybeUninit::uninit();
    if let Some(r) = refused(unsafe { lt::cublasLtMatmulDescCreate(desc.as_mut_ptr(), compute, scale) }) {
        return Err(r);
    }
    // From here on a failure drops `entry`, which destroys what was created so far.
    let mut entry = Entry {
        desc: unsafe { desc.assume_init() },
        layouts: [core::ptr::null_mut(); 4],
        pref: core::ptr::null_mut(),
        algo: core::mem::zeroed(),
        workspace: 0,
    };
    let op = |t: bool| if t { cb::cublasOperation_t::CUBLAS_OP_T } else { cb::cublasOperation_t::CUBLAS_OP_N };
    if let Some(r) =
        refused(set_desc(entry.desc, lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &op(a.transa)))
    {
        return Err(r);
    }
    if let Some(r) =
        refused(set_desc(entry.desc, lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &op(a.transb)))
    {
        return Err(r);
    }
    // The batch stride of each slot, in elements (the same triple `cublasGemmStridedBatched`
    // takes).
    let (stride_a, stride_b, stride_c) = crate::matmul::cublas_args::batch_strides(a);
    // `D` is `C`: the product is accumulated in place, or written over the unread `C`.
    let [s_a, s_b, s_c] = stored_matrices(a);
    let slots = [s_a, s_b, s_c, s_c].into_iter().zip([stride_a, stride_b, stride_c, stride_c]);
    for (l, (s, stride)) in entry.layouts.iter_mut().zip(slots) {
        if let Some(r) = refused(unsafe { lt::cublasLtMatrixLayoutCreate(l, scale, s.rows, s.cols, s.ld) }) {
            return Err(r);
        }
        if a.batch > 1 {
            set_layout(*l, lt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &a.batch);
            set_layout(*l, lt::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride);
        }
    }
    let mut pref = core::mem::MaybeUninit::uninit();
    if let Some(r) = refused(unsafe { lt::cublasLtMatmulPreferenceCreate(pref.as_mut_ptr()) }) {
        return Err(r);
    }
    entry.pref = unsafe { pref.assume_init() };
    if let Some(r) = refused(set_pref(
        entry.pref,
        lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &(WORKSPACE_BYTES as u64),
    )) {
        return Err(r);
    }

    let [l_a, l_b, l_c, l_d] = entry.layouts;
    let mut heuristic: lt::cublasLtMatmulHeuristicResult_t = core::mem::zeroed();
    let mut count = 0i32;
    let status = unsafe {
        lt::cublasLtMatmulAlgoGetHeuristic(
            handle,
            entry.desc,
            l_a,
            l_b,
            l_c,
            l_d,
            entry.pref,
            1,
            &mut heuristic,
            &mut count,
        )
    };
    if let Some(r) = refused(status) {
        return Err(r);
    }
    if count == 0 {
        return Err(Refusal::Cannot("no algorithm for this problem (layout, precision or workspace)"));
    }
    if let Some(r) = refused(heuristic.state) {
        return Err(r);
    }
    // The algorithm's configuration for exactly these descriptors (workspace, tiles, ...).
    let mut checked: lt::cublasLtMatmulHeuristicResult_t = core::mem::zeroed();
    let status =
        unsafe { lt::cublasLtMatmulAlgoCheck(handle, entry.desc, l_a, l_b, l_c, l_d, &heuristic.algo, &mut checked) };
    if let Some(r) = refused(status) {
        return Err(r);
    }
    if let Some(r) = refused(checked.state) {
        return Err(r);
    }
    if checked.workspaceSize > WORKSPACE_BYTES {
        return Err(Refusal::Cannot("the chosen algorithm needs a larger workspace"));
    }
    entry.algo = checked.algo;
    entry.workspace = checked.workspaceSize;
    Ok(entry)
}

/// The per-ordinal cuBLASLt state: the handle, the lock held across every call, and the cache.
pub(crate) struct CublasLtHandle {
    handle: lt::cublasLtHandle_t,
    locked: Mutex<HashMap<Key, Entry>>,
}

// SAFETY: the handle is only used under `locked`.
unsafe impl Send for CublasLtHandle {}
unsafe impl Sync for CublasLtHandle {}

impl CublasLtHandle {
    /// Create the handle on the context of `stream`'s ordinal.
    pub(crate) fn new(stream: &Arc<crate::stream::CudaStreamObj>) -> Result<Self> {
        stream.inner().context().bind_to_thread().map_err(crate::error::map_driver_error)?;
        let mut handle = core::mem::MaybeUninit::<lt::cublasLtHandle_t>::uninit();
        let status = unsafe { lt::cublasLtCreate(handle.as_mut_ptr()) };
        if !succeeded(status) {
            rstsr_raise!(RuntimeError, "cuda matmul: cuBLASLt status {status:?} while creating the handle")?;
        }
        Ok(Self { handle: unsafe { handle.assume_init() }, locked: Mutex::new(HashMap::new()) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Entry>> {
        self.locked.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for CublasLtHandle {
    fn drop(&mut self) {
        // Entries first: the descriptors must not outlive the handle.
        self.lock().clear();
        unsafe { lt::cublasLtDestroy(self.handle) };
    }
}

/// One attempt of the cuBLASLt driver.
pub(super) fn run<TA, TB, TC, TBuf>(p: &Problem<'_, TA, TB, TC, TBuf>, plan: CublasPlan) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    match plan {
        CublasPlan::F32(mode) => typed::<f32, TA, TB, TC, TBuf>(p, unsafe { retyped(mode.gemm_ex()) }),
        CublasPlan::F64(mode) => match mode.gemm_ex() {
            Some(compute) => typed::<f64, TA, TB, TC, TBuf>(p, unsafe { retyped(compute) }),
            None => Ok(Verdict::Declined(Refusal::Cannot("the fixed-point emulation is served by the classic driver"))),
        },
        CublasPlan::None => {
            Ok(Verdict::Declined(Refusal::Cannot("cuBLASLt serves f32 and f64 outputs in their own storage")))
        },
    }
}

fn typed<T: ExScalar, TA, TB, TC, TBuf>(
    p: &Problem<'_, TA, TB, TC, TBuf>,
    compute: lt::cublasComputeType_t,
) -> Result<Verdict>
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    TBuf: 'static,
{
    let Some(u) = p.uniform::<T>() else {
        return Ok(Verdict::Declined(Refusal::Cannot("the operands and the output must share one dtype")));
    };
    let call = match prepare(&u, p)? {
        Ok(call) => call,
        Err(r) => return Ok(Verdict::Declined(r)),
    };
    let Some(lt_state) = p.device.ordinal_obj().cublaslt() else {
        return Ok(Verdict::Declined(Refusal::NoCublasLt));
    };
    let Some(workspace) = call.exec.cublaslt_workspace() else {
        return Ok(Verdict::Declined(Refusal::Cannot("the cuBLASLt workspace could not be allocated")));
    };
    let key = key::<T>(&call.args, call.ptrs, compute);
    let mut cache = lt_state.lock();
    if !cache.contains_key(&key) {
        // SAFETY: the handle is live and this thread holds its lock.
        let entry = match unsafe { build::<T>(lt_state.handle, &call, compute) } {
            Ok(entry) => entry,
            Err(r) => return Ok(Verdict::Declined(r)),
        };
        if cache.len() >= CACHE_MAX {
            cache.clear();
        }
        cache.insert(key, entry);
    }
    let e = &cache[&key];
    let [ptr_a, ptr_b, ptr_c] = slot_ptrs(call.ptrs);
    let ptr = |p: u64| p as *mut c_void;
    let status = unsafe {
        lt::cublasLtMatmul(
            lt_state.handle,
            e.desc,
            &u.alpha as *const T as *const c_void,
            ptr(ptr_a) as *const c_void,
            e.layouts[0],
            ptr(ptr_b) as *const c_void,
            e.layouts[1],
            &u.beta as *const T as *const c_void,
            ptr(ptr_c) as *const c_void,
            e.layouts[2],
            ptr(ptr_c),
            e.layouts[3],
            &e.algo,
            workspace.device_ptr_on(call.exec) as *mut c_void,
            e.workspace,
            // the stream handle: the same opaque type in the driver and cublaslt modules
            call.exec.inner().cu_stream().cast::<c_void>() as lt::cudaStream_t,
        )
    };
    match refused(status) {
        None => Ok(Verdict::Ran),
        Some(r) => Ok(Verdict::Declined(r)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::cublas_args::{args, batch_strides};
    use crate::matmul::geom::resolve_geom;
    use crate::matmul::mode::{ComputeType32F, ComputeType64F};

    fn layout(shape: &[usize], stride: &[isize]) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), 0).unwrap()
    }

    /// (batch, a's strides, b's strides): the four operand orientations, then a strided batch
    /// and a broadcast one.
    fn cases() -> Vec<(&'static str, usize, &'static [isize], &'static [isize])> {
        vec![
            ("a k-contig, b n-contig", 1, &[4, 1], &[5, 1]),
            ("a m-contig, b n-contig", 1, &[1, 4], &[5, 1]),
            ("a k-contig, b k-contig", 1, &[4, 1], &[1, 5]),
            ("a m-contig, b k-contig", 1, &[1, 4], &[1, 5]),
            ("batched", 6, &[4, 1], &[5, 1]),
            ("a broadcast over the batch", 6, &[0, 4, 1], &[5, 1]),
        ]
    }

    fn arguments(a_strides: &[isize], b_strides: &[isize], batch: usize) -> CublasArgs {
        let (m, k, n) = (3usize, 4usize, 5usize);
        let (la, lb, lc) = if batch > 1 {
            // a 2-D stride list gets the batch axis in front, stepping by one matrix
            let batched = |s: &[isize], per: usize| match s.len() {
                2 => vec![per as isize, s[0], s[1]],
                _ => s.to_vec(),
            };
            (
                layout(&[batch, m, k], &batched(a_strides, m * k)),
                layout(&[batch, k, n], &batched(b_strides, k * n)),
                layout(&[batch, m, n], &[(m * n) as isize, n as isize, 1]),
            )
        } else {
            (layout(&[m, k], a_strides), layout(&[k, n], b_strides), layout(&[m, n], &[n as isize, 1]))
        };
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        args(&g).unwrap()
    }

    /// The descriptors declare what `cublas_args` declares: `op(A)` is `n x k` of `geom.b` and
    /// `op(B)` is `k x m` of `geom.a`, each stored one way or the other, and the output `n x m`
    /// with `ldc`; the batch count and the element strides follow the strided-batched triple.
    #[test]
    fn stored_matrices_follow_the_declared_dimensions() {
        for (label, batch, sa, sb) in cases() {
            let a = arguments(sa, sb, batch);
            let [s_a, s_b, s_c] = stored_matrices(&a);
            // cuBLAS's m = our n, n = our m, k = our k
            assert_eq!((a.mo, a.no, a.ko), (5, 3, 4), "{label}");
            let (rows_a, cols_a) = if a.transa { (a.ko as u64, a.mo as u64) } else { (a.mo as u64, a.ko as u64) };
            assert_eq!((s_a.rows, s_a.cols, s_a.ld), (rows_a, cols_a, a.lda as i64), "{label}: slot A");
            let (rows_b, cols_b) = if a.transb { (a.no as u64, a.ko as u64) } else { (a.ko as u64, a.no as u64) };
            assert_eq!((s_b.rows, s_b.cols, s_b.ld), (rows_b, cols_b, a.ldb as i64), "{label}: slot B");
            assert_eq!((s_c.rows, s_c.cols, s_c.ld), (5, 3, a.ldc as i64), "{label}: slot C/D");
            // a declared `ld` is never below the declared row count
            assert!(s_a.ld >= s_a.rows as i64 && s_b.ld >= s_b.rows as i64 && s_c.ld >= s_c.rows as i64, "{label}");

            let keys = layout_keys(&a, 0);
            let strides = batch_strides(&a);
            assert_eq!((keys[0].batch, keys[1].batch, keys[2].batch), (a.batch, a.batch, a.batch), "{label}");
            assert_eq!(
                (keys[0].stride, keys[1].stride, keys[2].stride),
                strides,
                "{label}: strides in elements, by slot"
            );
            assert_eq!(keys[0].rows, s_a.rows);
        }
    }

    /// The key is the whole problem: equal for an equal problem, different in the compute type,
    /// in a transpose, in a batch stride, and in the alignment class of any pointer.
    #[test]
    fn key_is_the_whole_problem() {
        let (plain, transposed) = (arguments(&[4, 1], &[5, 1], 1), arguments(&[1, 4], &[1, 5], 1));
        let ptrs = [0x1000, 0x2000, 0x3000];
        let base = key::<f32>(&plain, ptrs, unsafe { retyped(ComputeType32F::default().gemm_ex()) });
        assert_eq!(base, key::<f32>(&plain, ptrs, unsafe { retyped(ComputeType32F::default().gemm_ex()) }));
        assert_ne!(base, key::<f32>(&plain, ptrs, unsafe { retyped(ComputeType32F::FastTF32.gemm_ex()) }));
        assert_ne!(base, key::<f32>(&transposed, ptrs, unsafe { retyped(ComputeType32F::default().gemm_ex()) }));
        assert_ne!(
            base,
            key::<f32>(&arguments(&[4, 1], &[5, 1], 6), ptrs, unsafe { retyped(ComputeType32F::default().gemm_ex()) })
        );
        for i in 0..3 {
            let mut p = ptrs;
            p[i] += 4;
            assert_ne!(
                base,
                key::<f32>(&plain, p, unsafe { retyped(ComputeType32F::default().gemm_ex()) }),
                "pointer {i}"
            );
        }
        // 256-byte-equivalent addresses are one class
        let far = [ptrs[0] + 0x100, ptrs[1], ptrs[2]];
        assert_eq!(base, key::<f32>(&plain, far, unsafe { retyped(ComputeType32F::default().gemm_ex()) }));
        // the scale type is the output type
        let scale = |k: &Key| k.scale;
        assert_eq!(
            scale(&key::<f32>(&plain, ptrs, unsafe { retyped(ComputeType32F::default().gemm_ex()) })),
            lt::cudaDataType_t::CUDA_R_32F as u32
        );
        assert_eq!(
            scale(&key::<f64>(&plain, ptrs, unsafe { retyped(ComputeType64F::default().gemm_ex().unwrap()) })),
            lt::cudaDataType_t::CUDA_R_64F as u32
        );
    }

    /// The `cublas` and `cublaslt` bindings spell one enum twice: every value the drivers pass
    /// keeps its name and value.
    #[test]
    fn retyped_enums_keep_their_values() {
        let same = |a: cb::cublasComputeType_t| {
            let b: lt::cublasComputeType_t = unsafe { retyped(a) };
            assert_eq!(a as u32, b as u32, "{a:?}");
        };
        for mode in [
            ComputeType32F::default(),
            ComputeType32F::Pedantic,
            ComputeType32F::Fast16F,
            ComputeType32F::Fast16BF,
            ComputeType32F::FastTF32,
        ] {
            same(mode.gemm_ex());
        }
        same(ComputeType64F::default().gemm_ex().unwrap());
        same(ComputeType64F::Pedantic.gemm_ex().unwrap());
        let data: lt::cudaDataType_t = unsafe { retyped(cb::cudaDataType_t::CUDA_R_32F) };
        assert_eq!(data, lt::cudaDataType_t::CUDA_R_32F);
    }

    /// The pointers follow the slots: `geom.b` in cuBLAS's `A`, `geom.a` in its `B`.
    #[test]
    fn pointers_follow_the_slots() {
        assert_eq!(slot_ptrs([0x10, 0x20, 0x30]), [0x20, 0x10, 0x30]);
    }

    /// The fixed-point emulation of the 64F family is the classic driver's.
    #[test]
    fn fixed_point_emulation_is_left_to_the_classic_driver() {
        assert!(ComputeType64F::EmulatedFixedPoint.gemm_ex().is_none());
    }
}
