//! Generated kernels: byte copy, cast, fill, sequence (arange/linspace),
//! and the format gather (B2/B3 of the kernel round).
//!
//! Layout convention (D9): all elementwise kernels are rank-generic T0
//! kernels — runtime shape/stride arrays in element units (pre-multiplied
//! host-side), a flat grid-stride loop over the destination visit order,
//! and an inner-contiguity fast branch when applicable. Copy-in-bytes
//! kernels (`copy_bytes`, `gather_bytes`) are dtype-free; typed kernels
//! generate from `CudaDType::CNAME`.

use crate::buffer::CudaBuffer;
use std::sync::Arc;
use crate::dtype::CudaDType;
use crate::error::map_driver_result;
use crate::kernel::{assemble_source, make_key, order_input, FlatLaunch};
use crate::prelude_dev::*;

/// Host-side flattened layout for T0 kernels: element-unit strides.
pub(crate) struct FlatLayout {
    pub offset: usize,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
}

impl FlatLayout {
    /// Flatten `layout` (element-unit strides; layout must address `size()`
    /// elements of a buffer whose base pointer is passed separately).
    pub fn of<D: DimAPI>(layout: &Layout<D>) -> Self {
        Self {
            offset: layout.offset(),
            shape: layout.shape().as_ref().iter().map(|&v| v as u64).collect(),
            strides: layout.stride().as_ref().iter().map(|&v| v as u64).collect(),
        }
    }
}

impl FlatLayout {
    /// Device-side arg bytes: (shape array, strides array), each as
    /// [len: u64][elems: u64 × len].
    fn scratch_bytes_of(&self) -> Vec<u8> {
        let mut b = shape_bytes(&self.shape);
        b.extend_from_slice(&shape_bytes(&self.strides));
        b
    }
}

/// Split a packed (shape, strides) region into device pointers, skipping
/// each array's length prefix.
fn split_mark(scratch: &KernelScratch, bytes: &[u8], m: (usize, usize)) -> (u64, u64) {
    let n_elem = u64::from_ne_bytes(bytes[m.0..m.0 + 8].try_into().unwrap()) as usize;
    let mid = m.0 + 8 * (1 + n_elem);
    (scratch.ptr_at(m.0 + 8), scratch.ptr_at(mid + 8))
}

/// Serialize a `Vec<u64>` for kernel marshaling (padded to 8 bytes/elem).
fn shape_bytes(v: &[u64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 8 + 8);
    b.extend_from_slice(&(v.len() as u64).to_ne_bytes());
    for x in v {
        b.extend_from_slice(&x.to_ne_bytes());
    }
    b
}



// /* #region byte-space copy (dtype-free) */

/// Strided device-to-device copy in bytes. `esz_out`/`esz_in` are the dtype
/// sizes; the copy is a straight gather-scatter of byte blocks (equal sizes
/// for same-dtype copy).
///
/// Body shape: out[oo] = in[io] blockwise.
fn copy_bytes_source() -> String {
    let body = r#"
extern "C" __global__ void rstsr_copy_bytes(
    char* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const char* in, int irank, const rstsr_usz* ishape, const rstsr_usz* istrides, rstsr_usz ioffset,
    rstsr_usz n, rstsr_usz esz)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    rstsr_usz io = rstsr_index_to_offset(lin, orank, oshape, istrides, ioffset);
    // word-wise copy: esz is 1/2/4/8/16 and pointers are aligned
    const char* src = in + io * esz;
    char* dst = out + oo * esz;
    if (esz == 16) { *(double2*)dst = *(const double2*)src; }
    else if (esz == 8) { *(double*)dst = *(const double*)src; }
    else if (esz == 4) { *(int*)dst = *(const int*)src; }
    else if (esz == 2) { *(short*)dst = *(const short*)src; }
    else { *dst = *src; }
}
"#;
    body.to_string()
}

/// Same-dtype strided assign (no cast): `c[lc] = a[la]`.
pub(crate) fn copy_same_dtype<T: CudaDType, DC: DimAPI, DA: DimAPI>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<T>,
    lc: &Layout<DC>,
    a: &CudaBuffer<T>,
    la: &Layout<DA>,
) -> Result<()> {
    if lc.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let _guard = order_input(a, exec)?;
    let key = make_key(device.ordinal_obj(), "rstsr_copy_bytes", &assemble_source(&[], &copy_bytes_source()), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(lc);
    let li = FlatLayout::of(la);
    let mut bytes = lo.scratch_bytes_of();
    let o_mark = (0usize, bytes.len());
    bytes.extend_from_slice(&li.scratch_bytes_of());
    let i_mark = (o_mark.1, bytes.len());
    let scratch = KernelScratch::new(exec, &bytes)?;
    let (os, ost) = split_mark(&scratch, &bytes, o_mark);
    let (is, ist) = split_mark(&scratch, &bytes, i_mark);
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_usize(lo.shape.len())
        .arg_u64(os)
        .arg_u64(ost)
        .arg_usize(lo.offset)
        .arg_buf(a)
        .arg_usize(li.shape.len())
        .arg_u64(is)
        .arg_u64(ist)
        .arg_usize(li.offset)
        .arg_usize(lc.size())
        .arg_usize(core::mem::size_of::<T>());
    launch.launch()
}

// /* #endregion */

// /* #region cast kernels */

fn cast_source(tc: &str, ta: &str, expr: &str) -> String {
    let body = format!(
        r#"
extern "C" __global__ void rstsr_cast(
    {tc}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const {ta}* in, int irank, const rstsr_usz* ishape, const rstsr_usz* istrides, rstsr_usz ioffset,
    rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    rstsr_usz io = rstsr_index_to_offset(lin, orank, oshape, istrides, ioffset);
    out[oo] = ({expr});
}}
"#,
    );
    body
}

/// Cast expression from TA to TC in kernel C (host-side generated).
/// `src` is the input C expression (e.g. `in[io]` or a scalar name).
fn cast_expr_c(ta: &str, tc: &str, src: &str) -> Result<String> {
    if ta == tc {
        return Ok(src.to_string());
    }
    let load = match ta {
        "__half" => format!("__half2float({src})"),
        "__nv_bfloat16" => format!("__nv_bfloat162float({src})"),
        _ => src.to_string(),
    };
    Ok(match (ta, tc) {
        // complex -> complex: component cast
        ("rstsr_c32" | "rstsr_c64", "rstsr_c32" | "rstsr_c64") => {
            let ctor = if tc == "rstsr_c32" { "make_c32" } else { "make_c64" };
            let c = if tc == "rstsr_c32" { "float" } else { "double" };
            format!("{ctor}(({c})({src}.re), ({c})({src}.im))")
        }
        // primitive -> complex (half loads via float above)
        (_, "rstsr_c32") => format!("make_c32((float)({load}), 0.0f)"),
        (_, "rstsr_c64") => format!("make_c64((double)({load}), 0.0)"),
        // complex -> real: policy D6 keeps this out of kernels
        ("rstsr_c32" | "rstsr_c64", _) => {
            return Err(rstsr_error!(InvalidValue, "complex-to-real cast has no kernel (policy D6)"))
        }
        // T -> bool: nonzero test (half through float)
        (_, "bool") => format!("({load} != 0)"),
        // bool -> T: 0/1 (through float for half)
        ("bool", _) => match tc {
            "__half" => format!("__float2half(({src}) ? 1.0f : 0.0f)"),
            "__nv_bfloat16" => format!("__float2bfloat16(({src}) ? 1.0f : 0.0f)"),
            _ => format!("(({tc})(({src}) ? 1 : 0))"),
        },
        // numeric -> numeric (half target stores through float)
        _ => match tc {
            "__half" => format!("__float2half((float)({load}))"),
            "__nv_bfloat16" => format!("__float2bfloat16((float)({load}))"),
            _ => format!("(({tc})({load}))"),
        },
    })
}

/// Element-cast expression from TA to TC reading `in[io]`.
fn cast_expr<TA: CudaDType, TC: CudaDType>() -> Result<String> {
    cast_expr_c(TA::CNAME, TC::CNAME, "in[io]")
}

/// Cross-dtype strided assign: `c[lc] = cast(a[la])`.
pub(crate) fn copy_cast<TA: CudaDType, TC: CudaDType, DA: DimAPI, DC: DimAPI>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<DC>,
    a: &CudaBuffer<TA>,
    la: &Layout<DA>,
) -> Result<()> {
    if lc.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let _guard = order_input(a, exec)?;
    let expr = cast_expr::<TA, TC>()?;
    let includes = [TA::INCLUDES, TC::INCLUDES].concat();
    let body = cast_source(TC::CNAME, TA::CNAME, &expr);
    let key = make_key(device.ordinal_obj(), "rstsr_cast", &assemble_source(&includes, &body), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(lc);
    let li = FlatLayout::of(la);
    let mut bytes = lo.scratch_bytes_of();
    let o_mark = (0usize, bytes.len());
    bytes.extend_from_slice(&li.scratch_bytes_of());
    let i_mark = (o_mark.1, bytes.len());
    let scratch = KernelScratch::new(exec, &bytes)?;
    let (os, ost) = split_mark(&scratch, &bytes, o_mark);
    let (is, ist) = split_mark(&scratch, &bytes, i_mark);
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_usize(lo.shape.len())
        .arg_u64(os)
        .arg_u64(ost)
        .arg_usize(lo.offset)
        .arg_buf(a)
        .arg_usize(li.shape.len())
        .arg_u64(is)
        .arg_u64(ist)
        .arg_usize(li.offset)
        .arg_usize(lc.size());
    launch.launch()
}

// /* #endregion */

// /* #region fill */

/// Strided fill with a host value of dtype TA (cast to TC in-kernel when
/// different; same-dtype when not).
///
/// `value_bytes` must be the kernel-dtype (TC) representation, marshaled
/// little-endian on the host.
pub(crate) fn fill_dispatch<TA, TC, D>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<D>,
    fill: TA,
) -> Result<()>
where
    TA: CudaDType + DTypeCastAPI<TC>,
    TC: CudaDType,
    D: DimAPI,
{
    if lc.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let expr = cast_expr::<TA, TC>()?;
    let scalar_c = format!("({expr})").replace("in[io]", "v_in");
    let src = format!(
        r#"
extern "C" __global__ void rstsr_fill_cast(
    {tc}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    rstsr_usz n, {ta} v_in)
{{
    {tc} v = ({scalar_c});
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    out[oo] = v;
}}
"#,
        tc = TC::CNAME,
        ta = TA::CNAME,
    );
    let includes = [TA::INCLUDES, TC::INCLUDES].concat();
    let key = make_key(device.ordinal_obj(), "rstsr_fill_cast", &assemble_source(&includes, &src), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(lc);
    // marshal the scalar in TA's representation (host struct layout for
    // complex is {re, im} = the Rust Complex layout)
    let nbytes = core::mem::size_of::<TA>();
    let mut scalar_bytes = vec![0u8; nbytes];
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(fill) as *const u8, scalar_bytes.as_mut_ptr(), nbytes);
    }
    let bytes = lo.scratch_bytes_of();
    let o_mark = (0usize, bytes.len());
    let scratch = KernelScratch::new(exec, &bytes)?;
    let (os, ost) = split_mark(&scratch, &bytes, o_mark);
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_usize(lo.shape.len())
        .arg_u64(os)
        .arg_u64(ost)
        .arg_usize(lo.offset)
        .arg_usize(lc.size())
        .arg_bytes(&scalar_bytes);
    launch.launch()
}

// /* #endregion */

// /* #region sequence (arange/linspace) */

fn sequence_source(t: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_sequence(
    {t}* out, rstsr_usz n, {t} start, {t} step)
{{
    rstsr_usz i = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (i >= n) return;
    // i promoted through double for float dtypes; integer wraps naturally
    out[i] = ({t})(({t}) start + ({t}) step * ({t}) i);
}}
"#
    )
}

/// Contiguous `arange`-style sequence into a fresh buffer.
pub fn sequence<T: CudaDType>(device: &DeviceCuda, n: usize, start: T, step: T) -> Result<CudaBuffer<T>> {
    let buf = CudaBuffer::<T>::new(device.stream(), n)?;
    if n == 0 {
        return Ok(buf);
    }
    let exec = device.stream();
    let key = make_key(device.ordinal_obj(), "rstsr_sequence", &assemble_source(T::INCLUDES, &sequence_source(T::CNAME)), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let nbytes = core::mem::size_of::<T>();
    let mut sbytes = vec![0u8; nbytes];
    let mut stbytes = vec![0u8; nbytes];
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(start) as *const u8, sbytes.as_mut_ptr(), nbytes);
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(step) as *const u8, stbytes.as_mut_ptr(), nbytes);
    }
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(&buf)
        .arg_usize(n)
        .arg_bytes(&sbytes)
        .arg_bytes(&stbytes);
    launch.launch()?;
    Ok(buf)
}

fn linspace_source(t: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_linspace(
    {t}* out, rstsr_usz n, {t} start, {t} end, int endpoint)
{{
    rstsr_usz i = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (i >= n) return;
    rstsr_usz denom = (endpoint || n <= 1) ? (n > 1 ? n - 1 : 1) : n;
    {t} frac = ({t})(({f}) i / ({f}) denom);
    {t} span = ({t})(({f}) end - ({f}) start);
    out[i] = ({t})(({f}) start + ({f}) span * ({f}) frac);
}}
"#,
        t = t,
        f = if t == "float" { "float" } else { "double" },
    )
}

/// Contiguous linspace into a fresh buffer (endpoint per NumPy semantics).
pub(crate) fn linspace<T: CudaDType>(device: &DeviceCuda, n: usize, start: T, end: T, endpoint: bool) -> Result<CudaBuffer<T>> {
    let buf = CudaBuffer::<T>::new(device.stream(), n)?;
    if n == 0 {
        return Ok(buf);
    }
    let exec = device.stream();
    let key = make_key(device.ordinal_obj(), "rstsr_linspace", &assemble_source(T::INCLUDES, &linspace_source(T::CNAME)), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let nbytes = core::mem::size_of::<T>();
    let mut sbytes = vec![0u8; nbytes];
    let mut ebytes = vec![0u8; nbytes];
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(start) as *const u8, sbytes.as_mut_ptr(), nbytes);
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(end) as *const u8, ebytes.as_mut_ptr(), nbytes);
    }
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(&buf)
        .arg_usize(n)
        .arg_bytes(&sbytes)
        .arg_bytes(&ebytes)
        .arg_bytes(&(endpoint as u8).to_ne_bytes());
    launch.launch()?;
    Ok(buf)
}

// /* #endregion */

// /* #region format gather */

fn gather_bytes_source() -> String {
    let body = r#"
extern "C" __global__ void rstsr_gather_bytes(
    char* out, const char* in, const rstsr_usz* idx, rstsr_usz n, rstsr_usz esz)
{
    rstsr_usz i = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (i >= n) return;
    // word-wise gather (same dtype-size dispatch as the copy kernel)
    const char* src = in + idx[i] * esz;
    char* dst = out + i * esz;
    if (esz == 16) { *(double2*)dst = *(const double2*)src; }
    else if (esz == 8) { *(double*)dst = *(const double*)src; }
    else if (esz == 4) { *(int*)dst = *(const int*)src; }
    else if (esz == 2) { *(short*)dst = *(const short*)src; }
    else { *dst = *src; }
}
"#;
    body.to_string()
}

/// One batched D2H gather of `indices` (absolute element offsets) + single
/// sync — replaces the per-element `d2h_scalar` loop in formatting.
pub(crate) fn gather_to_host<T: Clone>(
    device: &DeviceCuda,
    raw: &CudaBuffer<T>,
    indices: &[usize],
) -> Result<Vec<T>> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let esz = core::mem::size_of::<T>();
    let exec = device.stream();
    let _guard = order_input(raw, exec)?;
    let key = make_key(device.ordinal_obj(), "rstsr_gather_bytes", &assemble_source(&[], &gather_bytes_source()), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    // staging: upload the index list, run the gather, download the values
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| (*i as u64).to_ne_bytes()).collect();
    let mut scratch = CudaBuffer::<u8>::new(exec, idx_bytes.len())?;
    unsafe {
        // blocking H2D of the indices (small; formatting is a sync point anyway)
        let ptr = scratch.device_ptr_mut() as u64;
        let src = idx_bytes.as_ptr() as *const core::ffi::c_void;
        map_driver_result(cudarc::driver::result::memcpy_htod_async(
            ptr,
            std::slice::from_raw_parts(src as *const u8, idx_bytes.len()),
            exec.inner().cu_stream(),
        ))?;
    }
    let out_len = indices.len() * esz;
    let out_buf = CudaBuffer::<u8>::new(exec, out_len)?;
    let launch = FlatLaunch::new(exec, &f, indices.len())
        .arg_buf(&out_buf)
        .arg_buf(raw)
        .arg_buf(&scratch)
        .arg_usize(indices.len())
        .arg_usize(esz);
    launch.launch()?;
    // one sync + one D2H for the gathered bytes
    let bytes = crate::transfer::d2h_full(&out_buf)?;
    let mut vec: Vec<T> = Vec::with_capacity(indices.len());
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, vec.as_mut_ptr(), indices.len());
        vec.set_len(indices.len());
    }
    Ok(vec)
}

// /* #endregion */

// /* #region elementwise math kernels (B5) */

/// Device scratch holding the shape/stride arrays of one launch (uploaded
/// once, H2D blocking — tiny). Allocated on the launch stream so its
/// stream-ordered free lands after the kernel (allocation-stream owns
/// free); keep alive until after `FlatLaunch::launch`.
struct KernelScratch {
    buf: CudaBuffer<u8>,
}

impl KernelScratch {
    fn new(exec: &Arc<CudaStreamObj>, bytes: &[u8]) -> Result<Self> {
        let mut buf = CudaBuffer::<u8>::new(exec, bytes.len().max(1))?;
        if !bytes.is_empty() {
            let stream = buf.as_bytes().stream().clone();
            let ptr = buf.device_ptr_mut() as u64;
            // SAFETY: plain byte upload into the fresh scratch allocation
            // async upload on the launch stream (same-stream ordering with
            // the subsequent kernel launch is guaranteed)
            unsafe {
                map_driver_result(cudarc::driver::result::memcpy_htod_async(
                    ptr,
                    bytes,
                    stream.cu_stream(),
                ))?;
            }
        }
        Ok(Self { buf })
    }

    /// Device pointer to byte `offset` inside the scratch (8-aligned).
    fn ptr_at(&self, offset: usize) -> u64 {
        (self.buf.device_ptr() as u64) + offset as u64
    }
}

/// Layout-argument packer: uploads shape/stride arrays to device scratch
/// and emits (ptr, rank, shape*, strides*, offset) in kernel order.
struct TensorArg<'a, T> {
    buf: &'a CudaBuffer<T>,
    layout: FlatLayout,
}

impl<'a, T> TensorArg<'a, T> {
    fn new(buf: &'a CudaBuffer<T>, layout: &Layout<impl DimAPI>) -> Self {
        Self { buf, layout: FlatLayout::of(layout) }
    }

    fn push(self, launch: FlatLaunch, shape_ptr: u64, strides_ptr: u64) -> FlatLaunch {
        launch
            .arg_buf(self.buf)
            .arg_usize(self.layout.shape.len())
            .arg_u64(shape_ptr)
            .arg_u64(strides_ptr)
            .arg_usize(self.layout.offset)
    }
}

/// Source of a two-input elementwise kernel: `out[oo] = EXPR(a[ia], b[ib])`
/// (dtypes may each differ; `TOUT_C` is the output C type).
fn binary_source(tout_c: &str, ta_c: &str, tb_c: &str, expr: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_binary(
    {tout_c}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const {ta_c}* a, int arank, const rstsr_usz* ashape, const rstsr_usz* astrides, rstsr_usz aoffset,
    const {tb_c}* b, int brank, const rstsr_usz* bshape, const rstsr_usz* bstrides, rstsr_usz boffset,
    rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    rstsr_usz ia = rstsr_index_to_offset(lin, orank, oshape, astrides, aoffset);
    rstsr_usz ib = rstsr_index_to_offset(lin, orank, oshape, bstrides, boffset);
    out[oo] = {expr};
}}
"#,
    )
}

/// Source of a one-input elementwise kernel: `out[oo] = EXPR(b[ib])`.
fn unary_source(tout_c: &str, tb_c: &str, expr: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_unary(
    {tout_c}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const {tb_c}* b, int brank, const rstsr_usz* bshape, const rstsr_usz* bstrides, rstsr_usz boffset,
    rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    rstsr_usz ib = rstsr_index_to_offset(lin, orank, oshape, bstrides, boffset);
    out[oo] = {expr};
}}
"#,
    )
}

/// Launch a compiled binary elementwise kernel (async, no sync).
#[allow(clippy::too_many_arguments)]
fn launch_binary<TOUT: CudaDType, TA: CudaDType, TB: CudaDType>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()> {
    if lc.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let includes = [TA::INCLUDES, TB::INCLUDES, TOUT::INCLUDES].concat();
    let src = assemble_source(
        &includes,
        &binary_source(TOUT::CNAME, TA::CNAME, TB::CNAME, expr),
    );
    let key = make_key(device.ordinal_obj(), "rstsr_binary", &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let c_typed: &mut CudaBuffer<TOUT> = unsafe {
        &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>)
    };
    let n = lc.size();
    let mut bytes = Vec::new();
    let mut marks = [(0usize, 0usize); 3];
    for (i, sb) in [
        FlatLayout::of(lc).scratch_bytes_of(),
        FlatLayout::of(la).scratch_bytes_of(),
        FlatLayout::of(lb).scratch_bytes_of(),
    ]
    .into_iter()
    .enumerate()
    {
        let start = bytes.len();
        bytes.extend_from_slice(&sb);
        marks[i] = (start, bytes.len());
    }
    let scratch = KernelScratch::new(exec, &bytes)?;
    let (os, ost) = split_mark(&scratch, &bytes, marks[0]);
    let (as_, ast) = split_mark(&scratch, &bytes, marks[1]);
    let (bs, bst) = split_mark(&scratch, &bytes, marks[2]);
    let launch = FlatLaunch::new(exec, &f, n);
    let launch = TensorArg::new(c_typed, lc).push(launch, os, ost);
    let launch = TensorArg::new(a, la).push(launch, as_, ast);
    let launch = TensorArg::new(b, lb).push(launch, bs, bst);
    launch.arg_usize(n).launch()
}

/// Launch a compiled unary elementwise kernel (async, no sync).
fn launch_unary<TOUT: CudaDType, TB: CudaDType>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()> {
    if lc.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let includes = [TB::INCLUDES, TOUT::INCLUDES].concat();
    let src = assemble_source(&includes, &unary_source(TOUT::CNAME, TB::CNAME, expr));
    let key = make_key(device.ordinal_obj(), "rstsr_unary", &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let c_typed: &mut CudaBuffer<TOUT> = unsafe {
        &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>)
    };
    let n = lc.size();
    let mut bytes = Vec::new();
    let mut marks = [(0usize, 0usize); 2];
    for (i, sb) in [
        FlatLayout::of(lc).scratch_bytes_of(),
        FlatLayout::of(lb).scratch_bytes_of(),
    ]
    .into_iter()
    .enumerate()
    {
        let start = bytes.len();
        bytes.extend_from_slice(&sb);
        marks[i] = (start, bytes.len());
    }
    let scratch = KernelScratch::new(exec, &bytes)?;
    let (os, ost) = split_mark(&scratch, &bytes, marks[0]);
    let (bs, bst) = split_mark(&scratch, &bytes, marks[1]);
    let launch = FlatLaunch::new(exec, &f, n);
    let launch = TensorArg::new(c_typed, lc).push(launch, os, ost);
    let launch = TensorArg::new(b, lb).push(launch, bs, bst);
    launch.arg_usize(n).launch()
}

/// The binary C expression for op `+ - * /` on C-typed operands.
fn arith_expr(op: char) -> String {
    format!("(a[ia] {op} b[ib])")
}

/// Public entry: binary arithmetic add/sub/mul/div into uninit output.
pub(crate) fn binary_arith<TOUT, TA, TB>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    op: char,
) -> Result<()>
where
    TOUT: CudaDType + crate::dtype::CudaArith,
    TA: CudaDType + crate::dtype::CudaArith,
    TB: CudaDType + crate::dtype::CudaArith,
{
    launch_binary(device, c, lc, a, la, b, lb, &arith_expr(op))
}

/// Public entry: binary comparison into a bool output (`op` is a C infix
/// operator: `<`, `>`, `<=`, `>=`, `==`, `!=`).
pub(crate) fn binary_cmp<TA, TB>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<bool>>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    op: &str,
) -> Result<()>
where
    TA: CudaDType + crate::dtype::CudaCmp,
    TB: CudaDType + crate::dtype::CudaCmp,
{
    launch_binary::<bool, TA, TB>(device, c, lc, a, la, b, lb, &format!("(a[ia] {op} b[ib])"))
}

/// Public entry: unary float function (`expr` is the full CUDA C call,
/// pre-built by the caller with the right float suffix).
pub(crate) fn unary_float<T>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<T>>,
    lc: &Layout<impl DimAPI>,
    b: &CudaBuffer<T>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    T: CudaDType + crate::dtype::CudaFloat,
{
    launch_unary::<T, T>(device, c, lc, b, lb, expr)
}

// /* #endregion */





/// Whether `v`'s dynamic type is in the fill-kernel's supported set.
#[doc(hidden)]
pub fn is_fill_supported<T: 'static>(_v: &T) -> bool {
    let t = core::any::TypeId::of::<T>();
    t == core::any::TypeId::of::<f32>()
        || t == core::any::TypeId::of::<f64>()
        || t == core::any::TypeId::of::<i8>()
        || t == core::any::TypeId::of::<i16>()
        || t == core::any::TypeId::of::<i32>()
        || t == core::any::TypeId::of::<i64>()
        || t == core::any::TypeId::of::<isize>()
        || t == core::any::TypeId::of::<u8>()
        || t == core::any::TypeId::of::<u16>()
        || t == core::any::TypeId::of::<u32>()
        || t == core::any::TypeId::of::<u64>()
        || t == core::any::TypeId::of::<usize>()
        || t == core::any::TypeId::of::<bool>()
}

/// Byte-erased fill: dispatches on `T`'s TypeId and re-types the scalar
/// into the TC=TA identity fill kernel (same TypeId-guard idiom as the CPU
/// fast paths; the caller checked `is_fill_supported`).
pub fn fill_bytes_erased<TC, T, D>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<D>,
    fill: &T,
) -> Result<()>
where
    TC: Clone + 'static,
    T: Clone + 'static,
    D: DimAPI,
{
    macro_rules! fill_as {
        ($ty:ty) => {{
            // identity fill: T == $ty == TC (caller checked is_fill_supported
            // and the tensor layer guarantees T == TC for fill)
            if core::any::TypeId::of::<T>() == core::any::TypeId::of::<$ty>()
                && core::any::TypeId::of::<TC>() == core::any::TypeId::of::<$ty>()
            {
                let v: $ty = unsafe { core::ptr::read(fill as *const T as *const $ty) };
                let c_typed: &mut CudaBuffer<$ty> = unsafe {
                    &mut *(c as *mut CudaBuffer<TC> as *mut CudaBuffer<$ty>)
                };
                return fill_dispatch::<$ty, $ty, D>(device, c_typed, lc, v);
            }
        }};
    }
    fill_as!(f32);
    fill_as!(f64);
    fill_as!(i8);
    fill_as!(i16);
    fill_as!(i32);
    fill_as!(i64);
    fill_as!(isize);
    fill_as!(u8);
    fill_as!(u16);
    fill_as!(u32);
    fill_as!(u64);
    fill_as!(usize);
    fill_as!(bool);
    rstsr_raise!(InvalidValue, "fill dtype not supported by the CUDA fill kernel")?
}
