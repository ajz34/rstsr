//! Generated kernels: byte copy, cast, fill, sequence, elementwise math,
//! and the format gather. Kernels are rank-generic (runtime shape/stride
//! arrays in element units); copy/gather kernels work in bytes and are
//! dtype-free, typed kernels generate from `CudaDType::CNAME`.

use crate::buffer::CudaBuffer;
use crate::dtype::CudaDType;

/// `&mut CudaBuffer<TA> -> &mut CudaBuffer<TC>` after a TypeId equality check.
///
/// # Safety
/// Caller must prove `TypeId::of::<TA>() == TypeId::of::<TC>()`.
unsafe fn transmute_ref_cross_mut<TA, TC>(b: &mut CudaBuffer<TA>) -> &mut CudaBuffer<TC> {
    &mut *(b as *mut CudaBuffer<TA> as *mut CudaBuffer<TC>)
}
use crate::error::map_driver_result;
use crate::kernel::{assemble_source, make_key, order_input, order_output, FlatLaunch};
use crate::prelude_dev::*;
use std::sync::{Arc, OnceLock};

/// Host-side flattened layout in element units.
pub(crate) struct FlatLayout {
    pub offset: usize,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
}

impl FlatLayout {
    /// Flatten `layout`; the base pointer is passed separately.
    pub fn of<D: DimAPI>(layout: &Layout<D>) -> Self {
        Self {
            offset: layout.offset(),
            shape: layout.shape().as_ref().iter().map(|&v| v as u64).collect(),
            strides: layout.stride().as_ref().iter().map(|&v| v as u64).collect(),
        }
    }

    /// Device-side arg bytes: (shape array, strides array), each as
    /// [len: u64][elems: u64 × len].
    fn scratch_bytes_of(&self) -> Vec<u8> {
        let mut b = shape_bytes(&self.shape);
        b.extend_from_slice(&shape_bytes(&self.strides));
        b
    }

    /// Strides array only — operands decompose by the output shape, so
    /// they need no shape array of their own.
    fn strides_bytes_of(&self) -> Vec<u8> {
        shape_bytes(&self.strides)
    }
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

/// One scratch upload holding the packed layout arrays of a launch.
/// `with_shape[i] == false` packs strides only (that shape pointer is 0
/// and never passed to a kernel).
struct LayoutPack {
    /// Keeps the upload alive until the launch is enqueued.
    _scratch: KernelScratch,
    shapes: Vec<u64>,
    strides: Vec<u64>,
}

impl LayoutPack {
    fn new(exec: &Arc<CudaStreamObj>, layouts: &[(&FlatLayout, bool)]) -> Result<Self> {
        let mut bytes = Vec::new();
        let mut starts = Vec::with_capacity(layouts.len());
        for (l, with_shape) in layouts {
            starts.push((bytes.len(), *with_shape));
            if *with_shape {
                bytes.extend_from_slice(&l.scratch_bytes_of());
            } else {
                bytes.extend_from_slice(&l.strides_bytes_of());
            }
        }
        let scratch = KernelScratch::new(exec, &bytes)?;
        let mut shapes = Vec::with_capacity(layouts.len());
        let mut strides = Vec::with_capacity(layouts.len());
        for &(start, with_shape) in &starts {
            let n = u64::from_ne_bytes(bytes[start..start + 8].try_into().unwrap()) as usize;
            if with_shape {
                shapes.push(scratch.ptr_at(start + 8));
                strides.push(scratch.ptr_at(start + 8 * (2 + n)));
            } else {
                shapes.push(0);
                strides.push(scratch.ptr_at(start + 8));
            }
        }
        Ok(Self { _scratch: scratch, shapes, strides })
    }
}

// /* #region byte-space copy (dtype-free) */
/// Strided device-to-device copy in bytes: `out[oo] = in[io]` blockwise.
/// Each side decomposes `lin` by its own shape and the device's order, so
/// rank/shape-changing assigns reproduce the CPU devices' positional zip.
const COPY_BYTES_SOURCE: &str = r#"
extern "C" __global__ void rstsr_copy_bytes(
    char* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const char* in, int irank, const rstsr_usz* ishape, const rstsr_usz* istrides, rstsr_usz ioffset,
    rstsr_usz n, rstsr_usz esz, int col_major)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_iter_offset(lin, orank, oshape, ostrides, ooffset, col_major);
    rstsr_usz io = rstsr_iter_offset(lin, irank, ishape, istrides, ioffset, col_major);
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

/// Assembled source of the dtype-free copy kernel (built once).
fn copy_bytes_assembled() -> &'static str {
    static SRC: OnceLock<String> = OnceLock::new();
    SRC.get_or_init(|| assemble_source(&[], COPY_BYTES_SOURCE))
}

/// Same-dtype strided assign (no cast): `c[lc] = a[la]`. Planned copy for
/// shape-equal layouts; rank/shape-changing positional zip stays generic.
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
    if lc.shape().as_ref() == la.shape().as_ref() {
        return crate::kernels::elementwise::copy_launch::<T, T>(device, c, lc, a, la, "a[ia]");
    }
    let exec = device.stream();
    // order both operands' allocation streams before the kernel (the
    // destination's includes a possibly pending allocation); order_output
    // below publishes the write back for host reads
    let _gc = order_input(c, exec)?;
    let _guard = order_input(a, exec)?;
    let key = make_key(device.ordinal_obj(), "rstsr_copy_bytes", copy_bytes_assembled(), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(lc);
    let li = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&li, true)])?;
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(a)
        .arg_i32(li.shape.len() as i32)
        .arg_u64(pack.shapes[1])
        .arg_u64(pack.strides[1])
        .arg_usize(li.offset)
        .arg_usize(lc.size())
        .arg_usize(core::mem::size_of::<T>())
        .arg_i32(col_major_of(device));
    launch.launch()?;
    // publish the write back to the destination's allocation stream, which
    // owns host-read ordering (see `order_output`)
    let _go = order_output(c, exec)?;
    Ok(())
}

/// Whether the device's default order is column-major (kernel argument).
fn col_major_of(device: &DeviceCuda) -> i32 {
    (device.default_order() == ColMajor) as i32
}

// /* #endregion */
// /* #region cast kernels */
fn cast_source(tc: &str, ta: &str, expr: &str) -> String {
    let body = format!(
        r#"
extern "C" __global__ void rstsr_cast(
    {tc}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const {ta}* in, int irank, const rstsr_usz* ishape, const rstsr_usz* istrides, rstsr_usz ioffset,
    rstsr_usz n, int col_major)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_iter_offset(lin, orank, oshape, ostrides, ooffset, col_major);
    rstsr_usz io = rstsr_iter_offset(lin, irank, ishape, istrides, ioffset, col_major);
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
            let c = if tc == "rstsr_c32" { "float" } else { "double" };
            format!("rstsr_make_complex(({c})({src}.re), ({c})({src}.im))")
        },
        // primitive -> complex (half loads via float above)
        (_, "rstsr_c32") => format!("rstsr_make_complex((float)({load}), 0.0f)"),
        (_, "rstsr_c64") => format!("rstsr_make_complex((double)({load}), 0.0)"),
        // complex -> real is a lossy dtype conversion the crate does not do
        ("rstsr_c32" | "rstsr_c64", _) => return Err(rstsr_error!(InvalidValue, "complex-to-real cast has no kernel")),
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
    if lc.shape().as_ref() == la.shape().as_ref() {
        let expr = cast_expr::<TA, TC>()?.replace("in[io]", "a[ia]");
        return crate::kernels::elementwise::copy_launch::<TC, TA>(device, c, lc, a, la, &expr);
    }
    let exec = device.stream();
    let _gc = order_input(c, exec)?;
    let _guard = order_input(a, exec)?;
    let expr = cast_expr::<TA, TC>()?;
    let includes = [TA::INCLUDES, TC::INCLUDES].concat();
    let body = cast_source(TC::CNAME, TA::CNAME, &expr);
    let key = make_key(device.ordinal_obj(), "rstsr_cast", &assemble_source(&includes, &body), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(lc);
    let li = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&li, true)])?;
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(a)
        .arg_i32(li.shape.len() as i32)
        .arg_u64(pack.shapes[1])
        .arg_u64(pack.strides[1])
        .arg_usize(li.offset)
        .arg_usize(lc.size())
        .arg_i32(col_major_of(device));
    launch.launch()?;
    let _go = order_output(c, exec)?;
    Ok(())
}

// /* #endregion */
// /* #region fill */
/// Strided fill with a host value of dtype TA (cast to TC in-kernel when
/// different; same-dtype when not). The value is marshaled in TA's
/// on-device representation.
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
    if core::any::TypeId::of::<TA>() == core::any::TypeId::of::<TC>() {
        // SAFETY: TypeId-equal TA/TC (crate TypeId fast-path idiom)
        let c_typed: &mut CudaBuffer<TA> = unsafe { transmute_ref_cross_mut(c) };
        return crate::kernels::elementwise::fill_launch::<TA, TA>(device, c_typed, lc, &fill);
    }
    let exec = device.stream();
    let _gc = order_input(c, exec)?;
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
    // SAFETY: TA is a plain scalar; reading its bytes into a byte buffer is
    // valid (Complex is repr(C) {re, im}, matching the kernel struct)
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(fill) as *const u8, scalar_bytes.as_mut_ptr(), nbytes);
    }
    let pack = LayoutPack::new(exec, &[(&lo, true)])?;
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_usize(lc.size())
        .arg_bytes(&scalar_bytes);
    launch.launch()?;
    let _go = order_output(c, exec)?;
    Ok(())
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
    // arithmetic in T's own type; integer steps wrap naturally
    out[i] = ({t})(({t}) start + ({t}) step * ({t}) i);
}}
"#
    )
}

/// Contiguous `arange`-style sequence into a fresh buffer.
pub(crate) fn sequence<T: CudaDType>(device: &DeviceCuda, n: usize, start: T, step: T) -> Result<CudaBuffer<T>> {
    let buf = CudaBuffer::<T>::new(device.stream(), n)?;
    if n == 0 {
        return Ok(buf);
    }
    let exec = device.stream();
    let key = make_key(
        device.ordinal_obj(),
        "rstsr_sequence",
        &assemble_source(T::INCLUDES, &sequence_source(T::CNAME)),
        &[],
    )?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let nbytes = core::mem::size_of::<T>();
    let mut sbytes = vec![0u8; nbytes];
    let mut stbytes = vec![0u8; nbytes];
    // SAFETY: T is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(start) as *const u8, sbytes.as_mut_ptr(), nbytes);
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(step) as *const u8, stbytes.as_mut_ptr(), nbytes);
    }
    let launch = FlatLaunch::new(exec, &f, n).arg_buf(&buf).arg_usize(n).arg_bytes(&sbytes).arg_bytes(&stbytes);
    launch.launch()?;
    Ok(buf)
}

fn linspace_source(t: &str) -> String {
    let f = if t == "float" || t == "rstsr_c32" { "float" } else { "double" };
    // complex: component-wise arithmetic (C has no struct<->scalar casts);
    // real: mirrors the CPU device's `start + i * step`, endpoint exact
    let body = if t == "rstsr_c32" || t == "rstsr_c64" {
        let make = "rstsr_make_complex";
        format!(
            r#"
    if (n <= 1) {{ out[i] = start; return; }}
    rstsr_usz denom = endpoint ? n - 1 : n;
    {f} span_re = (end.re - start.re) / ({f}) denom;
    {f} span_im = (end.im - start.im) / ({f}) denom;
    {f} ii = ({f}) i;
    out[i] = {make}(start.re + span_re * ii, start.im + span_im * ii);
    if (endpoint && i == n - 1) out[i] = end;
"#,
            f = f,
            make = make,
        )
    } else {
        format!(
            r#"
    if (n <= 1) {{ out[i] = start; return; }}
    rstsr_usz denom = endpoint ? n - 1 : n;
    {t} step = ({t})(({f}) end - ({f}) start) / ({t}) denom;
    out[i] = ({t})(({f}) start + ({f}) step * ({f}) i);
    if (endpoint && i == n - 1) out[i] = end;
"#,
            t = t,
            f = f,
        )
    };
    format!(
        r#"
extern "C" __global__ void rstsr_linspace(
    {t}* out, rstsr_usz n, {t} start, {t} end, int endpoint)
{{
    rstsr_usz i = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (i >= n) return;{body}}}
"#,
        t = t,
        body = body,
    )
}

/// Contiguous linspace into a fresh buffer (endpoint per NumPy semantics).
pub(crate) fn linspace<T: CudaDType>(
    device: &DeviceCuda,
    n: usize,
    start: T,
    end: T,
    endpoint: bool,
) -> Result<CudaBuffer<T>> {
    let buf = CudaBuffer::<T>::new(device.stream(), n)?;
    if n == 0 {
        return Ok(buf);
    }
    let exec = device.stream();
    let key = make_key(
        device.ordinal_obj(),
        "rstsr_linspace",
        &assemble_source(T::INCLUDES, &linspace_source(T::CNAME)),
        &[],
    )?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let nbytes = core::mem::size_of::<T>();
    let mut sbytes = vec![0u8; nbytes];
    let mut ebytes = vec![0u8; nbytes];
    // SAFETY: T is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(start) as *const u8, sbytes.as_mut_ptr(), nbytes);
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(end) as *const u8, ebytes.as_mut_ptr(), nbytes);
    }
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(&buf)
        .arg_usize(n)
        .arg_bytes(&sbytes)
        .arg_bytes(&ebytes)
        .arg_bytes(&(endpoint as i32).to_ne_bytes());
    launch.launch()?;
    Ok(buf)
}

// /* #endregion */
// /* #region format gather */
const GATHER_BYTES_SOURCE: &str = r#"
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

/// Assembled source of the dtype-free gather kernel (built once).
fn gather_bytes_assembled() -> &'static str {
    static SRC: OnceLock<String> = OnceLock::new();
    SRC.get_or_init(|| assemble_source(&[], GATHER_BYTES_SOURCE))
}

/// One batched D2H gather of `indices` (absolute element offsets) + single
/// sync — replaces the per-element `d2h_scalar` loop in formatting.
pub(crate) fn gather_to_host<T: Copy>(device: &DeviceCuda, raw: &CudaBuffer<T>, indices: &[usize]) -> Result<Vec<T>> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let esz = core::mem::size_of::<T>();
    let exec = device.stream();
    let _guard = order_input(raw, exec)?;
    let key = make_key(device.ordinal_obj(), "rstsr_gather_bytes", gather_bytes_assembled(), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    // staging: upload the index list, run the gather, download the values
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| (*i as u64).to_ne_bytes()).collect();
    let mut scratch = CudaBuffer::<u8>::new(exec, idx_bytes.len())?;
    // SAFETY: upload of `idx_bytes` (host slice, len matches) into the
    // freshly allocated scratch on the launch stream
    unsafe {
        let ptr = scratch.device_ptr_mut() as u64;
        map_driver_result(cudarc::driver::result::memcpy_htod_async(ptr, &idx_bytes, exec.inner().cu_stream()))?;
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
    // SAFETY: `bytes` holds `indices.len()` elements of T (the gather kernel
    // copied esz bytes per index); T: Copy has no drop glue
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, vec.as_mut_ptr(), indices.len());
        vec.set_len(indices.len());
    }
    Ok(vec)
}

// /* #endregion */
// /* #region elementwise math kernels */
/// Device scratch holding the shape/stride arrays of one launch (tiny,
/// one async upload on the launch stream). Allocated on that stream so the
/// stream-ordered free lands after the kernel; keep it alive until the
/// launch is enqueued.
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
                map_driver_result(cudarc::driver::result::memcpy_htod_async(ptr, bytes, stream.cu_stream()))?;
            }
        }
        Ok(Self { buf })
    }

    /// Device pointer to byte `offset` inside the scratch (8-aligned).
    fn ptr_at(&self, offset: usize) -> u64 {
        (self.buf.device_ptr() as u64) + offset as u64
    }
}

/// Source of a two-input elementwise kernel: `out[oo] = EXPR(a[ia], b[ib])`
/// (dtypes may each differ; `TOUT_C` is the output C type).
fn binary_source(tout_c: &str, ta_c: &str, tb_c: &str, expr: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_binary(
    {tout_c}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const {ta_c}* a, const rstsr_usz* astrides, rstsr_usz aoffset,
    const {tb_c}* b, const rstsr_usz* bstrides, rstsr_usz boffset,
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
    const {tb_c}* b, const rstsr_usz* bstrides, rstsr_usz boffset,
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
/// Rank-generic floor: shape/stride arrays ride in the scratch upload.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_binary_generic<TOUT: CudaDType, TA: CudaDType, TB: CudaDType>(
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
    // order foreign-stream inputs before the kernel; guards dropped after
    // the launch below
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let includes = [TA::INCLUDES, TB::INCLUDES, TOUT::INCLUDES].concat();
    let src = assemble_source(&includes, &binary_source(TOUT::CNAME, TA::CNAME, TB::CNAME, expr));
    let key = make_key(device.ordinal_obj(), "rstsr_binary", &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    // SAFETY: MaybeUninit<TOUT> and TOUT have identical layout; the kernel
    // writes every element it visits
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };
    let n = lc.size();
    let lo = FlatLayout::of(lc);
    let la = FlatLayout::of(la);
    let lb = FlatLayout::of(lb);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&la, false), (&lb, false)])?;
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(c_typed)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(a)
        .arg_u64(pack.strides[1])
        .arg_usize(la.offset)
        .arg_buf(b)
        .arg_u64(pack.strides[2])
        .arg_usize(lb.offset)
        .arg_usize(n);
    launch.launch()
}

/// Launch a compiled unary elementwise kernel (async, no sync).
/// Rank-generic floor: shape/stride arrays ride in the scratch upload.
pub(crate) fn launch_unary_generic<TOUT: CudaDType, TB: CudaDType>(
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
    // order a foreign-stream input before the kernel
    let _gb = order_input(b, exec)?;
    let includes = [TB::INCLUDES, TOUT::INCLUDES].concat();
    let src = assemble_source(&includes, &unary_source(TOUT::CNAME, TB::CNAME, expr));
    let key = make_key(device.ordinal_obj(), "rstsr_unary", &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    // SAFETY: MaybeUninit<TOUT> and TOUT have identical layout; the kernel
    // writes every element it visits
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };
    let n = lc.size();
    let lo = FlatLayout::of(lc);
    let lb = FlatLayout::of(lb);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&lb, false)])?;
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(c_typed)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(b)
        .arg_u64(pack.strides[1])
        .arg_usize(lb.offset)
        .arg_usize(n);
    launch.launch()
}

/* #region scalar-operand kernels */

/// Source of a tensor-scalar elementwise kernel. The scalar operand is a
/// kernel parameter (no device buffer, no H2D sync). `expr` uses `t[it]`
/// for the tensor element and `v` for the scalar.
fn scalar_source(tout_c: &str, tt_c: &str, ts_c: &str, expr: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_scalar(
    {tout_c}* out, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    const {tt_c}* t, const rstsr_usz* tstrides, rstsr_usz toffset,
    {ts_c} v, rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    rstsr_usz it = rstsr_index_to_offset(lin, orank, oshape, tstrides, toffset);
    out[oo] = {expr};
}}
"#,
    )
}

/// Launch a tensor-scalar elementwise kernel (async; the scalar rides in
/// the kernel parameter list). Rank-generic floor: shape/stride arrays
/// ride in the scratch upload.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_scalar_generic<TOUT: CudaDType, TT: CudaDType, TS: CudaDType>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    t: &CudaBuffer<TT>,
    lt: &Layout<impl DimAPI>,
    s: &TS,
    expr: &str,
) -> Result<()> {
    if lc.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    let _gt = order_input(t, exec)?;
    let includes = [TT::INCLUDES, TS::INCLUDES, TOUT::INCLUDES].concat();
    let src = assemble_source(&includes, &scalar_source(TOUT::CNAME, TT::CNAME, TS::CNAME, expr));
    let key = make_key(device.ordinal_obj(), "rstsr_scalar", &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    // SAFETY: MaybeUninit<TOUT> and TOUT share layout; every visited element
    // is written by the kernel
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };
    let n = lc.size();
    let lo = FlatLayout::of(lc);
    let lt = FlatLayout::of(lt);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&lt, false)])?;
    let nbytes = core::mem::size_of::<TS>();
    let mut sbytes = vec![0u8; nbytes];
    // SAFETY: TS is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*s) as *const u8, sbytes.as_mut_ptr(), nbytes);
    }
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(c_typed)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(t)
        .arg_u64(pack.strides[1])
        .arg_usize(lt.offset)
        .arg_bytes(&sbytes)
        .arg_usize(n);
    launch.launch()
}

/* #endregion */

/* #region in-place unary */

/// Source of an in-place unary kernel: `a[io] = EXPR` where the
/// expression reads `a[io]` (each thread owns its element).
fn unary_inplace_source(tout_c: &str, expr: &str) -> String {
    format!(
        r#"
extern "C" __global__ void rstsr_unary_inplace(
    {tout_c}* a, int orank, const rstsr_usz* oshape, const rstsr_usz* ostrides, rstsr_usz ooffset,
    rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz io = rstsr_index_to_offset(lin, orank, oshape, ostrides, ooffset);
    a[io] = {expr};
}}
"#,
    )
}

/// In-place unary kernel (each thread reads and writes its own
/// element, so no aliasing analysis is needed). Rank-generic floor.
pub(crate) fn unary_float_inplace<T>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<T>,
    la: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    T: CudaDType,
{
    if la.size() == 0 {
        return Ok(());
    }
    let exec = device.stream();
    // in-place: the buffer is both operand and destination, so order prior
    // work on its allocation stream in and the kernel back out
    let _ga = order_input(a, exec)?;
    let src = assemble_source(T::INCLUDES, &unary_inplace_source(T::CNAME, expr));
    let key = make_key(device.ordinal_obj(), "rstsr_unary_inplace", &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let n = la.size();
    let lo = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lo, true)])?;
    let launch = FlatLaunch::new(exec, &f, n)
        .arg_buf(a)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_usize(n);
    launch.launch()?;
    let _go = order_output(a, exec)?;
    Ok(())
}

/* #endregion */

/// Whether `v`'s dynamic type is in the fill-kernel's supported set.
#[doc(hidden)]
pub(crate) fn is_fill_supported<T: 'static>(_v: &T) -> bool {
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
/// into the TC=TA identity fill kernel (the same TypeId-guard idiom as the
/// CPU fast paths; the caller checked `is_fill_supported`).
pub(crate) fn fill_bytes_erased<TC, T, D>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<D>,
    fill: &T,
) -> Result<()>
where
    TC: 'static,
    T: Clone + 'static,
    D: DimAPI,
{
    /// Identity fill for the concrete scalar `V` after the caller's check.
    ///
    /// # Safety
    /// `TypeId::of::<T>() == TypeId::of::<V>() == TypeId::of::<TC>()` must
    /// hold (buffers and scalar bytes are then layout-identical).
    unsafe fn fill_identity<TC, T, V, D>(
        device: &DeviceCuda,
        c: &mut CudaBuffer<TC>,
        lc: &Layout<D>,
        fill: &T,
    ) -> Result<()>
    where
        TC: 'static,
        V: CudaDType,
        D: DimAPI,
    {
        let c_typed: &mut CudaBuffer<V> = unsafe { &mut *(c as *mut CudaBuffer<TC> as *mut CudaBuffer<V>) };
        let v: V = unsafe { core::ptr::read(fill as *const T as *const V) };
        fill_dispatch::<V, V, D>(device, c_typed, lc, v)
    }

    macro_rules! try_fill {
        ($ty:ty) => {
            if core::any::TypeId::of::<T>() == core::any::TypeId::of::<$ty>()
                && core::any::TypeId::of::<TC>() == core::any::TypeId::of::<$ty>()
            {
                // SAFETY: both TypeIds were just checked against $ty
                return unsafe { fill_identity::<TC, T, $ty, D>(device, c, lc, fill) };
            }
        };
    }
    try_fill!(f32);
    try_fill!(f64);
    try_fill!(i8);
    try_fill!(i16);
    try_fill!(i32);
    try_fill!(i64);
    try_fill!(isize);
    try_fill!(u8);
    try_fill!(u16);
    try_fill!(u32);
    try_fill!(u64);
    try_fill!(usize);
    try_fill!(bool);
    rstsr_raise!(InvalidValue, "fill dtype not supported by the CUDA fill kernel")?
}

/// Identity-dtype fill through the generic strided fill kernel (planned
/// path fallback; caller proved TypeId(TA) == TypeId(TC)).
pub(crate) fn fill_dispatch_same<TA, TC, D>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<D>,
    v: &TA,
) -> Result<()>
where
    TA: CudaDType,
    TC: CudaDType,
    D: DimAPI,
{
    // SAFETY: caller proved TypeId equality (buffers layout-identical)
    let c_typed: &mut CudaBuffer<TA> = unsafe { transmute_ref_cross_mut(c) };
    let v_typed: TA = unsafe { core::ptr::read(v as *const TA) };
    fill_dispatch(device, c_typed, lc, v_typed)
}

/// Generic strided copy with a caller-provided element expression (the
/// planned copy's fallback when plans would be Generic; shape-equality is
/// the caller's contract, cast_expr reads `a[ia]` and yields TC).
pub(crate) fn copy_dispatch_generic<TC, TA>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    cast_expr: &str,
) -> Result<()>
where
    TC: CudaDType,
    TA: CudaDType,
{
    let exec = device.stream();
    let _gc = order_input(c, exec)?;
    let _ga = order_input(a, exec)?;
    let body = cast_source(TC::CNAME, TA::CNAME, cast_expr);
    let includes = [TA::INCLUDES, TC::INCLUDES].concat();
    let key = make_key(device.ordinal_obj(), "rstsr_cast", &assemble_source(&includes, &body), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(lc);
    let li = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&li, true)])?;
    let launch = FlatLaunch::new(exec, &f, lc.size())
        .arg_buf(c)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(a)
        .arg_i32(li.shape.len() as i32)
        .arg_u64(pack.shapes[1])
        .arg_u64(pack.strides[1])
        .arg_usize(li.offset)
        .arg_usize(lc.size())
        .arg_i32(col_major_of(device));
    launch.launch()?;
    let _go = order_output(c, exec)?;
    Ok(())
}

/// Generic in-place binary (compound assign) floor: `t[it] = EXPR` with
/// `t` decomposing by its own layout (broadcast destination rejected at
/// the tensor layer); `EXPR` reads `t[it]` and `b[ib]`.
pub(crate) fn binary_assign_generic<TA, TB>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    TA: CudaDType,
    TB: CudaDType,
{
    let src = format!(
        r#"
extern "C" __global__ void rstsr_binary_assign_g(
    {ta}* t, int trank, const rstsr_usz* tshape, const rstsr_usz* tstrides, rstsr_usz toffset,
    const {tb}* b, const rstsr_usz* bstrides, rstsr_usz boffset,
    rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz it = rstsr_index_to_offset(lin, trank, tshape, tstrides, toffset);
    rstsr_usz ib = rstsr_index_to_offset(lin, trank, tshape, bstrides, boffset);
    t[it] = {expr};
}}
"#,
        ta = TA::CNAME,
        tb = TB::CNAME,
    );
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let includes = [TA::INCLUDES, TB::INCLUDES].concat();
    let key = make_key(device.ordinal_obj(), "rstsr_binary_assign_g", &assemble_source(&includes, &src), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(la);
    let lb2 = FlatLayout::of(lb);
    let pack = LayoutPack::new(exec, &[(&lo, true), (&lb2, false)])?;
    let launch = FlatLaunch::new(exec, &f, la.size())
        .arg_buf(a)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_buf(b)
        .arg_u64(pack.strides[1])
        .arg_usize(lb2.offset)
        .arg_usize(la.size());
    launch.launch()?;
    let _go = order_output(a, exec)?;
    Ok(())
}

/// Generic in-place tensor-scalar (compound assign, scalar RHS) floor.
pub(crate) fn scalar_assign_generic<TA, TB>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &TB,
    expr: &str,
) -> Result<()>
where
    TA: CudaDType,
    TB: CudaDType,
{
    let nbytes = core::mem::size_of::<TB>();
    let mut bbytes = vec![0u8; nbytes];
    // SAFETY: TB is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*b) as *const u8, bbytes.as_mut_ptr(), nbytes);
    }
    let src = format!(
        r#"
extern "C" __global__ void rstsr_scalar_assign_g(
    {ta}* t, int trank, const rstsr_usz* tshape, const rstsr_usz* tstrides, rstsr_usz toffset,
    {tb} v, rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz it = rstsr_index_to_offset(lin, trank, tshape, tstrides, toffset);
    t[it] = {expr};
}}
"#,
        ta = TA::CNAME,
        tb = TB::CNAME,
    );
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let includes = [TA::INCLUDES, TB::INCLUDES].concat();
    let key = make_key(device.ordinal_obj(), "rstsr_scalar_assign_g", &assemble_source(&includes, &src), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lo = FlatLayout::of(la);
    let pack = LayoutPack::new(exec, &[(&lo, true)])?;
    let launch = FlatLaunch::new(exec, &f, la.size())
        .arg_buf(a)
        .arg_i32(lo.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lo.offset)
        .arg_bytes(&bbytes)
        .arg_usize(la.size());
    launch.launch()?;
    let _go = order_output(a, exec)?;
    Ok(())
}

/// Generic ternary select floor (`d = a ? b : c`, runtime strides).
#[allow(clippy::too_many_arguments)]
pub(crate) fn where_generic<TD, TB, TC>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<impl DimAPI>,
    a: &CudaBuffer<bool>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    c: &CudaBuffer<TC>,
    lc: &Layout<impl DimAPI>,
) -> Result<()>
where
    TD: CudaDType,
    TB: CudaDType,
    TC: CudaDType,
{
    let src = format!(
        r#"
extern "C" __global__ void rstsr_where_g(
    {td}* d, int rank, const rstsr_usz* shape, const rstsr_usz* dstr, rstsr_usz doff,
    const bool* a, const rstsr_usz* astr, rstsr_usz aoff,
    const {tb}* b, const rstsr_usz* bstr, rstsr_usz boff,
    const {tc}* c, const rstsr_usz* cstr, rstsr_usz coff,
    rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz id = rstsr_index_to_offset(lin, rank, shape, dstr, doff);
    rstsr_usz ia = rstsr_index_to_offset(lin, rank, shape, astr, aoff);
    rstsr_usz ib = rstsr_index_to_offset(lin, rank, shape, bstr, boff);
    rstsr_usz ic = rstsr_index_to_offset(lin, rank, shape, cstr, coff);
    d[id] = a[ia] ? b[ib] : c[ic];
}}
"#,
        td = TD::CNAME,
        tb = TB::CNAME,
        tc = TC::CNAME,
    );
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let _gc = order_input(c, exec)?;
    let includes = [TB::INCLUDES, TC::INCLUDES, TD::INCLUDES].concat();
    let key = make_key(device.ordinal_obj(), "rstsr_where_g", &assemble_source(&includes, &src), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lod = FlatLayout::of(ld);
    let loa = FlatLayout::of(la);
    let lob = FlatLayout::of(lb);
    let loc = FlatLayout::of(lc);
    // all four operands share the output's logical shape at the tensor
    // tier (broadcast applied before the device call): one shape array
    // (packed with the output) + four stride arrays
    let pack = LayoutPack::new(exec, &[(&lod, true), (&loa, false), (&lob, false), (&loc, false)])?;
    let launch = FlatLaunch::new(exec, &f, ld.size())
        .arg_buf(d)
        .arg_i32(lod.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lod.offset)
        .arg_buf(a)
        .arg_u64(pack.strides[1])
        .arg_usize(loa.offset)
        .arg_buf(b)
        .arg_u64(pack.strides[2])
        .arg_usize(lob.offset)
        .arg_buf(c)
        .arg_u64(pack.strides[3])
        .arg_usize(loc.offset)
        .arg_usize(ld.size());
    launch.launch()?;
    let _go = order_output(d, exec)?;
    Ok(())
}

/// Generic where scalar-c floor.
pub(crate) fn where_sc_generic<TD, TB, TC>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<impl DimAPI>,
    a: &CudaBuffer<bool>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    c: &TC,
) -> Result<()>
where
    TD: CudaDType,
    TB: CudaDType,
    TC: CudaDType,
{
    let nb = core::mem::size_of::<TC>();
    let mut cb = vec![0u8; nb];
    // SAFETY: TC is a plain scalar; bytes are its device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*c) as *const u8, cb.as_mut_ptr(), nb);
    }
    let src = format!(
        r#"
extern "C" __global__ void rstsr_where_sc_g(
    {td}* d, int rank, const rstsr_usz* shape, const rstsr_usz* dstr, rstsr_usz doff,
    const bool* a, const rstsr_usz* astr, rstsr_usz aoff,
    const {tb}* b, const rstsr_usz* bstr, rstsr_usz boff,
    {tc} v, rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz id = rstsr_index_to_offset(lin, rank, shape, dstr, doff);
    rstsr_usz ia = rstsr_index_to_offset(lin, rank, shape, astr, aoff);
    rstsr_usz ib = rstsr_index_to_offset(lin, rank, shape, bstr, boff);
    d[id] = a[ia] ? b[ib] : v;
}}
"#,
        td = TD::CNAME,
        tb = TB::CNAME,
        tc = TC::CNAME,
    );
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let includes = [TB::INCLUDES, TC::INCLUDES, TD::INCLUDES].concat();
    let key = make_key(device.ordinal_obj(), "rstsr_where_sc_g", &assemble_source(&includes, &src), &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lod = FlatLayout::of(ld);
    let loa = FlatLayout::of(la);
    let lob = FlatLayout::of(lb);
    let pack = LayoutPack::new(exec, &[(&lod, true), (&loa, false), (&lob, false)])?;
    let launch = FlatLaunch::new(exec, &f, ld.size())
        .arg_buf(d)
        .arg_i32(lod.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lod.offset)
        .arg_buf(a)
        .arg_u64(pack.strides[1])
        .arg_usize(loa.offset)
        .arg_buf(b)
        .arg_u64(pack.strides[2])
        .arg_usize(lob.offset)
        .arg_bytes(&cb)
        .arg_usize(ld.size());
    launch.launch()?;
    let _go = order_output(d, exec)?;
    Ok(())
}

/// Generic clip floor (runtime strides; bound kinds as ClipBound).
#[allow(clippy::too_many_arguments)]
pub(crate) fn clip_generic<TD, TB, TC>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<impl DimAPI>,
    a: &CudaBuffer<TD>,
    la: &Layout<impl DimAPI>,
    lo: crate::kernels::elementwise::ClipBound<TB, impl DimAPI>,
    hi: crate::kernels::elementwise::ClipBound<TC, impl DimAPI>,
    max_expr: &str,
    min_expr: &str,
) -> Result<()>
where
    TD: CudaDType,
    TB: CudaDType,
    TC: CudaDType,
{
    use crate::kernels::elementwise::ClipBound;
    let nb = core::mem::size_of::<TD>();
    let mut blob = vec![0u8; nb];
    let mut hib = vec![0u8; nb];
    // SAFETY: scalar bytes (TD-sized, same-type promotion at tensor tier)
    unsafe {
        if let ClipBound::Scalar(v) = &lo {
            core::ptr::copy_nonoverlapping(core::ptr::addr_of!(**v) as *const u8, blob.as_mut_ptr(), nb);
        }
        if let ClipBound::Scalar(v) = &hi {
            core::ptr::copy_nonoverlapping(core::ptr::addr_of!(**v) as *const u8, hib.as_mut_ptr(), nb);
        }
    }
    let src = format!(
        r#"
extern "C" __global__ void rstsr_clip_g(
    {t}* d, int rank, const rstsr_usz* shape, const rstsr_usz* dstr, rstsr_usz doff,
    const {t}* a, const rstsr_usz* astr, rstsr_usz aoff,
    const {tb}* b, const rstsr_usz* bstr, rstsr_usz boff, int b_kind,
    const {tc}* c, const rstsr_usz* cstr, rstsr_usz coff, int c_kind,
    {tb} v_lo, {tc} v_hi, rstsr_usz n)
{{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz id = rstsr_index_to_offset(lin, rank, shape, dstr, doff);
    rstsr_usz x = rstsr_index_to_offset(lin, rank, shape, astr, aoff);
    {t} val = a[x];
    if (b_kind != 0) {{
        {t} lo = (b_kind == 1) ? b[rstsr_index_to_offset(lin, rank, shape, bstr, boff)] : ({t}) v_lo;
        val = {max_expr};
    }}
    if (c_kind != 0) {{
        {t} hi = (c_kind == 1) ? c[rstsr_index_to_offset(lin, rank, shape, cstr, coff)] : ({t}) v_hi;
        val = {min_expr};
    }}
    d[id] = val;
}}
"#,
        t = TD::CNAME,
        tb = TB::CNAME,
        tc = TC::CNAME,
        max_expr = max_expr,
        min_expr = min_expr,
    );
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    if let ClipBound::Tensor(buf, _) = &lo {
        let _ = order_input(buf, exec)?;
    }
    if let ClipBound::Tensor(buf, _) = &hi {
        let _ = order_input(buf, exec)?;
    }
    let key = make_key(
        device.ordinal_obj(),
        "rstsr_clip_g",
        &assemble_source(&[TD::INCLUDES, TB::INCLUDES, TC::INCLUDES].concat(), &src),
        &[],
    )?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    let lod = FlatLayout::of(ld);
    let loa = FlatLayout::of(la);
    let (lob, loc) = (
        match &lo {
            ClipBound::Tensor(_, lb) => FlatLayout::of(lb),
            _ => FlatLayout::of(ld), // unused; packed for pointer stability
        },
        match &hi {
            ClipBound::Tensor(_, lc) => FlatLayout::of(lc),
            _ => FlatLayout::of(ld),
        },
    );
    let pack = LayoutPack::new(exec, &[(&lod, true), (&loa, false), (&lob, false), (&loc, false)])?;
    let b_kind = match &lo {
        ClipBound::None => 0i32,
        ClipBound::Scalar(_) => 2,
        ClipBound::Tensor(_, _) => 1,
    };
    let c_kind = match &hi {
        ClipBound::None => 0i32,
        ClipBound::Scalar(_) => 2,
        ClipBound::Tensor(_, _) => 1,
    };
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let d_typed: &mut CudaBuffer<TD> = unsafe { &mut *(d as *mut CudaBuffer<MaybeUninit<TD>> as *mut CudaBuffer<TD>) };
    let launch = FlatLaunch::new(exec, &f, ld.size())
        .arg_buf(d_typed)
        .arg_i32(lod.shape.len() as i32)
        .arg_u64(pack.shapes[0])
        .arg_u64(pack.strides[0])
        .arg_usize(lod.offset)
        .arg_buf(a)
        .arg_u64(pack.strides[1])
        .arg_usize(loa.offset)
        .arg_u64(match &lo {
            ClipBound::Tensor(buf, lb2) => {
                let _ = lb2;
                buf.device_ptr() as u64
            },
            _ => 0,
        })
        .arg_u64(pack.strides[2])
        .arg_usize(match &lo {
            ClipBound::Tensor(_, lb2) => lb2.offset(),
            _ => 0,
        })
        .arg_i32(b_kind)
        .arg_u64(match &hi {
            ClipBound::Tensor(buf, lc2) => {
                let _ = lc2;
                buf.device_ptr() as u64
            },
            _ => 0,
        })
        .arg_u64(pack.strides[3])
        .arg_usize(match &hi {
            ClipBound::Tensor(_, lc2) => lc2.offset(),
            _ => 0,
        })
        .arg_i32(c_kind)
        .arg_bytes(&blob)
        .arg_bytes(&hib)
        .arg_usize(ld.size());
    launch.launch()?;
    let _go = order_output(d, exec)?;
    Ok(())
}
