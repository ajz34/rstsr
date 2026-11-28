//! The nine cuTENSOR compute descriptors.
//!
//! cuTENSOR stores the `CUTENSOR_COMPUTE_DESC_*` values as exported `const` globals, not enum
//! constants, so cudarc's bindings do not carry them. Each function here resolves its symbol once
//! per process; a missing library or symbol is an `UnImplemented` error, never a panic.
//!
//! The descriptor is the *minimum* accuracy guarantee of the internal arithmetic, not an exact
//! specification — cuTENSOR may compute at higher precision. Which descriptors a dtype can serve
//! is the library's own table: an unsupported combination surfaces as `NOT_SUPPORTED` at plan
//! creation and maps to a decline ([`crate::cutensor_wrap::is_decline`]).
//!
//! `desc_9x16bf` needs cuTENSOR 2.2+, `desc_8xint8` 2.4+, `desc_4x16f` 2.5+; on an older library
//! the symbol lookup fails and the function returns the load error.

use crate::cutensor_wrap::error;
use cudarc::cutensor::sys;
use rstsr_core::prelude_dev::*;
use std::sync::OnceLock;

#[cfg(cutensor_dynamic_loading)]
fn symbol(name: &[u8]) -> Result<sys::cutensorComputeDescriptor_t> {
    // culib() panics when nothing loads; probe first so the failure is an error.
    if !unsafe { sys::is_culib_present() } {
        return Err(error::not_loadable());
    }
    // SAFETY: `name` is NUL-terminated. The symbol addresses a `const cutensorComputeDescriptor_t`
    // global, so one dereference of the looked-up address copies the descriptor pointer value.
    let sym =
        unsafe { sys::culib().get::<sys::cutensorComputeDescriptor_t>(name) }.map_err(|_| error::not_loadable())?;
    Ok(*sym)
}

#[cfg(not(cutensor_dynamic_loading))]
fn symbol(name: &[u8]) -> Result<sys::cutensorComputeDescriptor_t> {
    extern "C" {
        static CUTENSOR_COMPUTE_DESC_16F: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_16BF: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_TF32: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_3XTF32: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_32F: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_64F: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_9X16BF: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_8XINT8: sys::cutensorComputeDescriptor_t;
        static CUTENSOR_COMPUTE_DESC_4X16F: sys::cutensorComputeDescriptor_t;
    }
    // SAFETY: the statics are resolved by the linker (cudarc links `-lcutensor` under
    // `dynamic-linking`); reading a `const` pointer value from data memory.
    Ok(match name {
        b"CUTENSOR_COMPUTE_DESC_16F\0" => unsafe { CUTENSOR_COMPUTE_DESC_16F },
        b"CUTENSOR_COMPUTE_DESC_16BF\0" => unsafe { CUTENSOR_COMPUTE_DESC_16BF },
        b"CUTENSOR_COMPUTE_DESC_TF32\0" => unsafe { CUTENSOR_COMPUTE_DESC_TF32 },
        b"CUTENSOR_COMPUTE_DESC_3XTF32\0" => unsafe { CUTENSOR_COMPUTE_DESC_3XTF32 },
        b"CUTENSOR_COMPUTE_DESC_32F\0" => unsafe { CUTENSOR_COMPUTE_DESC_32F },
        b"CUTENSOR_COMPUTE_DESC_64F\0" => unsafe { CUTENSOR_COMPUTE_DESC_64F },
        b"CUTENSOR_COMPUTE_DESC_9X16BF\0" => unsafe { CUTENSOR_COMPUTE_DESC_9X16BF },
        b"CUTENSOR_COMPUTE_DESC_8XINT8\0" => unsafe { CUTENSOR_COMPUTE_DESC_8XINT8 },
        b"CUTENSOR_COMPUTE_DESC_4X16F\0" => unsafe { CUTENSOR_COMPUTE_DESC_4X16F },
        _ => return Err(error::not_loadable()),
    })
}

/// A resolved descriptor, `Sync` so it can sit in a static slot: the pointer addresses
/// library-owned state valid for the process lifetime, and the descriptor is immutable
/// (`cutensorCreateTensorDescriptor` etc. are documented thread-safe, and this one is a `const`
/// global by construction).
struct Desc(sys::cutensorComputeDescriptor_t);
unsafe impl Send for Desc {}
unsafe impl Sync for Desc {}

macro_rules! compute_descriptor {
    ($($(#[$meta:meta])* $fn_name:ident => $symbol:literal),* $(,)?) => {
        $(
            $(#[$meta])*
            pub fn $fn_name() -> Result<sys::cutensorComputeDescriptor_t> {
                // Every `symbol` failure is the not-loadable error, so the cached verdict keeps
                // a `()` placeholder instead of the unclonable `Error`.
                static SLOT: OnceLock<core::result::Result<Desc, ()>> = OnceLock::new();
                match SLOT.get_or_init(|| symbol($symbol).map_err(|_| ()).map(Desc)) {
                    Ok(desc) => Ok(desc.0),
                    Err(()) => Err(error::not_loadable()),
                }
            }
        )*
    };
}

compute_descriptor! {
    /// Half-precision products (`16F`).
    desc_16f => b"CUTENSOR_COMPUTE_DESC_16F\0",
    /// Bfloat-precision products (`16BF`).
    desc_16bf => b"CUTENSOR_COMPUTE_DESC_16BF\0",
    /// TensorFloat-32 products, fp32 accumulate (`TF32`).
    desc_tf32 => b"CUTENSOR_COMPUTE_DESC_TF32\0",
    /// Three TF32 products per fp32 multiply, near-fp32 accuracy (`3XTF32`).
    desc_3xtf32 => b"CUTENSOR_COMPUTE_DESC_3XTF32\0",
    /// Plain fp32 (`32F`); the strict choice for `f32`/`c32` storage.
    desc_32f => b"CUTENSOR_COMPUTE_DESC_32F\0",
    /// Plain fp64 (`64F`); the strict choice for `f64`/`c64` storage.
    desc_64f => b"CUTENSOR_COMPUTE_DESC_64F\0",
    /// Three bf16 values per fp32 multiply, 23 mantissa bits (`9X16BF`, Hopper+).
    desc_9x16bf => b"CUTENSOR_COMPUTE_DESC_9X16BF\0",
    /// INT8-based fp64 emulation (`8XINT8`, Hopper+).
    desc_8xint8 => b"CUTENSOR_COMPUTE_DESC_8XINT8\0",
    /// Two fp16 values per fp32 multiply, 23 mantissa bits (`4X16F`, Blackwell+).
    desc_4x16f => b"CUTENSOR_COMPUTE_DESC_4X16F\0",
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every accessor returns — an error without the library, the descriptor with it — and never
    /// panics (cudarc's own loader panics when nothing loads; this module must not).
    #[test]
    fn accessors_never_panic() {
        let accessors: [fn() -> Result<sys::cutensorComputeDescriptor_t>; 9] =
            [desc_16f, desc_16bf, desc_tf32, desc_3xtf32, desc_32f, desc_64f, desc_9x16bf, desc_8xint8, desc_4x16f];
        for f in accessors {
            let _ = f();
        }
    }

    /// On hardware with the library loadable, the two strict descriptors resolve to distinct
    /// non-null pointers.
    #[test]
    fn descriptors_resolve_on_hardware() {
        if !crate::registry::is_available() {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
        if !crate::cutensor_wrap::handle::libcutensor_available() {
            eprintln!("skipping: libcutensor is not loadable");
            return;
        }
        let f32d = desc_32f().expect("desc_32f");
        let f64d = desc_64f().expect("desc_64f");
        assert!(!f32d.is_null());
        assert!(!f64d.is_null());
        assert_ne!(f32d, f64d);
    }
}
