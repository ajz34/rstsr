//! `CudaDType`: device-side scalar capability traits for kernel generation.
//!
//! Kernel families are generic over these traits; the Rust monomorphization
//! is the dtype dispatch — kernel source is built from `T::CNAME` and the
//! cache is keyed by the same string. Unsupported dtypes fail by missing
//! bound (compile error), matching the "omit" policy for host-closure ops.

use core::fmt::Debug;
use duplicate::duplicate_item;
use half::{bf16, f16};
use num::Complex;

/// A scalar type storable and kernel-representable on `DeviceCuda`.
///
/// `CNAME` is the CUDA C type name used in generated kernel signatures and
/// in the kernel-cache key. (i128/u128 have no portable CUDA C type and
/// intentionally have no impl; they keep host round-trip paths.)
pub trait CudaDType: Copy + Send + Sync + Debug + 'static {
    /// CUDA C scalar name of this dtype.
    const CNAME: &'static str;
    /// Byte size of the on-device representation.
    const NBYTES: usize = core::mem::size_of::<Self>();
    /// Extra `#include` lines required by this dtype (half headers, ...).
    const INCLUDES: &'static [&'static str] = &[];
}

#[duplicate_item(
    T   name;
    [bool]              ["bool"];
    [i8]                ["signed char"];
    [i16]               ["short"];
    [i32]               ["int"];
    [i64]               ["long long"];
    [isize]             ["long long"];
    [u8]                ["unsigned char"];
    [u16]               ["unsigned short"];
    [u32]               ["unsigned int"];
    [u64]               ["unsigned long long"];
    [usize]             ["unsigned long long"];
    [f32]               ["float"];
    [f64]               ["double"];
)]
impl CudaDType for T {
    const CNAME: &'static str = name;
}

#[duplicate_item(
    T      name             inc;
    [f16]  ["__half"]       ["#include <cuda_fp16.h>"];
    [bf16] ["__nv_bfloat16"] ["#include <cuda_bf16.h>"];
)]
impl CudaDType for T {
    const CNAME: &'static str = name;
    const INCLUDES: &'static [&'static str] = &[inc];
}

/// Types with native CUDA C arithmetic (`+ - * /`); the fill/copy/free
/// families do not need this, kernel math does. Half is excluded (needs
/// explicit promotion wrappers; a later wave).
pub trait CudaArith: CudaDType {}
#[duplicate_item(
    T;
    [i8]; [i16]; [i32]; [i64]; [isize];
    [u8]; [u16]; [u32]; [u64]; [usize];
    [f32]; [f64]; [Complex<f32>]; [Complex<f64>];
)]
impl CudaArith for T {}

/// Float types with CUDA C transcendental functions (f32/f64; a later wave
/// adds half via float promotion and complex via a complex-math prelude).
pub trait CudaFloat: CudaArith {}
impl CudaFloat for f32 {}
impl CudaFloat for f64 {}

/// Types with native CUDA C ordering/equality comparisons (`< > <= >= == !=`
/// produce a scalar bool). Complex is excluded (component-wise comparisons
/// are not orderings; rstsr declines them at the tensor layer as CPU does).
pub trait CudaCmp: CudaDType {}
#[duplicate_item(
    T;
    [bool]; [i8]; [i16]; [i32]; [i64]; [isize];
    [u8]; [u16]; [u32]; [u64]; [usize];
    [f32]; [f64]; [f16]; [bf16];
)]
impl CudaCmp for T {}

/// Integer types with CUDA C bitwise/shift operators (`& | ^ << >>`).
/// Bool is excluded here (its bitwise ops are emitted as logical `&&/||`
/// and `!`-based xor by the op layer); shifts are only generated for
/// these types.
pub trait CudaInt: CudaDType {}
#[duplicate_item(
    T;
    [i8]; [i16]; [i32]; [i64]; [isize];
    [u8]; [u16]; [u32]; [u64]; [usize];
)]
impl CudaInt for T {}

/// Real types with CUDA C `<`/`>` and `fmin`/`fmax`-class ordering
/// (everything CudaCmp covers minus the halfs, whose device min/max
/// compare through float promotion in the op expressions).
pub trait CudaReal: CudaDType {}
#[duplicate_item(
    T;
    [bool]; [i8]; [i16]; [i32]; [i64]; [isize];
    [u8]; [u16]; [u32]; [u64]; [usize];
    [f32]; [f64];
)]
impl CudaReal for T {}

/// Complex-capable types for the mixed complex kernels
/// (primitive→complex promotion expressions exist in `cast_expr_c`).
pub trait CudaComplexFrom: CudaDType {}
#[duplicate_item(
    T;
    [i8]; [i16]; [i32]; [i64]; [isize];
    [u8]; [u16]; [u32]; [u64]; [usize];
    [f32]; [f64];
)]
impl CudaComplexFrom for T {}
impl CudaComplexFrom for Complex<f32> {}
impl CudaComplexFrom for Complex<f64> {}

// complex dtypes: the INCLUDES marker is replaced by the complex-math
// header content at source-assembly time (a custom .cuh cannot ride the
// NVRTC builtin include path; see kernels/mod.rs COMPLEX_HEADER).
impl CudaDType for Complex<f32> {
    const CNAME: &'static str = "rstsr_c32";
    const INCLUDES: &'static [&'static str] = &["//@RSTSR_COMPLEX_HEADER@//"];
}
impl CudaDType for Complex<f64> {
    const CNAME: &'static str = "rstsr_c64";
    const INCLUDES: &'static [&'static str] = &["//@RSTSR_COMPLEX_HEADER@//"];
}

/// C names of the complex dtypes (expression-level dispatch).
pub const COMPLEX_CNAMES: [&str; 2] = ["rstsr_c32", "rstsr_c64"];

/// Whether a C type name is a rstsr complex pair.
pub fn is_complex_cname(t: &str) -> bool {
    COMPLEX_CNAMES.contains(&t)
}

/// Whether a C type name is a half dtype.
pub fn is_half_cname(t: &str) -> bool {
    t == "__half" || t == "__nv_bfloat16"
}
