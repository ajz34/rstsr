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
pub trait CudaDType: Clone + Send + Sync + Debug + 'static {
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
    [Complex<f32>]      ["rstsr_c32"];
    [Complex<f64>]      ["rstsr_c64"];
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
