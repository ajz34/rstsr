//! `CudaDType`: device-side scalar capability traits for kernel generation.
//!
//! Kernel families are generic over these traits; the Rust monomorphization
//! is the dtype dispatch — kernel source is built from `T::CNAME` and the
//! cache is keyed by the same string. Unsupported dtypes fail by missing
//! bound (compile error), matching the "omit" policy for host-closure ops.

use core::fmt::Debug;
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

macro_rules! impl_cuda_dtype {
    ($t: ty, $name: literal) => {
        impl CudaDType for $t {
            const CNAME: &'static str = $name;
        }
    };
    ($t: ty, $name: literal, [$($inc: literal),*]) => {
        impl CudaDType for $t {
            const CNAME: &'static str = $name;
            const INCLUDES: &'static [&'static str] = &[$($inc),*];
        }
    };
}

impl_cuda_dtype!(bool, "bool");
impl_cuda_dtype!(i8, "signed char");
impl_cuda_dtype!(i16, "short");
impl_cuda_dtype!(i32, "int");
impl_cuda_dtype!(i64, "long long");
impl_cuda_dtype!(isize, "long long");
impl_cuda_dtype!(u8, "unsigned char");
impl_cuda_dtype!(u16, "unsigned short");
impl_cuda_dtype!(u32, "unsigned int");
impl_cuda_dtype!(u64, "unsigned long long");
impl_cuda_dtype!(usize, "unsigned long long");
impl_cuda_dtype!(f32, "float");
impl_cuda_dtype!(f64, "double");
impl_cuda_dtype!(f16, "__half", ["#include <cuda_fp16.h>"]);
impl_cuda_dtype!(bf16, "__nv_bfloat16", ["#include <cuda_bf16.h>"]);
impl_cuda_dtype!(Complex<f32>, "rstsr_c32");
impl_cuda_dtype!(Complex<f64>, "rstsr_c64");

/// Types with native CUDA C arithmetic (`+ - * /`); the fill/copy/free
/// families do not need this, kernel math does. Half is excluded (needs
/// explicit promotion wrappers; a later wave).
pub trait CudaArith: CudaDType {}
macro_rules! impl_cuda_arith {
    ($($t: ty),*) => { $(impl CudaArith for $t {})* };
}
impl_cuda_arith!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64, Complex<f32>, Complex<f64>);

/// Float types with CUDA C transcendental functions (f32/f64; a later wave
/// adds half via float promotion and complex via a complex-math prelude).
pub trait CudaFloat: CudaArith {}
impl CudaFloat for f32 {}
impl CudaFloat for f64 {}

/// Types with native CUDA C ordering/equality comparisons (`< > <= >= == !=`
/// produce a scalar bool). Complex is excluded (component-wise comparisons
/// are not orderings; rstsr declines them at the tensor layer as CPU does).
pub trait CudaCmp: CudaDType {}
macro_rules! impl_cuda_cmp {
    ($($t: ty),*) => { $(impl CudaCmp for $t {})* };
}
impl_cuda_cmp!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64, f16, bf16);
