//! Dtype mapping: the storage types cuTENSOR serves, their `cudaDataType_t`s and host scalars.

use crate::dtype::CudaDType;
use cudarc::cutensor::sys;
use num::Complex;
use rstsr_core::prelude_dev::*;

/// The host scalar type of `alpha` / `beta` / `gamma`.
///
/// cuTENSOR reads these from **host** pointers and decides their type from the output dtype and
/// the compute descriptor ("its data type is determined by 'descCompute'"); it never checks the
/// pointer, so a wrong type is silently wrong (e.g. an all-zeros result). Scalars travel through
/// [`AlignedScale`], and the operation descriptor's `CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE` is
/// verified against [`CutensorScalar::DATA_TYPE`] when an operation is created.
pub trait CutensorScalar: Copy + Send + Sync + Debug + 'static {
    /// The `cudaDataType_t` of this scalar.
    const DATA_TYPE: sys::cudaDataType_t;
}

impl CutensorScalar for f32 {
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_R_32F;
}

impl CutensorScalar for f64 {
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_R_64F;
}

impl CutensorScalar for Complex<f32> {
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_C_32F;
}

impl CutensorScalar for Complex<f64> {
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_C_64F;
}

/// A storage dtype cuTENSOR can serve: `f32`, `f64`, `f16`, `bf16`, `Complex<f32>`,
/// `Complex<f64>`. No integer or bool (cuTENSOR creates no descriptor for them: `R32I` fails with
/// `INTERNAL_ERROR`, `R8U` with `NOT_SUPPORTED` at plan).
///
/// `f16`/`bf16` operands take `f32` scalars (cuTENSOR's contraction table: the scalar type of a
/// half output under a `32F` compute descriptor is `R_32F`).
pub trait CutensorDType: CudaDType {
    /// Host scalar of the `alpha` / `beta` / `gamma` arguments.
    type Scalar: CutensorScalar;
    /// The `cudaDataType_t` of this storage dtype.
    const DATA_TYPE: sys::cudaDataType_t;
}

impl CutensorDType for f32 {
    type Scalar = f32;
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_R_32F;
}

impl CutensorDType for f64 {
    type Scalar = f64;
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_R_64F;
}

impl CutensorDType for half::f16 {
    type Scalar = f32;
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_R_16F;
}

impl CutensorDType for half::bf16 {
    type Scalar = f32;
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_R_16BF;
}

impl CutensorDType for Complex<f32> {
    type Scalar = Complex<f32>;
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_C_32F;
}

impl CutensorDType for Complex<f64> {
    type Scalar = Complex<f64>;
    const DATA_TYPE: sys::cudaDataType_t = sys::cudaDataType_t::CUDA_C_64F;
}

/// Host storage for one scalar passed to cuTENSOR by pointer, over-aligned so any load width the
/// library picks is safe (the cuBLASLt scale lesson: it loads 16-byte `c64` scales with aligned
/// SIMD loads).
#[allow(dead_code)] // wired by the B2 scalar passing
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignedScale<T>(pub T);

impl<T> AlignedScale<T> {
    #[allow(dead_code)]
    pub(crate) fn new(value: T) -> Self {
        Self(value)
    }

    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> *const T {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dtype table matches cuTENSOR's header values; the scalar types are exact.
    #[test]
    fn dtype_table() {
        use sys::cudaDataType_t as D;
        macro_rules! check {
            ($t:ty, $dt:expr, $s:ty, $sd:expr) => {{
                assert_eq!(<$t as CutensorDType>::DATA_TYPE, $dt, "{}", <$t as CudaDType>::CNAME);
                assert_eq!(
                    <<$t as CutensorDType>::Scalar as CutensorScalar>::DATA_TYPE,
                    $sd,
                    "{} scalar DATA_TYPE",
                    <$t as CudaDType>::CNAME
                );
                assert_eq!(
                    core::any::TypeId::of::<<$t as CutensorDType>::Scalar>(),
                    core::any::TypeId::of::<$s>(),
                    "{} scalar type",
                    <$t as CudaDType>::CNAME
                );
            }};
        }
        check!(f32, D::CUDA_R_32F, f32, D::CUDA_R_32F);
        check!(f64, D::CUDA_R_64F, f64, D::CUDA_R_64F);
        check!(half::f16, D::CUDA_R_16F, f32, D::CUDA_R_32F);
        check!(half::bf16, D::CUDA_R_16BF, f32, D::CUDA_R_32F);
        check!(Complex<f32>, D::CUDA_C_32F, Complex<f32>, D::CUDA_C_32F);
        check!(Complex<f64>, D::CUDA_C_64F, Complex<f64>, D::CUDA_C_64F);
    }

    /// The scalar storage keeps its 16-byte alignment for every scalar type.
    #[test]
    fn aligned_scale_is_16_byte_aligned() {
        for ok in [
            core::mem::align_of::<AlignedScale<f32>>() >= 16,
            core::mem::align_of::<AlignedScale<f64>>() >= 16,
            core::mem::align_of::<AlignedScale<Complex<f32>>>() >= 16,
            core::mem::align_of::<AlignedScale<Complex<f64>>>() >= 16,
        ] {
            assert!(ok);
        }
        // `Complex<f64>` is the largest scalar; the storage must not widen past its size.
        assert_eq!(core::mem::size_of::<AlignedScale<Complex<f64>>>(), 16);
    }

    /// `AlignedScale::as_ptr` keeps the alignment of the storage, not of `T`.
    #[test]
    fn aligned_scale_pointer_is_aligned() {
        let scale = AlignedScale::new(Complex::new(1.0f64, 2.0));
        assert_eq!(scale.as_ptr() as usize % 16, 0);
    }
}
