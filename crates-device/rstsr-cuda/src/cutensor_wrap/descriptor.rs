//! Tensor descriptors: extent, stride, dtype and pointer alignment of one operand.

use crate::cutensor_wrap::dtype::CutensorDType;
use crate::cutensor_wrap::error::{self, Stage};
use crate::cutensor_wrap::handle::CutensorHandle;
use crate::cutensor_wrap::operation::OperandPlan;
use crate::prelude_dev::*;
use cudarc::cutensor::sys;

/// The alignment of `cudaMalloc`, which cuTENSOR's own examples declare.
const MAX_ALIGNMENT: usize = 256;

/// The byte alignment of a device address: the largest power of two dividing it, capped at
/// [`MAX_ALIGNMENT`] (`0` for the null address). This is the honest `alignmentRequirement` —
/// cuTENSOR does not check it against the pointer (a 256-byte declaration on an 8-byte-aligned
/// pointer is accepted), so declaring less than the truth only gives the planner freedom.
pub(crate) fn ptr_alignment(addr: usize) -> u32 {
    if addr == 0 {
        return 0;
    }
    (addr & addr.wrapping_neg()).min(MAX_ALIGNMENT) as u32
}

/// Validate the layout facts of one operand and normalize them to cuTENSOR's `int64` arrays.
///
/// Refused here: a negative stride and a zero extent (cuTENSOR rejects both with
/// `INVALID_VALUE`; the wrapper refuses with a message instead). A stride of `0` with extent
/// `> 1` **passes**: inputs broadcast correctly on 2.6.0+ and decline with `NOT_SUPPORTED` on
/// older ones — the runtime status is the verdict; only *outputs* with stride `0` are refused,
/// at operation creation, because every version either rejects them or writes garbage.
fn validate_layout(extent: &[usize], stride: &[isize]) -> Result<(Vec<i64>, Vec<i64>)> {
    if extent.len() != stride.len() {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: tensor descriptor: extent length {} does not match stride length {}",
            extent.len(),
            stride.len()
        ));
    }
    if let Some(&s) = stride.iter().find(|&&s| s < 0) {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: tensor descriptor: negative stride {s} (flip the operand into a \
             materialized copy first; cuTENSOR has no negative strides)"
        ));
    }
    if let Some(&e) = extent.iter().find(|&&e| e == 0) {
        return Err(rstsr_error!(
            InvalidValue,
            "cutensor: tensor descriptor: zero extent {e} (empty operands are refused; \
             cuTENSOR rejects them with a bare INVALID_VALUE)"
        ));
    }
    Ok((extent.iter().map(|&e| e as i64).collect(), stride.iter().map(|&s| s as i64).collect()))
}

/// The RAII descriptor of one operand: extent, stride, dtype and the declared base-pointer
/// alignment. It holds no pointer — descriptors describe *layouts*, and the same descriptor
/// serves every buffer of that layout (plan reuse).
///
/// The declared alignment (cuTENSOR's `alignmentRequirement`) is the byte alignment of the base
/// pointer the descriptor will be used with (buffer pointer plus the layout offset). By default
/// [`TensorDescriptor::new`] declares the pointer's own alignment (largest power of two dividing
/// the address, capped at 256); [`TensorDescriptor::with_alignment`] declares a floor of the
/// caller's choice — what makes one plan serve every buffer of a shape regardless of offset
/// views. Execute-time re-validation checks `actual >= declared`.
pub struct TensorDescriptor {
    ptr: sys::cutensorTensorDescriptor_t,
    pub(crate) extent: Vec<i64>,
    pub(crate) stride: Vec<i64>,
    dtype: sys::cudaDataType_t,
    alignment: u32,
}

impl TensorDescriptor {
    /// Describe `view` with its base pointer's own alignment (capped at 256).
    pub fn new<T: CutensorDType>(handle: &CutensorHandle, view: &TensorView<'_, T, DeviceCuda, IxD>) -> Result<Self> {
        Self::build(handle, view, None)
    }

    /// Describe `view` with a declared alignment floor (`power of two >= element size`) instead
    /// of the pointer's own alignment. Use this when one plan should serve every buffer of a
    /// shape: declaring `size_of::<T>()` is valid for any buffer and never blocks reuse.
    pub fn with_alignment<T: CutensorDType>(
        handle: &CutensorHandle,
        view: &TensorView<'_, T, DeviceCuda, IxD>,
        alignment: u32,
    ) -> Result<Self> {
        Self::build(handle, view, Some(alignment))
    }

    fn build<T: CutensorDType>(
        handle: &CutensorHandle,
        view: &TensorView<'_, T, DeviceCuda, IxD>,
        alignment: Option<u32>,
    ) -> Result<Self> {
        handle.bind()?;
        let (extent, stride) = validate_layout(view.shape().as_ref(), view.stride().as_ref())?;
        let esz = core::mem::size_of::<T>();
        // The base address of the *view* (buffer pointer plus the layout offset); a mere address
        // query, no ordering.
        let addr = view.raw().device_ptr() as usize + view.offset() * esz;
        let alignment = match alignment {
            Some(a) => {
                if !a.is_power_of_two() || (a as usize) < esz {
                    return Err(rstsr_error!(
                        InvalidValue,
                        "cutensor: tensor descriptor: declared alignment {a} must be a power of \
                         two and at least the element size {esz}"
                    ));
                }
                a
            },
            None => {
                let computed = ptr_alignment(addr) as usize;
                if computed < esz {
                    return Err(rstsr_error!(
                        InvalidValue,
                        "cutensor: tensor descriptor: the operand pointer's alignment \
                         ({computed} bytes) is below its element size ({esz} bytes)"
                    ));
                }
                computed as u32
            },
        };
        let mut ptr = core::mem::MaybeUninit::<sys::cutensorTensorDescriptor_t>::uninit();
        // SAFETY: `ptr` is a valid output slot; the arrays have `extent.len()` entries, and rank
        // 0 passes null (cuTENSOR accepts null and dummy arrays alike at `numModes` 0).
        let status = unsafe {
            sys::cutensorCreateTensorDescriptor(
                handle.ptr(),
                ptr.as_mut_ptr(),
                extent.len() as u32,
                if extent.is_empty() { core::ptr::null() } else { extent.as_ptr() },
                if stride.is_empty() { core::ptr::null() } else { stride.as_ptr() },
                T::DATA_TYPE,
                alignment,
            )
        };
        error::check(Stage::TensorDescriptor, status, "")?;
        // SAFETY: the descriptor was written on success.
        let ptr = unsafe { ptr.assume_init() };
        Ok(Self { ptr, extent, stride, dtype: T::DATA_TYPE, alignment })
    }

    /// The planned facts of one operand.
    pub(crate) fn plan(&self) -> OperandPlan {
        OperandPlan {
            extent: self.extent.clone(),
            stride: self.stride.clone(),
            dtype: self.dtype,
            alignment: self.alignment,
        }
    }

    pub(crate) fn ptr(&self) -> sys::cutensorTensorDescriptor_t {
        self.ptr
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        // SAFETY: created by `cutensorCreateTensorDescriptor`, destroyed exactly once.
        let status = unsafe { sys::cutensorDestroyTensorDescriptor(self.ptr) };
        debug_assert_eq!(status, sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS);
    }
}

impl core::fmt::Debug for TensorDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorDescriptor")
            .field("extent", &self.extent)
            .field("stride", &self.stride)
            .field("dtype", &self.dtype)
            .field("alignment", &self.alignment)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Powers of two dividing the address, capped at 256; zero stays zero.
    #[test]
    fn pointer_alignment_table() {
        assert_eq!(ptr_alignment(0), 0);
        assert_eq!(ptr_alignment(1), 1);
        assert_eq!(ptr_alignment(2), 2);
        assert_eq!(ptr_alignment(24), 8);
        assert_eq!(ptr_alignment(256), 256);
        assert_eq!(ptr_alignment(4096), 256, "capped");
        assert_eq!(ptr_alignment(96), 32);
    }

    /// Negative strides and zero extents are refused with a message; a stride-0 *input* passes
    /// (the runtime status is the verdict).
    #[test]
    fn layout_refusals() {
        assert!(validate_layout(&[4, 3], &[-4, 1]).is_err());
        assert!(validate_layout(&[4, 0], &[3, 1]).is_err());
        assert!(validate_layout(&[4], &[1, 2]).is_err());
        let (extent, stride) = validate_layout(&[4, 3], &[0, 1]).expect("stride-0 input passes");
        assert_eq!(extent, vec![4, 3]);
        assert_eq!(stride, vec![0, 1]);
        let (extent, stride) = validate_layout(&[], &[]).expect("rank 0 passes");
        assert!(extent.is_empty() && stride.is_empty());
    }
}
