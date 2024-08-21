use crate::prelude_dev::*;
use num::{complex::ComplexFloat, Float};

pub trait StorageCreationAPI: StorageBaseAPI {
    fn arange_impl(
        device: &Self::Device,
        start: Self::DType,
        end: Self::DType,
        step: Self::DType,
    ) -> Result<Self>
    where
        Self::DType: Float;
    fn arange_int_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn zeros_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn ones_impl(device: &Self::Device, len: usize) -> Result<Self>;
    /// # Safety
    ///
    /// This function is unsafe because it does not initialize the memory.
    unsafe fn empty_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn full_impl(device: &Self::Device, len: usize, fill: Self::DType) -> Result<Self>;
    fn outof_cpu_vec(device: &Self::Device, vec: Vec<Self::DType>) -> Result<Self>;
    fn from_cpu_vec(device: &Self::Device, vec: &[Self::DType]) -> Result<Self>;
    fn linspace_impl(
        device: &Self::Device,
        start: Self::DType,
        end: Self::DType,
        n: usize,
    ) -> Result<Self>
    where
        Self::DType: ComplexFloat;
}
