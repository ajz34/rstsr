use num::{complex::ComplexFloat, Float};

use super::StorageAPI;
use crate::Result;

pub trait StorageCreationAPI: StorageAPI {
    fn zeros_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn ones_impl(device: &Self::Device, len: usize) -> Result<Self>;
    unsafe fn empty_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn full_impl(device: &Self::Device, len: usize, fill: Self::DType) -> Result<Self>;
    fn arange_int_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn outof_cpu_vec(device: &Self::Device, vec: Vec<Self::DType>) -> Result<Self>;
    fn from_cpu_vec(device: &Self::Device, vec: &[Self::DType]) -> Result<Self>;
}

pub trait StorageCreationComplexFloatAPI: StorageCreationAPI
where
    Self::DType: ComplexFloat,
{
    fn linspace_impl(
        device: &Self::Device,
        start: Self::DType,
        end: Self::DType,
        n: usize,
    ) -> Result<Self>;
}

pub trait StorageCreationFloatAPI: StorageCreationAPI
where
    Self::DType: Float,
{
    fn arange_float_impl(
        device: &Self::Device,
        start: Self::DType,
        end: Self::DType,
        step: Self::DType,
    ) -> Result<Self>;
}
