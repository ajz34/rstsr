use super::StorageAPI;
use crate::Result;

pub trait StorageCreationAPI: StorageAPI {
    fn zeros_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn ones_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn arange_int_impl(device: &Self::Device, len: usize) -> Result<Self>;
    unsafe fn empty_impl(device: &Self::Device, len: usize) -> Result<Self>;
    fn outof_cpu_vec(device: &Self::Device, vec: Vec<Self::DType>) -> Result<Self>;
    fn from_cpu_vec(device: &Self::Device, vec: &Vec<Self::DType>) -> Result<Self>;
}
