use crate::prelude_dev::*;
use core::ops::{Deref, DerefMut};

#[derive(Debug, Clone)]
pub struct Stride<D>(pub D::Stride)
where
    D: DimBaseAPI;

impl<D> Deref for Stride<D>
where
    D: DimBaseAPI,
{
    type Target = D::Stride;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D> DerefMut for Stride<D>
where
    D: DimBaseAPI,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait DimStrideAPI: DimBaseAPI {
    /// Number of dimensions of the shape.
    fn ndim(stride: &Stride<Self>) -> usize;
    /// Check if the strides are f-preferred.
    fn is_f_prefer(stride: &Stride<Self>) -> bool;
    /// Check if the strides are c-preferred.
    fn is_c_prefer(stride: &Stride<Self>) -> bool;
}

impl<D> Stride<D>
where
    D: DimStrideAPI,
{
    pub fn ndim(&self) -> usize {
        <D as DimStrideAPI>::ndim(self)
    }

    pub fn is_f_prefer(&self) -> bool {
        D::is_f_prefer(self)
    }

    pub fn is_c_prefer(&self) -> bool {
        D::is_c_prefer(self)
    }
}

impl<const N: usize> DimStrideAPI for Ix<N> {
    fn ndim(stride: &Stride<Ix<N>>) -> usize {
        stride.len()
    }

    fn is_f_prefer(stride: &Stride<Ix<N>>) -> bool {
        if N == 0 {
            return true;
        }
        if stride.first().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..stride.len() {
            if !((stride[i] >= stride[i - 1]) && (stride[i - 1] >= 0) && (stride[i] > 0)) {
                return false;
            }
        }
        true
    }

    fn is_c_prefer(stride: &Stride<Ix<N>>) -> bool {
        if N == 0 {
            return true;
        }
        if stride.last().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..stride.len() {
            if !((stride[i] <= stride[i - 1]) && (stride[i - 1] >= 0) && (stride[i] > 0)) {
                return false;
            }
        }
        true
    }
}

impl DimStrideAPI for IxD {
    fn ndim(stride: &Stride<IxD>) -> usize {
        stride.len()
    }

    fn is_f_prefer(stride: &Stride<IxD>) -> bool {
        if stride.is_empty() {
            return true;
        }
        if stride.first().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..stride.len() {
            if !((stride[i] > stride[i - 1]) && (stride[i - 1] > 0) && (stride[i] > 0)) {
                return false;
            }
        }
        true
    }

    fn is_c_prefer(stride: &Stride<IxD>) -> bool {
        if stride.is_empty() {
            return true;
        }
        if stride.last().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..stride.len() {
            if !((stride[i] < stride[i - 1]) && (stride[i - 1] > 0) && (stride[i] > 0)) {
                return false;
            }
        }
        true
    }
}
