use super::*;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone)]
pub struct Shape<D>(pub D::Shape)
where
    D: DimBaseAPI;

impl<D> Deref for Shape<D>
where
    D: DimBaseAPI,
{
    type Target = D::Shape;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D> DerefMut for Shape<D>
where
    D: DimBaseAPI,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait DimShapeAPI: DimBaseAPI {
    /// Number of dimensions of the shape.
    fn ndim(shape: &Shape<Self>) -> usize;
    /// Total number of elements in tensor.
    fn size(shape: &Shape<Self>) -> usize;
    /// Stride for a f-contiguous tensor using this shape.
    fn stride_f_contig(shape: &Shape<Self>) -> Stride<Self>;
    /// Stride for a c-contiguous tensor using this shape.
    fn stride_c_contig(shape: &Shape<Self>) -> Stride<Self>;
    /// Stride for contiguous tensor using this shape.
    /// Whether c-contiguous or f-contiguous will depends on cargo feature
    /// `c_prefer`.
    fn stride_contig(shape: &Shape<Self>) -> Stride<Self>;
}

impl<D> Shape<D>
where
    D: DimShapeAPI,
{
    pub fn ndim(&self) -> usize {
        <D as DimShapeAPI>::ndim(self)
    }

    pub fn size(&self) -> usize {
        D::size(self)
    }

    pub fn stride_f_contig(&self) -> Stride<D> {
        D::stride_f_contig(self)
    }

    pub fn stride_c_contig(&self) -> Stride<D> {
        D::stride_c_contig(self)
    }

    pub fn stride_contig(&self) -> Stride<D> {
        D::stride_contig(self)
    }
}

impl<const N: usize> DimShapeAPI for Ix<N> {
    fn ndim(shape: &Shape<Ix<N>>) -> usize {
        shape.len()
    }

    fn size(shape: &Shape<Ix<N>>) -> usize {
        shape.iter().product()
    }

    fn stride_f_contig(shape: &Shape<Ix<N>>) -> Stride<Ix<N>> {
        let mut stride = [1; N];
        for i in 1..N {
            stride[i] = stride[i - 1] * shape[i - 1].max(1) as isize;
        }
        Stride(stride)
    }

    fn stride_c_contig(shape: &Shape<Ix<N>>) -> Stride<Ix<N>> {
        let mut stride = [1; N];
        if N == 0 {
            return Stride(stride);
        }
        for i in (0..N - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1].max(1) as isize;
        }
        Stride(stride)
    }

    fn stride_contig(shape: &Shape<Ix<N>>) -> Stride<Ix<N>> {
        match crate::C_PREFER {
            true => Self::stride_c_contig(shape),
            false => Self::stride_f_contig(shape),
        }
    }
}

impl DimShapeAPI for IxD {
    fn ndim(shape: &Shape<IxD>) -> usize {
        shape.len()
    }

    fn size(shape: &Shape<IxD>) -> usize {
        shape.iter().product()
    }

    fn stride_f_contig(shape: &Shape<IxD>) -> Stride<IxD> {
        let mut stride = vec![1; shape.len()];
        for i in 1..shape.len() {
            stride[i] = stride[i - 1] * shape[i - 1] as isize;
        }
        Stride(stride)
    }

    fn stride_c_contig(shape: &Shape<IxD>) -> Stride<IxD> {
        let mut stride = vec![1; shape.len()];
        if shape.is_empty() {
            return Stride(stride);
        }
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1] as isize;
        }
        Stride(stride)
    }

    fn stride_contig(shape: &Shape<IxD>) -> Stride<IxD> {
        match crate::C_PREFER {
            true => Self::stride_c_contig(shape),
            false => Self::stride_f_contig(shape),
        }
    }
}
