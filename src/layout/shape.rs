use crate::prelude_dev::*;
use core::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Shape<D>(pub D)
where
    D: DimBaseAPI;

impl<D> Deref for Shape<D>
where
    D: DimBaseAPI,
{
    type Target = D;

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
    /// Number of dimensions of the shape.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::prelude_dev::*;
    ///
    /// // fixed dimension
    /// let shape = Shape([2, 3]);
    /// assert_eq!(shape.ndim(), 2);
    ///
    /// // dynamic dimension
    /// let shape = Shape(vec![2, 3]);
    /// assert_eq!(shape.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        <D as DimShapeAPI>::ndim(self)
    }

    /// Total number of elements in tensor.
    ///
    /// # Note
    ///
    /// For 0-dimension tensor, it contains one element.
    /// For multi-dimension tensor with a dimension that have zero length, it
    /// contains zero elements.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::prelude_dev::*;
    ///
    /// let shape = Shape([2, 3]);
    /// assert_eq!(shape.size(), 6);
    ///
    /// let shape = Shape(vec![]);
    /// assert_eq!(shape.size(), 1);
    /// ```
    pub fn size(&self) -> usize {
        D::size(self)
    }

    /// Stride for a f-contiguous tensor using this shape.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::prelude_dev::*;
    ///
    /// let stride = Shape([2, 3, 5]).stride_f_contig();
    /// assert_eq!(stride, Stride([1, 2, 6]));
    /// ```
    pub fn stride_f_contig(&self) -> Stride<D> {
        D::stride_f_contig(self)
    }

    /// Stride for a c-contiguous tensor using this shape.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::prelude_dev::*;
    ///
    /// let stride = Shape([2, 3, 5]).stride_c_contig();
    /// assert_eq!(stride, Stride([15, 5, 1]));
    /// ```
    pub fn stride_c_contig(&self) -> Stride<D> {
        D::stride_c_contig(self)
    }

    /// Stride for contiguous tensor using this shape.
    ///
    /// # Cargo feature dependent
    ///
    /// Whether c-contiguous or f-contiguous will depends on cargo feature
    /// `c_prefer`.
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

impl<D> From<D> for Shape<D>
where
    D: DimBaseAPI,
{
    fn from(value: D) -> Self {
        Shape(value)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ndim() {
        // general test
        let shape = Shape([2, 3]);
        assert_eq!(shape.ndim(), 2);
        let shape = Shape(vec![2, 3]);
        assert_eq!(shape.ndim(), 2);
        // empty dimension test
        let shape = Shape([]);
        assert_eq!(shape.ndim(), 0);
        let shape = Shape(vec![]);
        assert_eq!(shape.ndim(), 0);
    }

    #[test]
    fn test_size() {
        // general test
        let shape = Shape([2, 3]);
        assert_eq!(shape.size(), 6);
        let shape = Shape(vec![]);
        assert_eq!(shape.size(), 1);
        // empty dimension test
        let shape = Shape([]);
        assert_eq!(shape.size(), 1);
        let shape = Shape(vec![]);
        assert_eq!(shape.size(), 1);
        // zero element test
        let shape = Shape([1, 2, 0, 4]);
        assert_eq!(shape.size(), 0);
    }

    #[test]
    fn test_stride_f_contig() {
        // general test
        let stride = Shape([2, 3, 5]).stride_f_contig();
        assert_eq!(stride, Stride([1, 2, 6]));
        // empty dimension test
        let stride = Shape([]).stride_f_contig();
        assert_eq!(stride, Stride([]));
        let stride = Shape(vec![]).stride_f_contig();
        assert_eq!(stride, Stride(vec![]));
        // zero element test
        let stride = Shape([1, 2, 0, 4]).stride_f_contig();
        println!("{stride:?}");
    }

    #[test]
    fn test_stride_c_contig() {
        // general test
        let stride = Shape([2, 3, 5]).stride_c_contig();
        assert_eq!(stride, Stride([15, 5, 1]));
        // empty dimension test
        let stride = Shape([]).stride_c_contig();
        assert_eq!(stride, Stride([]));
        let stride = Shape(vec![]).stride_c_contig();
        assert_eq!(stride, Stride(vec![]));
        // zero element test
        let stride = Shape([1, 2, 0, 4]).stride_c_contig();
        println!("{stride:?}");
    }
}
