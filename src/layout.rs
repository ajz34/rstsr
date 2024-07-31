use crate::{Error, Result};

/* #region Struct Definitions */

pub struct Dim<I>
where
    I: AsRef<[usize]>,
{
    _phantom: std::marker::PhantomData<I>,
}

pub struct Layout<D>
where
    D: Dimension,
{
    shape: D::Shape,
    stride: D::Stride,
    offset: usize,
}

/* #endregion */

/* #region Dimension */

pub trait Dimension {
    type Index: AsRef<[usize]>;
    type Shape: AsRef<[usize]>;
    type Stride: AsRef<[isize]>;
}

impl<const N: usize> Dimension for Dim<[usize; N]> {
    type Index = [usize; N];
    type Shape = [usize; N];
    type Stride = [isize; N];
}

impl Dimension for Dim<Vec<usize>> {
    type Index = Vec<usize>;
    type Shape = Vec<usize>;
    type Stride = Vec<isize>;
}

pub type Ix<const N: usize> = Dim<[usize; N]>;
pub type Ix0 = Ix<0>;
pub type Ix1 = Ix<1>;
pub type Ix2 = Ix<2>;
pub type Ix3 = Ix<3>;
pub type Ix4 = Ix<4>;
pub type Ix5 = Ix<5>;
pub type Ix6 = Ix<6>;
pub type IxDyn = Dim<Vec<usize>>;

/* #endregion */

/* #region Shape */

pub trait Shape: AsRef<[usize]> {
    /// Number of dimensions of the shape.
    fn rank(&self) -> usize;
    /// Total number of elements in tensor.
    fn size(&self) -> usize;
    /// Stride for a f-contiguous tensor using this shape.
    fn stride_f_contig(&self) -> impl AsRef<[isize]>;
    /// Stride for a c-contiguous tensor using this shape.
    fn stride_c_contig(&self) -> impl AsRef<[isize]>;
}

impl<const N: usize> Shape for [usize; N] {
    fn rank(&self) -> usize {
        self.len()
    }

    fn size(&self) -> usize {
        self.iter().product()
    }

    fn stride_f_contig(&self) -> [isize; N] {
        let mut stride = [1; N];
        for i in 1..N {
            stride[i] = stride[i - 1] * self[i - 1] as isize;
        }
        stride
    }

    fn stride_c_contig(&self) -> [isize; N] {
        let mut stride = [1; N];
        if N == 0 {
            return [1; N];
        }
        for i in (0..N - 1).rev() {
            stride[i] = stride[i + 1] * self[i + 1] as isize;
        }
        stride
    }
}

impl Shape for Vec<usize> {
    fn rank(&self) -> usize {
        self.len()
    }

    fn size(&self) -> usize {
        self.iter().product()
    }

    fn stride_f_contig(&self) -> Vec<isize> {
        let mut stride = vec![1; self.len()];
        for i in 1..self.len() {
            stride[i] = stride[i - 1] * self[i - 1] as isize;
        }
        stride
    }

    fn stride_c_contig(&self) -> Vec<isize> {
        let mut stride = vec![1; self.len()];
        if self.is_empty() {
            return vec![1; self.len()];
        }
        for i in (0..self.len() - 1).rev() {
            stride[i] = stride[i + 1] * self[i + 1] as isize;
        }
        stride
    }
}

/* #endregion */

/* #region Strides */

pub trait Stride: AsRef<[isize]> {
    /// Number of dimensions of the shape.
    fn rank(&self) -> usize;
    /// Check if the strides are f-preferred.
    fn is_f_prefer(&self) -> bool;
    /// Check if the strides are c-preferred.
    fn is_c_prefer(&self) -> bool;
}

impl<const N: usize> Stride for [isize; N] {
    fn rank(&self) -> usize {
        self.len()
    }

    fn is_f_prefer(&self) -> bool {
        if N == 0 {
            return true;
        }
        if self.first().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..self.len() {
            if !((self[i] > self[i - 1]) && (self[i - 1] > 0) && (self[i] > 0)) {
                return false;
            }
        }
        true
    }

    fn is_c_prefer(&self) -> bool {
        if N == 0 {
            return true;
        }
        if self.last().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..self.len() {
            if !((self[i] < self[i - 1]) && (self[i - 1] > 0) && (self[i] > 0)) {
                return false;
            }
        }
        true
    }
}

impl Stride for Vec<isize> {
    fn rank(&self) -> usize {
        self.len()
    }

    fn is_f_prefer(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        if self.first().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..self.len() {
            if !((self[i] > self[i - 1]) && (self[i - 1] > 0) && (self[i] > 0)) {
                return false;
            }
        }
        true
    }

    fn is_c_prefer(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        if self.last().is_some_and(|&a| a != 1) {
            return false;
        }
        for i in 1..self.len() {
            if !((self[i] < self[i - 1]) && (self[i - 1] > 0) && (self[i] > 0)) {
                return false;
            }
        }
        true
    }
}

/* #endregion Strides */

/* #region Layout */

pub trait LayoutTrait {
    type Shape: AsRef<[usize]>;
    type Stride: AsRef<[isize]>;

    /// Shape of tensor. Getter function.
    fn shape(&self) -> impl AsRef<[usize]>;
    /// Stride of tensor. Getter function.
    fn stride(&self) -> impl AsRef<[isize]>;
    /// Starting offset of tensor. Getter function.
    fn offset(&self) -> usize;
    /// Number of dimensions of tensor.
    fn rank(&self) -> usize;
    /// Total number of elements in tensor.
    fn size(&self) -> usize;
    /// Whether this tensor is f-preferred.
    fn is_f_prefer(&self) -> bool;
    /// Whether this tensor is c-preferred.
    fn is_c_prefer(&self) -> bool;
    /// Whether this tensor is f-contiguous.
    fn is_f_contig(&self) -> bool;
    /// Whether this tensor is c-contiguous.
    fn is_c_contig(&self) -> bool;

    /// Generate new layout by providing everything.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Shape and stride length mismatch
    fn new(shape: Self::Shape, stride: Self::Stride, offset: usize) -> Self;
    /// Generate new layout by providing shape and offset; stride fits into
    /// c-contiguous.
    fn new_c_contig(shape: Self::Shape, offset: usize) -> Self;
    /// Generate new layout by providing shape and offset; stride fits into
    /// f-contiguous.
    fn new_f_contig(shape: Self::Shape, offset: usize) -> Self;
    /// Index of tensor by list of indexes to dimensions.
    fn try_index(&self, index: Self::Shape) -> Result<usize>;
    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Negative index
    /// - Index greater than shape
    fn index(&self, index: Self::Shape) -> usize;
    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    unsafe fn index_uncheck(&self, index: Self::Shape) -> usize;

    /// Number of dimensions of tensor. Alias (numpy convention) to
    /// [LayoutTrait::rank].
    fn ndim(&self) -> usize {
        self.rank()
    }
}

impl<const N: usize> LayoutTrait for Layout<Ix<N>> {
    type Shape = [usize; N];
    type Stride = [isize; N];

    fn shape(&self) -> [usize; N] {
        self.shape
    }

    fn stride(&self) -> [isize; N] {
        self.stride
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn rank(&self) -> usize {
        self.shape.rank()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn is_f_prefer(&self) -> bool {
        self.stride.is_f_prefer()
    }

    fn is_c_prefer(&self) -> bool {
        let mut acc = 1;
        for (&s, &d) in self.stride.iter().zip(self.shape.iter()).rev() {
            if s != acc {
                return false;
            }
            acc *= d as isize;
        }
        return true;
    }

    fn is_f_contig(&self) -> bool {
        let mut acc = 1;
        for (&s, &d) in self.stride.iter().zip(self.shape.iter()) {
            if s != acc {
                return false;
            }
            acc *= d as isize;
        }
        return true;
    }

    fn is_c_contig(&self) -> bool {
        self.stride.is_c_prefer() && self.offset == 0
    }

    fn new(shape: [usize; N], stride: [isize; N], offset: usize) -> Self {
        Layout { shape, stride, offset }
    }

    fn new_c_contig(shape: [usize; N], offset: usize) -> Self {
        let stride = shape.stride_c_contig();
        Layout { shape, stride, offset }
    }

    fn new_f_contig(shape: [usize; N], offset: usize) -> Self {
        let stride = shape.stride_f_contig();
        Layout { shape, stride, offset }
    }

    fn try_index(&self, index: [usize; N]) -> Result<usize> {
        let mut pos = self.offset as isize;
        for i in 0..N {
            if index[i] >= self.shape[i] {
                return Err(Error::IndexOutOfBound {
                    index: index.into(),
                    shape: self.shape.into(),
                });
            }
            pos += self.stride[i] * index[i] as isize;
        }
        if pos < 0 {
            return Err(Error::IndexOutOfBound { index: index.into(), shape: self.shape.into() });
        }
        return Ok(pos as usize);
    }

    fn index(&self, index: [usize; N]) -> usize {
        self.try_index(index).unwrap()
    }

    unsafe fn index_uncheck(&self, index: Self::Shape) -> usize {
        let mut pos = self.offset as isize;
        for i in 0..N {
            pos += self.stride[i] * index[i] as isize;
        }
        return pos as usize;
    }
}

impl LayoutTrait for Layout<IxDyn> {
    type Shape = Vec<usize>;
    type Stride = Vec<isize>;

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn stride(&self) -> Vec<isize> {
        self.stride.clone()
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn rank(&self) -> usize {
        self.shape.rank()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn is_f_prefer(&self) -> bool {
        self.stride.is_f_prefer()
    }

    fn is_c_prefer(&self) -> bool {
        let mut acc = 1;
        for (&s, &d) in self.stride.iter().zip(self.shape.iter()).rev() {
            if s != acc {
                return false;
            }
            acc *= d as isize;
        }
        return true;
    }

    fn is_f_contig(&self) -> bool {
        let mut acc = 1;
        for (&s, &d) in self.stride.iter().zip(self.shape.iter()) {
            if s != acc {
                return false;
            }
            acc *= d as isize;
        }
        return true;
    }

    fn is_c_contig(&self) -> bool {
        self.stride.is_c_prefer() && self.offset == 0
    }

    fn new(shape: Vec<usize>, stride: Vec<isize>, offset: usize) -> Self {
        if shape.len() != stride.len() {
            panic!("Shape and stride length mismatch, shape {:?}, stride {:?}", shape, stride);
        }
        Layout { shape, stride, offset }
    }

    fn new_c_contig(shape: Vec<usize>, offset: usize) -> Self {
        let stride = shape.stride_c_contig();
        Layout { shape, stride, offset }
    }

    fn new_f_contig(shape: Vec<usize>, offset: usize) -> Self {
        let stride = shape.stride_f_contig();
        Layout { shape, stride, offset }
    }

    fn try_index(&self, index: Vec<usize>) -> Result<usize> {
        let mut pos = self.offset as isize;
        for i in 0..self.shape.rank() {
            if index[i] >= self.shape[i] {
                return Err(Error::IndexOutOfBound { index, shape: self.shape.clone() });
            }
            pos += self.stride[i] * index[i] as isize;
        }
        if pos < 0 {
            return Err(Error::IndexOutOfBound { index, shape: self.shape.clone() });
        }
        return Ok(pos as usize);
    }

    fn index(&self, index: Self::Shape) -> usize {
        self.try_index(index).unwrap()
    }

    unsafe fn index_uncheck(&self, index: Vec<usize>) -> usize {
        let mut pos = self.offset as isize;
        for i in 0..self.shape.rank() {
            pos += self.stride[i] * index[i] as isize;
        }
        return pos as usize;
    }
}

/* #endregion Layout */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_layout() {
        let layout = Layout::<Ix<3>> { shape: [1, 2, 3], stride: [6, 3, 1], offset: 0 };
        assert_eq!(layout.shape, [1, 2, 3]);
        assert_eq!(layout.stride, [6, 3, 1]);
        assert_eq!(layout.offset, 0);
    }

    #[test]
    fn test_shape() {
        let shape: [usize; 3] = [4, 2, 3];
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.size(), 24);
        assert_eq!(shape.stride_f_contig(), [1, 4, 8]);
        assert_eq!(shape.stride_c_contig(), [6, 3, 1]);
    }

    #[test]
    fn test_shape_zero_dim() {
        let shape: [usize; 0] = [];
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.size(), 1);
        assert_eq!(shape.stride_f_contig(), []);
        assert_eq!(shape.stride_c_contig(), []);
    }

    #[test]
    fn test_shape_dyn() {
        let shape: Vec<usize> = vec![4, 2, 3];
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.size(), 24);
        assert_eq!(shape.stride_f_contig(), vec![1, 4, 8]);
        assert_eq!(shape.stride_c_contig(), vec![6, 3, 1]);
    }

    #[test]
    fn test_shape_dyn_zero_dim() {
        let shape: Vec<usize> = vec![];
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.size(), 1);
        assert_eq!(shape.stride_f_contig(), vec![]);
        assert_eq!(shape.stride_c_contig(), vec![]);
    }

    #[test]
    fn test_stride() {
        let stride: [isize; 3] = [6, 3, 1];
        assert_eq!(stride.rank(), 3);
        assert_eq!(stride.is_f_prefer(), false);
        assert_eq!(stride.is_c_prefer(), true);
        let stride: [isize; 3] = [1, 4, 8];
        assert_eq!(stride.rank(), 3);
        assert_eq!(stride.is_f_prefer(), true);
        assert_eq!(stride.is_c_prefer(), false);
    }

    #[test]
    fn test_stride_zero_dim() {
        let stride: [isize; 0] = [];
        assert_eq!(stride.rank(), 0);
        assert_eq!(stride.is_f_prefer(), true);
        assert_eq!(stride.is_c_prefer(), true);
    }

    #[test]
    fn test_stride_dyn() {
        let stride: Vec<isize> = vec![6, 3, 1];
        assert_eq!(stride.rank(), 3);
        assert_eq!(stride.is_f_prefer(), false);
        assert_eq!(stride.is_c_prefer(), true);
        let stride: Vec<isize> = vec![1, 4, 8];
        assert_eq!(stride.rank(), 3);
        assert_eq!(stride.is_f_prefer(), true);
        assert_eq!(stride.is_c_prefer(), false);
    }

    #[test]
    fn test_stride_dyn_zero_dim() {
        let stride: Vec<isize> = vec![];
        assert_eq!(stride.rank(), 0);
        assert_eq!(stride.is_f_prefer(), true);
        assert_eq!(stride.is_c_prefer(), true);
    }

    #[test]
    fn test_layout_trait() {
        let layout = Layout::<Ix<3>> { shape: [1, 2, 3], stride: [6, 3, 1], offset: 0 };
        assert_eq!(layout.shape(), [1, 2, 3]);
        assert_eq!(layout.stride(), [6, 3, 1]);
        assert_eq!(layout.offset(), 0);
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.size(), 6);
        assert_eq!(layout.is_f_prefer(), false);
        assert_eq!(layout.is_c_prefer(), true);
        assert_eq!(layout.is_f_contig(), false);
        assert_eq!(layout.is_c_contig(), true);
        assert_eq!(layout.ndim(), 3);
        assert_eq!(layout.index([0, 1, 2]), 5);
        assert_eq!(unsafe { layout.index_uncheck([0, 1, 2]) }, 5);
    }

    #[test]
    fn test_layout_trait_dyn() {
        let layout = Layout::<IxDyn> { shape: vec![1, 2, 3], stride: vec![6, 3, 1], offset: 0 };
        assert_eq!(layout.shape(), vec![1, 2, 3]);
        assert_eq!(layout.stride(), vec![6, 3, 1]);
        assert_eq!(layout.offset(), 0);
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.size(), 6);
        assert_eq!(layout.is_f_prefer(), false);
        assert_eq!(layout.is_c_prefer(), true);
        assert_eq!(layout.is_f_contig(), false);
        assert_eq!(layout.is_c_contig(), true);
        assert_eq!(layout.ndim(), 3);
        assert_eq!(layout.index(vec![0, 1, 2]), 5);
        assert_eq!(unsafe { layout.index_uncheck(vec![0, 1, 2]) }, 5);
    }
}
