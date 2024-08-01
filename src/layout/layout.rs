use std::ops::{Deref, DerefMut};

use crate::{Error, Result};

/* #region Dimension Definition and alias */

pub type Ix<const N: usize> = [usize; N];
pub type Ix0 = Ix<0>;
pub type Ix1 = Ix<1>;
pub type Ix2 = Ix<2>;
pub type Ix3 = Ix<3>;
pub type Ix4 = Ix<4>;
pub type Ix5 = Ix<5>;
pub type Ix6 = Ix<6>;
pub type Ix7 = Ix<7>;
pub type Ix8 = Ix<8>;
pub type Ix9 = Ix<9>;
pub type IxD = Vec<usize>;
pub type IxDyn = IxD;

pub trait DimAPI {
    type Shape: AsMut<[usize]> + AsRef<[usize]> + core::fmt::Debug + Clone;
    type Stride: AsMut<[isize]> + AsRef<[isize]> + core::fmt::Debug + Clone;
}

impl<const N: usize> DimAPI for Ix<N> {
    type Shape = [usize; N];
    type Stride = [isize; N];
}

impl DimAPI for IxD {
    type Shape = Vec<usize>;
    type Stride = Vec<isize>;
}

/* #endregion */

/* #region Strides */

#[derive(Debug, Clone)]
pub struct Stride<D>(pub(crate) D::Stride)
where
    D: DimAPI;

impl<D> Deref for Stride<D>
where
    D: DimAPI,
{
    type Target = D::Stride;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D> DerefMut for Stride<D>
where
    D: DimAPI,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait StrideAPI {
    type Dim: DimAPI;
    /// Number of dimensions of the shape.
    fn rank(&self) -> usize;
    /// Check if the strides are f-preferred.
    fn is_f_prefer(&self) -> bool;
    /// Check if the strides are c-preferred.
    fn is_c_prefer(&self) -> bool;
}

impl<const N: usize> StrideAPI for Stride<Ix<N>> {
    type Dim = Ix<N>;

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

impl StrideAPI for Stride<IxD> {
    type Dim = IxD;

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

/* #region Shape */

#[derive(Debug, Clone)]
pub struct Shape<D>(pub(crate) D::Shape)
where
    D: DimAPI;

impl<D> Deref for Shape<D>
where
    D: DimAPI,
    Self: ShapeAPI,
{
    type Target = D::Shape;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D> DerefMut for Shape<D>
where
    D: DimAPI,
    Self: ShapeAPI,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait ShapeAPI {
    type Dim: DimAPI;
    /// Number of dimensions of the shape.
    fn rank(&self) -> usize;
    /// Total number of elements in tensor.
    fn size(&self) -> usize;
    /// Stride for a f-contiguous tensor using this shape.
    fn stride_f_contig(&self) -> Stride<Self::Dim>;
    /// Stride for a c-contiguous tensor using this shape.
    fn stride_c_contig(&self) -> Stride<Self::Dim>;
    /// Stride for contiguous tensor using this shape.
    /// Whether c-contiguous or f-contiguous will depends on cargo feature
    /// `c_prefer`.
    fn stride_config(&self) -> Stride<Self::Dim>;
}

impl<const N: usize> ShapeAPI for Shape<Ix<N>> {
    type Dim = Ix<N>;

    fn rank(&self) -> usize {
        self.len()
    }

    fn size(&self) -> usize {
        self.iter().product()
    }

    fn stride_f_contig(&self) -> Stride<Ix<N>> {
        let mut stride = [1; N];
        for i in 1..N {
            stride[i] = stride[i - 1] * self[i - 1] as isize;
        }
        Stride(stride)
    }

    fn stride_c_contig(&self) -> Stride<Ix<N>> {
        let mut stride = [1; N];
        if N == 0 {
            return Stride(stride);
        }
        for i in (0..N - 1).rev() {
            stride[i] = stride[i + 1] * self[i + 1] as isize;
        }
        Stride(stride)
    }

    fn stride_config(&self) -> Stride<Ix<N>> {
        match crate::C_PREFER {
            true => self.stride_c_contig(),
            false => self.stride_f_contig(),
        }
    }
}

impl ShapeAPI for Shape<IxD> {
    type Dim = IxD;

    fn rank(&self) -> usize {
        self.len()
    }

    fn size(&self) -> usize {
        self.iter().product()
    }

    fn stride_f_contig(&self) -> Stride<IxD> {
        let mut stride = vec![1; self.len()];
        for i in 1..self.len() {
            stride[i] = stride[i - 1] * self[i - 1] as isize;
        }
        Stride(stride)
    }

    fn stride_c_contig(&self) -> Stride<IxD> {
        let mut stride = vec![1; self.len()];
        if self.is_empty() {
            return Stride(stride);
        }
        for i in (0..self.len() - 1).rev() {
            stride[i] = stride[i + 1] * self[i + 1] as isize;
        }
        Stride(stride)
    }

    fn stride_config(&self) -> Stride<IxD> {
        match crate::C_PREFER {
            true => self.stride_c_contig(),
            false => self.stride_f_contig(),
        }
    }
}

/* #endregion */

/* #region Struct Definitions */

#[derive(Debug, Clone)]
pub struct Layout<D>
where
    D: DimAPI,
{
    pub(crate) shape: Shape<D>,
    pub(crate) stride: Stride<D>,
    pub(crate) offset: usize,
}

/* #endregion */

/* #region Layout */

pub trait LayoutAPI {
    type Dim: DimAPI;

    /// Shape of tensor. Getter function.
    fn shape(&self) -> Shape<Self::Dim>;
    fn shape_ref(&self) -> &Shape<Self::Dim>;

    /// Stride of tensor. Getter function.
    fn stride(&self) -> Stride<Self::Dim>;
    fn stride_ref(&self) -> &Stride<Self::Dim>;

    /// Starting offset of tensor. Getter function.
    fn offset(&self) -> usize;

    /// Number of dimensions of tensor.
    fn ndim(&self) -> usize;

    /// Total number of elements in tensor.
    fn size(&self) -> usize;

    /// Whether this tensor is f-preferred.
    fn is_f_prefer(&self) -> bool;

    /// Whether this tensor is c-preferred.
    fn is_c_prefer(&self) -> bool;

    /// Whether this tensor is f-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus f-contiguous.
    fn is_f_contig(&self) -> bool;

    /// Whether this tensor is c-contiguous.
    fn is_c_contig(&self) -> bool;

    /// Generate new layout by providing everything.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Shape and stride length mismatch
    fn new(shape: Shape<Self::Dim>, stride: Stride<Self::Dim>, offset: usize) -> Self;

    /// Index of tensor by list of indexes to dimensions.
    fn try_index(&self, index: Shape<Self::Dim>) -> Result<usize>;

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Negative index
    /// - Index greater than shape
    fn index(&self, index: Shape<Self::Dim>) -> usize;

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    unsafe fn index_uncheck(&self, index: Shape<Self::Dim>) -> usize;
}

impl<D> LayoutAPI for Layout<D>
where
    D: DimAPI,
    Shape<D>: ShapeAPI,
    Stride<D>: StrideAPI,
{
    type Dim = D;

    fn shape(&self) -> Shape<D> {
        Shape(self.shape.clone())
    }

    fn shape_ref(&self) -> &Shape<D> {
        &self.shape
    }

    fn stride(&self) -> Stride<D> {
        Stride(self.stride.clone())
    }

    fn stride_ref(&self) -> &Stride<D> {
        &self.stride
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn ndim(&self) -> usize {
        self.shape_ref().rank()
    }

    fn size(&self) -> usize {
        self.shape_ref().size()
    }

    fn is_f_prefer(&self) -> bool {
        self.stride_ref().is_f_prefer()
    }

    fn is_c_prefer(&self) -> bool {
        self.stride_ref().is_c_prefer()
    }

    fn is_f_contig(&self) -> bool {
        let mut acc = 1;
        let mut contig = true;
        for (&s, &d) in self.stride_ref().as_ref().iter().zip(self.shape_ref().as_ref().iter()) {
            if d == 1 {
                continue;
            } else if d == 0 {
                return true;
            } else if s != acc {
                contig = false;
            }
            acc *= d as isize;
        }
        return contig;
    }

    fn is_c_contig(&self) -> bool {
        let mut acc = 1;
        let mut contig = true;
        for (&s, &d) in
            self.stride_ref().as_ref().iter().zip(self.shape_ref().as_ref().iter()).rev()
        {
            if d == 1 {
                continue;
            } else if d == 0 {
                return true;
            } else if s != acc {
                contig = false;
            }
            acc *= d as isize;
        }
        return contig;
    }

    fn new(shape: Shape<D>, stride: Stride<D>, offset: usize) -> Self {
        Layout { shape, stride, offset }
    }

    fn try_index(&self, index: Shape<Self::Dim>) -> Result<usize> {
        let mut pos = self.offset() as isize;
        for i in 0..self.ndim() {
            if index.as_ref()[i] >= self.shape_ref().as_ref()[i] {
                return Err(Error::IndexOutOfBound {
                    index: index.as_ref()[i] as isize,
                    shape: self.shape().as_ref()[i] as isize,
                });
            }
            pos += self.stride_ref().as_ref()[i] * index.as_ref()[i] as isize;
        }
        if pos < 0 {
            return Err(Error::IndexOutOfBound { index: pos, shape: 0 });
        }
        return Ok(pos as usize);
    }

    fn index(&self, index: Shape<Self::Dim>) -> usize {
        self.try_index(index).unwrap()
    }

    unsafe fn index_uncheck(&self, index: Shape<Self::Dim>) -> usize {
        let mut pos = self.offset() as isize;
        for i in 0..self.ndim() {
            pos += self.stride_ref().as_ref()[i] * index.as_ref()[i] as isize;
        }
        return pos as usize;
    }
}

pub trait LayoutContigAPI {
    type Dim: DimAPI;

    /// Generate new layout by providing shape and offset; stride fits into
    /// c-contiguous.
    fn new_c_contig(shape: Shape<Self::Dim>, offset: usize) -> Self;

    /// Generate new layout by providing shape and offset; stride fits into
    /// f-contiguous.
    fn new_f_contig(shape: Shape<Self::Dim>, offset: usize) -> Self;

    /// Generate new layout by providing shape and offset; Whether c-contiguous
    /// or f-contiguous depends on cargo feature `c_prefer`.
    fn new_contig(shape: Shape<Self::Dim>, offset: usize) -> Self;
}

impl<const N: usize> LayoutContigAPI for Layout<Ix<N>> {
    type Dim = Ix<N>;

    fn new_c_contig(shape: Shape<Ix<N>>, offset: usize) -> Self {
        let stride = shape.stride_c_contig();
        Layout { shape, stride, offset }
    }

    fn new_f_contig(shape: Shape<Ix<N>>, offset: usize) -> Self {
        let stride = shape.stride_f_contig();
        Layout { shape, stride, offset }
    }

    fn new_contig(shape: Shape<Self::Dim>, offset: usize) -> Self {
        match crate::C_PREFER {
            true => Self::new_c_contig(shape, offset),
            false => Self::new_f_contig(shape, offset),
        }
    }
}

impl LayoutContigAPI for Layout<IxD> {
    type Dim = IxD;

    fn new_c_contig(shape: Shape<IxD>, offset: usize) -> Self {
        let stride = shape.stride_c_contig();
        Layout { shape, stride, offset }
    }

    fn new_f_contig(shape: Shape<IxD>, offset: usize) -> Self {
        let stride = shape.stride_f_contig();
        Layout { shape, stride, offset }
    }

    fn new_contig(shape: Shape<Self::Dim>, offset: usize) -> Self {
        match crate::C_PREFER {
            true => Self::new_c_contig(shape, offset),
            false => Self::new_f_contig(shape, offset),
        }
    }
}

/* #endregion Layout */

/* #region Dimension Conversion */

impl<const N: usize> From<Layout<Ix<N>>> for Layout<IxD> {
    fn from(layout: Layout<Ix<N>>) -> Self {
        let Layout { shape: Shape(shape), stride: Stride(stride), offset } = layout;
        Layout { shape: Shape(shape.to_vec()), stride: Stride(stride.to_vec()), offset }
    }
}

impl<const N: usize> TryFrom<Layout<IxD>> for Layout<Ix<N>> {
    type Error = Error;

    fn try_from(layout: Layout<IxD>) -> Result<Self> {
        let Layout { shape: Shape(shape), stride: Stride(stride), offset } = layout;
        Ok(Layout {
            shape: Shape(
                shape
                    .try_into()
                    .map_err(|_| Error::Msg(format!("Cannot convert IxD to Ix< {:} >", N)))?,
            ),
            stride: Stride(
                stride
                    .try_into()
                    .map_err(|_| Error::Msg(format!("Cannot convert IxD to Ix< {:} >", N)))?,
            ),
            offset,
        })
    }
}

/* #endregion */

/* #region Shape to Layout */

pub trait IxToLayoutAPI {
    type Layout: LayoutContigAPI;

    fn new_c_contig(&self, offset: usize) -> Self::Layout;
    fn new_f_contig(&self, offset: usize) -> Self::Layout;
    fn new_contig(&self, offset: usize) -> Self::Layout;
}

impl<const N: usize> IxToLayoutAPI for Ix<N> {
    type Layout = Layout<Ix<N>>;

    fn new_c_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_c_contig(Shape(self.clone()), offset)
    }
    fn new_f_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_f_contig(Shape(self.clone()), offset)
    }
    fn new_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_contig(Shape(self.clone()), offset)
    }
}

impl IxToLayoutAPI for IxD {
    type Layout = Layout<IxD>;

    fn new_c_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_c_contig(Shape(self.clone()), offset)
    }
    fn new_f_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_f_contig(Shape(self.clone()), offset)
    }
    fn new_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_contig(Shape(self.clone()), offset)
    }
}

/* #endregion */

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn test_layout() {
//         let layout = Layout::<Ix<3>> { shape: [1, 2, 3], stride: [6, 3, 1],
// offset: 0 };         assert_eq!(layout.shape, [1, 2, 3]);
//         assert_eq!(layout.stride, [6, 3, 1]);
//         assert_eq!(layout.offset, 0);
//     }

//     #[test]
//     fn test_shape() {
//         let shape: [usize; 3] = [4, 2, 3];
//         assert_eq!(shape.rank(), 3);
//         assert_eq!(shape.size(), 24);
//         assert_eq!(shape.stride_f_contig(), [1, 4, 8]);
//         assert_eq!(shape.stride_c_contig(), [6, 3, 1]);
//     }

//     #[test]
//     fn test_shape_zero_dim() {
//         let shape: [usize; 0] = [];
//         assert_eq!(shape.rank(), 0);
//         assert_eq!(shape.size(), 1);
//         assert_eq!(shape.stride_f_contig(), []);
//         assert_eq!(shape.stride_c_contig(), []);
//     }

//     #[test]
//     fn test_shape_dyn() {
//         let shape: Vec<usize> = vec![4, 2, 3];
//         assert_eq!(shape.rank(), 3);
//         assert_eq!(shape.size(), 24);
//         assert_eq!(shape.stride_f_contig(), vec![1, 4, 8]);
//         assert_eq!(shape.stride_c_contig(), vec![6, 3, 1]);
//     }

//     #[test]
//     fn test_shape_dyn_zero_dim() {
//         let shape: Vec<usize> = vec![];
//         assert_eq!(shape.rank(), 0);
//         assert_eq!(shape.size(), 1);
//         assert_eq!(shape.stride_f_contig(), vec![]);
//         assert_eq!(shape.stride_c_contig(), vec![]);
//     }

//     #[test]
//     fn test_stride() {
//         let stride: [isize; 3] = [6, 3, 1];
//         assert_eq!(stride.rank(), 3);
//         assert_eq!(stride.is_f_prefer(), false);
//         assert_eq!(stride.is_c_prefer(), true);
//         let stride: [isize; 3] = [1, 4, 8];
//         assert_eq!(stride.rank(), 3);
//         assert_eq!(stride.is_f_prefer(), true);
//         assert_eq!(stride.is_c_prefer(), false);
//     }

//     #[test]
//     fn test_stride_zero_dim() {
//         let stride: [isize; 0] = [];
//         assert_eq!(stride.rank(), 0);
//         assert_eq!(stride.is_f_prefer(), true);
//         assert_eq!(stride.is_c_prefer(), true);
//     }

//     #[test]
//     fn test_stride_dyn() {
//         let stride: Vec<isize> = vec![6, 3, 1];
//         assert_eq!(stride.rank(), 3);
//         assert_eq!(stride.is_f_prefer(), false);
//         assert_eq!(stride.is_c_prefer(), true);
//         let stride: Vec<isize> = vec![1, 4, 8];
//         assert_eq!(stride.rank(), 3);
//         assert_eq!(stride.is_f_prefer(), true);
//         assert_eq!(stride.is_c_prefer(), false);
//     }

//     #[test]
//     fn test_stride_dyn_zero_dim() {
//         let stride: Vec<isize> = vec![];
//         assert_eq!(stride.rank(), 0);
//         assert_eq!(stride.is_f_prefer(), true);
//         assert_eq!(stride.is_c_prefer(), true);
//     }

//     #[test]
//     fn test_layout_trait() {
//         let layout = Layout::<Ix<3>> { shape: [1, 2, 3], stride: [6, 3, 1],
// offset: 0 };         assert_eq!(layout.shape(), [1, 2, 3]);
//         assert_eq!(layout.stride(), [6, 3, 1]);
//         assert_eq!(layout.offset(), 0);
//         assert_eq!(layout.ndim(), 3);
//         assert_eq!(layout.size(), 6);
//         assert_eq!(layout.is_f_prefer(), false);
//         assert_eq!(layout.is_c_prefer(), true);
//         assert_eq!(layout.is_f_contig(), false);
//         assert_eq!(layout.is_c_contig(), true);
//         assert_eq!(layout.ndim(), 3);
//         assert_eq!(layout.index([0, 1, 2]), 5);
//         assert_eq!(unsafe { layout.index_uncheck([0, 1, 2]) }, 5);
//     }

//     #[test]
//     fn test_layout_trait_dyn() {
//         let layout = Layout::<IxD> { shape: vec![1, 2, 3], stride: vec![6, 3,
// 1], offset: 0 };         assert_eq!(layout.shape(), vec![1, 2, 3]);
//         assert_eq!(layout.stride(), vec![6, 3, 1]);
//         assert_eq!(layout.offset(), 0);
//         assert_eq!(layout.ndim(), 3);
//         assert_eq!(layout.size(), 6);
//         assert_eq!(layout.is_f_prefer(), false);
//         assert_eq!(layout.is_c_prefer(), true);
//         assert_eq!(layout.is_f_contig(), false);
//         assert_eq!(layout.is_c_contig(), true);
//         assert_eq!(layout.ndim(), 3);
//         assert_eq!(layout.index(vec![0, 1, 2]), 5);
//         assert_eq!(unsafe { layout.index_uncheck(vec![0, 1, 2]) }, 5);
//     }

//     #[test]
//     fn test_anyway() {
//         let layout = Layout::<Ix<4>> { shape: [8, 10, 0, 15], stride: [3, 4,
// 0, 7], offset: 0 };         println!("{:?}", layout.is_c_contig());
//         println!("{:?}", layout.is_f_contig());

//         let layout = [8, 10, 3, 15].new_contig(0);
//         println!("{:?}", layout);
//     }
// }
