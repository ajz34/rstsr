use crate::{Error, Result};

/* #region Struct Definitions */

#[derive(Debug, Clone)]
pub struct Dim<I>
where
    I: AsMut<[usize]>,
{
    _phantom: std::marker::PhantomData<I>,
}

#[derive(Debug, Clone)]
pub struct Layout<D>
where
    D: DimAPI,
{
    pub(crate) shape: D::Shape,
    pub(crate) stride: D::Stride,
    pub(crate) offset: usize,
}

/* #endregion */

/* #region Dimension */

pub trait DimAPI {
    type Index: AsMut<[usize]> + core::fmt::Debug + Clone;
    type Shape: AsMut<[usize]> + core::fmt::Debug + Clone;
    type Stride: AsMut<[isize]> + core::fmt::Debug + Clone;
}

impl<const N: usize> DimAPI for Dim<[usize; N]> {
    type Index = [usize; N];
    type Shape = [usize; N];
    type Stride = [isize; N];
}

impl DimAPI for Dim<Vec<usize>> {
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
pub type Ix7 = Ix<7>;
pub type Ix8 = Ix<8>;
pub type Ix9 = Ix<9>;
pub type IxD = Dim<Vec<usize>>;
pub type IxDyn = IxD;

/* #endregion */

/* #region Shape */

pub trait ShapeAPI: AsMut<[usize]> {
    /// Number of dimensions of the shape.
    fn rank(&self) -> usize;
    /// Total number of elements in tensor.
    fn size(&self) -> usize;
    /// Stride for a f-contiguous tensor using this shape.
    fn stride_f_contig(&self) -> impl StrideAPI;
    /// Stride for a c-contiguous tensor using this shape.
    fn stride_c_contig(&self) -> impl StrideAPI;
    /// Stride for contiguous tensor using this shape.
    /// Whether c-contiguous or f-contiguous will depends on cargo feature `c_prefer`.
    fn stride_config(&self) -> impl StrideAPI;
}

impl<const N: usize> ShapeAPI for [usize; N] {
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

    fn stride_config(&self) -> [isize; N] {
        match crate::C_PREFER {
            true => self.stride_c_contig(),
            false => self.stride_f_contig(),
        }
    }
}

impl ShapeAPI for Vec<usize> {
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

    fn stride_config(&self) -> Vec<isize> {
        match crate::C_PREFER {
            true => self.stride_c_contig(),
            false => self.stride_f_contig(),
        }
    }
}

/* #endregion */

/* #region Strides */

pub trait StrideAPI: AsMut<[isize]> {
    /// Number of dimensions of the shape.
    fn rank(&self) -> usize;
    /// Check if the strides are f-preferred.
    fn is_f_prefer(&self) -> bool;
    /// Check if the strides are c-preferred.
    fn is_c_prefer(&self) -> bool;
}

impl<const N: usize> StrideAPI for [isize; N] {
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

impl StrideAPI for Vec<isize> {
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

pub trait LayoutAPI: Sized {
    type Shape: ShapeAPI + AsRef<[usize]>;
    type Stride: StrideAPI + AsRef<[isize]>;

    /// Shape of tensor. Getter function.
    fn shape(&self) -> Self::Shape;
    fn shape_ref(&self) -> &Self::Shape;

    /// Stride of tensor. Getter function.
    fn stride(&self) -> Self::Stride;
    fn stride_ref(&self) -> &Self::Stride;

    /// Starting offset of tensor. Getter function.
    fn offset(&self) -> usize;

    /// Number of dimensions of tensor.
    fn ndim(&self) -> usize {
        self.shape_ref().rank()
    }

    /// Total number of elements in tensor.
    fn size(&self) -> usize {
        self.shape_ref().size()
    }

    /// Whether this tensor is f-preferred.
    fn is_f_prefer(&self) -> bool {
        self.stride_ref().is_f_prefer()
    }

    /// Whether this tensor is c-preferred.
    fn is_c_prefer(&self) -> bool {
        self.stride_ref().is_c_prefer()
    }

    /// Whether this tensor is f-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus f-contiguous.
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

    /// Whether this tensor is c-contiguous.
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

    /// Generate new layout by providing shape and offset; Whether c-contiguous or f-contiguous depends on cargo feature `c_prefer`.
    fn new_contig(shape: Self::Shape, offset: usize) -> Self {
        match crate::C_PREFER {
            true => Self::new_c_contig(shape, offset),
            false => Self::new_f_contig(shape, offset),
        }
    }

    /// Index of tensor by list of indexes to dimensions.
    fn try_index(&self, index: Self::Shape) -> Result<usize> {
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

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Negative index
    /// - Index greater than shape
    fn index(&self, index: Self::Shape) -> usize {
        self.try_index(index).unwrap()
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    unsafe fn index_uncheck(&self, index: Self::Shape) -> usize {
        let mut pos = self.offset() as isize;
        for i in 0..self.ndim() {
            pos += self.stride_ref().as_ref()[i] * index.as_ref()[i] as isize;
        }
        return pos as usize;
    }
}

impl<const N: usize> LayoutAPI for Layout<Ix<N>> {
    type Shape = [usize; N];
    type Stride = [isize; N];

    fn shape(&self) -> [usize; N] {
        self.shape
    }

    fn shape_ref(&self) -> &Self::Shape {
        &self.shape
    }

    fn stride(&self) -> [isize; N] {
        self.stride
    }

    fn stride_ref(&self) -> &Self::Stride {
        &self.stride
    }

    fn offset(&self) -> usize {
        self.offset
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
}

impl LayoutAPI for Layout<IxD> {
    type Shape = Vec<usize>;
    type Stride = Vec<isize>;

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn shape_ref(&self) -> &Self::Shape {
        &self.shape
    }

    fn stride(&self) -> Vec<isize> {
        self.stride.clone()
    }

    fn stride_ref(&self) -> &Self::Stride {
        &self.stride
    }

    fn offset(&self) -> usize {
        self.offset
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
}

/* #endregion Layout */

/* #region Dimension Conversion */

impl<const N: usize> From<Layout<Ix<N>>> for Layout<IxD> {
    fn from(layout: Layout<Ix<N>>) -> Self {
        let Layout { shape, stride, offset } = layout;
        Layout { shape: shape.to_vec(), stride: stride.to_vec(), offset }
    }
}

impl<const N: usize> TryFrom<Layout<IxD>> for Layout<Ix<N>> {
    type Error = Error;

    fn try_from(layout: Layout<IxD>) -> Result<Self> {
        let Layout { shape, stride, offset } = layout;
        Ok(Layout {
            shape: shape
                .try_into()
                .map_err(|_| Error::Msg(format!("Cannot convert IxD to Ix< {:} >", N)))?,
            stride: stride
                .try_into()
                .map_err(|_| Error::Msg(format!("Cannot convert IxD to Ix< {:} >", N)))?,
            offset,
        })
    }
}

/* #endregion */

/* #region Shape to Layout */

pub trait ShapeToLayoutAPI {
    type Layout: LayoutAPI;

    fn new_c_contig(&self, offset: usize) -> Self::Layout;
    fn new_f_contig(&self, offset: usize) -> Self::Layout;
    fn new_contig(&self, offset: usize) -> Self::Layout;
}

impl<const N: usize> ShapeToLayoutAPI for [usize; N] {
    type Layout = Layout<Ix<N>>;

    fn new_c_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_c_contig(self.clone(), offset)
    }
    fn new_f_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_f_contig(self.clone(), offset)
    }
    fn new_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_contig(self.clone(), offset)
    }
}

impl ShapeToLayoutAPI for Vec<usize> {
    type Layout = Layout<IxD>;

    fn new_c_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_c_contig(self.clone(), offset)
    }
    fn new_f_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_f_contig(self.clone(), offset)
    }
    fn new_contig(&self, offset: usize) -> Self::Layout {
        Self::Layout::new_contig(self.clone(), offset)
    }
}

/* #endregion */

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
        assert_eq!(layout.ndim(), 3);
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
        let layout = Layout::<IxD> { shape: vec![1, 2, 3], stride: vec![6, 3, 1], offset: 0 };
        assert_eq!(layout.shape(), vec![1, 2, 3]);
        assert_eq!(layout.stride(), vec![6, 3, 1]);
        assert_eq!(layout.offset(), 0);
        assert_eq!(layout.ndim(), 3);
        assert_eq!(layout.size(), 6);
        assert_eq!(layout.is_f_prefer(), false);
        assert_eq!(layout.is_c_prefer(), true);
        assert_eq!(layout.is_f_contig(), false);
        assert_eq!(layout.is_c_contig(), true);
        assert_eq!(layout.ndim(), 3);
        assert_eq!(layout.index(vec![0, 1, 2]), 5);
        assert_eq!(unsafe { layout.index_uncheck(vec![0, 1, 2]) }, 5);
    }

    #[test]
    fn test_anyway() {
        let layout = Layout::<Ix<4>> { shape: [8, 10, 0, 15], stride: [3, 4, 0, 7], offset: 0 };
        println!("{:?}", layout.is_c_contig());
        println!("{:?}", layout.is_f_contig());

        let layout = [8, 10, 3, 15].new_contig(0);
        println!("{:?}", layout);
    }
}
