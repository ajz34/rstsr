use crate::{Error, Result};
use std::ops::{Deref, DerefMut};

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
    fn stride_contig(&self) -> Stride<Self::Dim>;
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

    fn stride_contig(&self) -> Stride<Ix<N>> {
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

    fn stride_contig(&self) -> Stride<IxD> {
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

    /// Index range bounds of current layout. This bound is [min, max), which
    /// could be feed into range (min..max). If min == max, then this layout
    /// should not contains any element.
    ///
    /// This function will raise error when minimum index is smaller than zero.
    fn bounds_index(&self) -> Result<(usize, usize)>;

    /// Check if strides is correct.
    ///
    /// This will check if all number of elements in dimension of small strides
    /// is less than larger strides. For example of valid stride:
    /// ```output
    /// shape:  (3,    2,  6)  -> sorted ->  ( 3,   6,   2)
    /// stride: (3, -300, 15)  -> sorted ->  ( 3,  15, 300)
    /// number of elements:                    9,  90,
    /// stride of next dimension              15, 300,
    /// number of elem < stride of next dim?   +,   +,
    /// ```
    ///
    /// # TODO
    ///
    /// Correctness of this function is not ensured.
    fn check_strides(&self) -> Result<()>;
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

    fn bounds_index(&self) -> Result<(usize, usize)> {
        if self.ndim() == 0 {
            return Ok((self.offset(), self.offset()));
        }
        let n = self.ndim();
        let shape = self.shape_ref().as_ref();
        let stride = self.stride_ref().as_ref();
        let mut min = self.offset() as isize;
        let mut max = self.offset() as isize;

        for i in 0..n {
            if shape[i] == 0 {
                return Ok((self.offset(), self.offset()));
            }
            if stride[i] > 0 {
                max += stride[i] * (shape[i] as isize - 1);
            } else {
                min += stride[i] * shape[i] as isize;
            }
        }
        if min < 0 {
            return Err(Error::IndexOutOfBound { index: min, shape: 0 });
        } else {
            return Ok((min as usize, max as usize + 1));
        }
    }

    fn check_strides(&self) -> Result<()> {
        let Self { shape: Shape(shape), stride: Stride(stride), .. } = self;
        let shape: Vec<usize> = shape.as_ref().into();
        let stride: Vec<isize> = stride.as_ref().into();
        if shape.len() != stride.len() {
            return Err(Error::USizeNotMatch { got: shape.len(), expect: stride.len() });
        }
        let n = shape.len();
        if n <= 1 {
            return Ok(());
        }

        let mut indices = (0..n).collect::<Vec<usize>>();
        indices.sort_by_key(|&i| stride[i].abs());
        let shape_sorted = indices.iter().map(|&i| shape[i]).collect::<Vec<usize>>();
        let stride_sorted = indices.iter().map(|&i| stride[i] as usize).collect::<Vec<usize>>();

        for i in 0..n - 1 {
            if shape_sorted[i] * stride_sorted[i] > stride_sorted[i + 1] {
                return Err(Error::IndexOutOfBound {
                    index: (shape_sorted[i] * stride_sorted[i]) as isize,
                    shape: stride_sorted[i + 1] as isize,
                });
            }
        }
        return Ok(());
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

impl<const N: usize> TryFrom<Layout<Ix<N>>> for Layout<IxD> {
    type Error = Error;

    fn try_from(layout: Layout<Ix<N>>) -> Result<Self> {
        let Layout { shape: Shape(shape), stride: Stride(stride), offset } = layout;
        let layout =
            Layout { shape: Shape(shape.to_vec()), stride: Stride(stride.to_vec()), offset };
        Ok(layout)
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

/* #region Dim/Shape to Layout */

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

impl<const N: usize> From<Ix<N>> for Layout<Ix<N>> {
    fn from(shape: Ix<N>) -> Self {
        shape.new_contig(0)
    }
}

impl From<IxD> for Layout<IxD> {
    fn from(shape: IxD) -> Self {
        shape.new_contig(0)
    }
}

impl<const N: usize> From<Ix<N>> for Shape<Ix<N>> {
    fn from(shape: Ix<N>) -> Self {
        Shape(shape)
    }
}

impl From<IxD> for Shape<IxD> {
    fn from(shape: IxD) -> Self {
        Shape(shape)
    }
}

/* #endregion */

#[cfg(test)]
mod playground {
    use super::*;

    #[test]
    fn test0() {
        let shape: [usize; 3] = [3, 2, 6];
        let stride: [isize; 3] = [3, -300, 15];
        let layout = Layout::<Ix3> { shape: Shape(shape), stride: Stride(stride), offset: 917 };
        let _ = layout.check_strides();
    }
}
