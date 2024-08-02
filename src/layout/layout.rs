use crate::{Error, Result};
use core::fmt::Debug;
use core::ops::{Deref, DerefMut};

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

pub trait DimBaseAPI: AsMut<[usize]> + AsRef<[usize]> + Debug + Clone {
    type Shape: AsMut<[usize]> + AsRef<[usize]> + Debug + Clone;
    type Stride: AsMut<[isize]> + AsRef<[isize]> + Debug + Clone;
}

impl<const N: usize> DimBaseAPI for Ix<N> {
    type Shape = [usize; N];
    type Stride = [isize; N];
}

impl DimBaseAPI for IxD {
    type Shape = Vec<usize>;
    type Stride = Vec<isize>;
}

/* #endregion */

/* #region Strides */

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
        D::ndim(self)
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
            if !((stride[i] > stride[i - 1]) && (stride[i - 1] > 0) && (stride[i] > 0)) {
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
            if !((stride[i] < stride[i - 1]) && (stride[i - 1] > 0) && (stride[i] > 0)) {
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

/* #endregion Strides */

/* #region Shape */

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
        D::ndim(self)
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
            stride[i] = stride[i - 1] * shape[i - 1] as isize;
        }
        Stride(stride)
    }

    fn stride_c_contig(shape: &Shape<Ix<N>>) -> Stride<Ix<N>> {
        let mut stride = [1; N];
        if N == 0 {
            return Stride(stride);
        }
        for i in (0..N - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1] as isize;
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

/* #endregion */

/* #region Struct Definitions */

#[derive(Debug, Clone)]
pub struct Layout<D>
where
    D: DimBaseAPI,
{
    pub(crate) shape: Shape<D>,
    pub(crate) stride: Stride<D>,
    pub(crate) offset: usize,
}

/* #endregion */

/* #region Layout */

pub trait DimLayoutAPI: DimBaseAPI {
    /// Shape of tensor. Getter function.
    fn shape(layout: &Layout<Self>) -> Shape<Self>;
    fn as_shape(layout: &Layout<Self>) -> &Shape<Self>;

    /// Stride of tensor. Getter function.
    fn stride(layout: &Layout<Self>) -> Stride<Self>;
    fn stride_ref(layout: &Layout<Self>) -> &Stride<Self>;

    /// Starting offset of tensor. Getter function.
    fn offset(layout: &Layout<Self>) -> usize;

    /// Number of dimensions of tensor.
    fn ndim(layout: &Layout<Self>) -> usize;

    /// Total number of elements in tensor.
    fn size(layout: &Layout<Self>) -> usize;

    /// Whether this tensor is f-preferred.
    fn is_f_prefer(layout: &Layout<Self>) -> bool;

    /// Whether this tensor is c-preferred.
    fn is_c_prefer(layout: &Layout<Self>) -> bool;

    /// Whether this tensor is f-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus f-contiguous.
    fn is_f_contig(layout: &Layout<Self>) -> bool;

    /// Whether this tensor is c-contiguous.
    fn is_c_contig(layout: &Layout<Self>) -> bool;

    /// Generate new layout by providing everything.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Shape and stride length mismatch
    fn new(shape: Shape<Self>, stride: Stride<Self>, offset: usize) -> Layout<Self>;

    /// Index of tensor by list of indexes to dimensions.
    fn try_index(layout: &Layout<Self>, index: Self) -> Result<usize>;

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Panics
    ///
    /// This function panics when
    /// - Negative index
    /// - Index greater than shape
    fn index(layout: &Layout<Self>, index: Self) -> usize;

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    unsafe fn index_uncheck(layout: &Layout<Self>, index: Self) -> usize;

    /// Index range bounds of current layout. This bound is [min, max), which
    /// could be feed into range (min..max). If min == max, then this layout
    /// should not contains any element.
    ///
    /// This function will raise error when minimum index is smaller than zero.
    fn bounds_index(layout: &Layout<Self>) -> Result<(usize, usize)>;

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
    fn check_strides(layout: &Layout<Self>) -> Result<()>;
}

impl<D> Layout<D>
where
    D: DimLayoutAPI,
{
    pub fn shape(&self) -> Shape<D> {
        D::shape(self)
    }

    pub fn as_shape(&self) -> &Shape<D> {
        D::as_shape(self)
    }

    pub fn stride(&self) -> Stride<D> {
        D::stride(self)
    }

    pub fn stride_ref(&self) -> &Stride<D> {
        D::stride_ref(self)
    }

    pub fn offset(&self) -> usize {
        D::offset(self)
    }

    pub fn ndim(&self) -> usize {
        D::ndim(self)
    }

    pub fn size(&self) -> usize {
        D::ndim(self)
    }

    pub fn is_f_prefer(&self) -> bool {
        D::is_f_prefer(self)
    }

    pub fn is_c_prefer(&self) -> bool {
        D::is_c_prefer(self)
    }

    pub fn is_f_contig(&self) -> bool {
        D::is_f_contig(self)
    }

    pub fn new(shape: Shape<D>, stride: Stride<D>, offset: usize) -> Self {
        D::new(shape, stride, offset)
    }

    pub fn try_index(&self, index: D) -> Result<usize> {
        D::try_index(self, index)
    }

    pub fn index(&self, index: D) -> usize {
        <D as DimLayoutAPI>::index(self, index)
    }

    pub unsafe fn index_uncheck(&self, index: D) -> usize {
        D::index_uncheck(self, index)
    }

    pub fn bounds_index(&self) -> Result<(usize, usize)> {
        D::bounds_index(self)
    }

    pub fn check_strides(&self) -> Result<()> {
        D::check_strides(self)
    }
}

impl<D> DimLayoutAPI for D
where
    D: DimBaseAPI + DimStrideAPI + DimShapeAPI,
{
    fn shape(layout: &Layout<D>) -> Shape<D> {
        layout.shape.clone()
    }

    fn as_shape(layout: &Layout<D>) -> &Shape<D> {
        &layout.shape
    }

    fn stride(layout: &Layout<D>) -> Stride<D> {
        layout.stride.clone()
    }

    fn stride_ref(layout: &Layout<D>) -> &Stride<D> {
        &layout.stride
    }

    fn offset(layout: &Layout<D>) -> usize {
        layout.offset
    }

    fn ndim(layout: &Layout<D>) -> usize {
        layout.shape.as_ref().len()
    }

    fn size(layout: &Layout<D>) -> usize {
        layout.shape.size()
    }

    fn is_f_prefer(layout: &Layout<D>) -> bool {
        layout.stride.is_f_prefer()
    }

    fn is_c_prefer(layout: &Layout<D>) -> bool {
        layout.stride.is_c_prefer()
    }

    fn is_f_contig(layout: &Layout<D>) -> bool {
        let mut acc = 1;
        let mut contig = true;
        for (&s, &d) in layout.stride.as_ref().iter().zip(layout.shape.as_ref().iter()) {
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

    fn is_c_contig(layout: &Layout<D>) -> bool {
        let mut acc = 1;
        let mut contig = true;
        for (&s, &d) in layout.stride.as_ref().iter().zip(layout.shape.as_ref().iter()).rev() {
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

    fn new(shape: Shape<D>, stride: Stride<D>, offset: usize) -> Layout<D> {
        Layout { shape, stride, offset }
    }

    fn try_index(layout: &Layout<D>, index: D) -> Result<usize> {
        let mut pos = layout.offset() as isize;
        let index = index.as_ref();
        let shape = layout.shape.as_ref();
        let stride = layout.stride.as_ref();

        for i in 0..layout.ndim() {
            if index[i] >= shape[i] {
                return Err(Error::IndexOutOfBound {
                    index: index[i] as isize,
                    bound: shape[i] as isize,
                });
            }
            pos += stride[i] * index[i] as isize;
        }
        if pos < 0 {
            return Err(Error::IndexOutOfBound { index: pos, bound: 0 });
        }
        return Ok(pos as usize);
    }

    fn index(layout: &Layout<D>, index: D) -> usize {
        layout.try_index(index).unwrap()
    }

    unsafe fn index_uncheck(layout: &Layout<D>, index: D) -> usize {
        let mut pos = layout.offset as isize;
        let index = index.as_ref();
        let stride = layout.stride.as_ref();
        for i in 0..layout.ndim() {
            pos += stride[i] * index[i] as isize;
        }
        return pos as usize;
    }

    fn bounds_index(layout: &Layout<D>) -> Result<(usize, usize)> {
        let offset = layout.offset;

        if layout.ndim() == 0 {
            return Ok((offset, offset));
        }

        let n = layout.ndim();
        let shape = layout.shape.as_ref();
        let stride = layout.stride.as_ref();
        let mut min = offset as isize;
        let mut max = offset as isize;

        for i in 0..n {
            if shape[i] == 0 {
                return Ok((offset, offset));
            }
            if stride[i] > 0 {
                max += stride[i] * (shape[i] as isize - 1);
            } else {
                min += stride[i] * shape[i] as isize;
            }
        }
        if min < 0 {
            return Err(Error::IndexOutOfBound { index: min, bound: 0 });
        } else {
            return Ok((min as usize, max as usize + 1));
        }
    }

    fn check_strides(layout: &Layout<D>) -> Result<()> {
        let shape = layout.shape.as_ref();
        let stride = layout.stride.as_ref();
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
                    bound: stride_sorted[i + 1] as isize,
                });
            }
        }
        return Ok(());
    }
}

pub trait DimLayoutContigAPI: DimBaseAPI {
    /// Generate new layout by providing shape and offset; stride fits into
    /// c-contiguous.
    fn new_c_contig(&self, offset: usize) -> Layout<Self>;

    /// Generate new layout by providing shape and offset; stride fits into
    /// f-contiguous.
    fn new_f_contig(&self, offset: usize) -> Layout<Self>;

    /// Generate new layout by providing shape and offset; Whether c-contiguous
    /// or f-contiguous depends on cargo feature `c_prefer`.
    fn new_contig(&self, offset: usize) -> Layout<Self>;

    /// Simplified function to generate c-contiguous layout. See also
    /// [DimLayoutContigAPI::new_c_contig].
    fn c(&self) -> Layout<Self> {
        self.new_c_contig(0)
    }

    /// Simplified function to generate f-contiguous layout. See also
    /// [DimLayoutContigAPI::new_f_contig].
    fn f(&self) -> Layout<Self> {
        self.new_f_contig(0)
    }
}

impl<const N: usize> DimLayoutContigAPI for Ix<N> {
    fn new_c_contig(&self, offset: usize) -> Layout<Ix<N>> {
        let shape = Shape::<Self>(*self);
        let stride = shape.stride_c_contig();
        Layout { shape, stride, offset }
    }

    fn new_f_contig(&self, offset: usize) -> Layout<Ix<N>> {
        let shape = Shape::<Self>(*self);
        let stride = shape.stride_f_contig();
        Layout { shape, stride, offset }
    }

    fn new_contig(&self, offset: usize) -> Layout<Ix<N>> {
        match crate::C_PREFER {
            true => self.new_c_contig(offset),
            false => self.new_f_contig(offset),
        }
    }
}

impl DimLayoutContigAPI for IxD {
    fn new_c_contig(&self, offset: usize) -> Layout<IxD> {
        let shape = Shape::<Self>(self.clone());
        let stride = shape.stride_c_contig();
        Layout { shape, stride, offset }
    }

    fn new_f_contig(&self, offset: usize) -> Layout<IxD> {
        let shape = Shape::<Self>(self.clone());
        let stride = shape.stride_f_contig();
        Layout { shape, stride, offset }
    }

    fn new_contig(&self, offset: usize) -> Layout<IxD> {
        match crate::C_PREFER {
            true => self.new_c_contig(offset),
            false => self.new_f_contig(offset),
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

impl<const N: usize> From<Ix<N>> for Layout<Ix<N>> {
    fn from(index: Ix<N>) -> Self {
        let shape: Shape<Ix<N>> = Shape(index);
        let stride: Stride<Ix<N>> = shape.stride_contig();
        Layout { shape, stride, offset: 0 }
    }
}

impl From<IxD> for Layout<IxD> {
    fn from(index: IxD) -> Self {
        let shape: Shape<IxD> = Shape(index);
        let stride: Stride<IxD> = shape.stride_contig();
        Layout { shape, stride, offset: 0 }
    }
}

/* #endregion */

/* #region Dim/Shape to Layout */

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
