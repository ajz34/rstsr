use crate::{Error, Result};
use super::*;
use core::fmt::Debug;
use core::fmt::Write;

/* #region Struct Definitions */

#[derive(Clone)]
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

    pub fn is_c_contig(&self) -> bool {
        D::is_c_contig(self)
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
            if d == 0 {
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
            if d == 0 {
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

/* #region Format */

impl<D> Debug for Layout<D>
where 
    D: DimBaseAPI + DimLayoutAPI,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();
        let offset = self.offset;
        let is_c_contig = self.is_c_contig();
        let is_f_contig = self.is_f_contig();
        let is_c_prefer = self.is_c_prefer();
        let is_f_prefer = self.is_f_prefer();
        let mut contig = String::new();
        if is_c_contig { write!(contig, "C")?; }
        if is_c_prefer { write!(contig, "c")?; }
        if is_f_contig { write!(contig, "F")?; }
        if is_f_prefer { write!(contig, "f")?; }
        if contig.is_empty() { write!(contig, "Custom")?; }
        write!(f, "Layout<{}>, shape: {:?}, stride: {:?}, offset: {}, contiguous: {} }}",
            core::any::type_name::<D>(),
            shape,
            stride,
            offset,
            contig,
        )?;
        Ok(())
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
        println!("{:?}", layout);
        let _ = layout.check_strides();
    }
}
