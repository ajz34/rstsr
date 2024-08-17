//! Layout of tensor.

use crate::prelude_dev::*;
use itertools::izip;

/* #region Struct Definitions */

/// Layout of tensor.
///
/// Layout is a struct that contains shape, stride, and offset of tensor.
/// - Shape is the size of each dimension of tensor.
/// - Stride is the number of elements to skip to get to the next element in
///   each dimension.
/// - Offset is the starting position of tensor.
#[doc = include_str!("readme.md")]
#[derive(Clone)]
pub struct Layout<D>
where
    D: DimBaseAPI,
{
    // essential definitions to layout
    pub(crate) shape: Shape<D>,
    pub(crate) stride: Stride<D>,
    pub(crate) offset: usize,
}

/* #endregion */

/* #region Layout */

/// Getter functions for layout.
impl<D> Layout<D>
where
    D: DimBaseAPI,
{
    /// Shape of tensor. Getter function.
    pub fn shape(&self) -> Shape<D> {
        self.shape.clone()
    }

    /// Shape of tensor as reference. Getter function.
    pub fn shape_ref(&self) -> &Shape<D> {
        &self.shape
    }

    /// Shape of tensor as `[usize; N]` for fixed dim, or `Vec<usize>` for
    /// duynamic dim.
    pub fn shape_as_array(&self) -> D {
        self.shape.0.clone()
    }

    /// Stride of tensor. Getter function.
    pub fn stride(&self) -> Stride<D> {
        self.stride.clone()
    }

    /// Stride of tensor as reference. Getter function.
    pub fn stride_ref(&self) -> &Stride<D> {
        &self.stride
    }

    /// Stride of tensor as `[isize; N]` for fixed dim, or `Vec<isize>` for
    /// duynamic dim.
    pub fn stride_as_array(&self) -> D::Stride {
        self.stride.0.clone()
    }

    /// Starting offset of tensor. Getter function.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of dimensions of tensor.
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
}

/// Properties of layout.
impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Total number of elements in tensor.
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Whether this tensor is f-preferred.
    pub fn is_f_prefer(&self) -> bool {
        self.stride.is_f_prefer()
    }

    /// Whether this tensor is c-preferred.
    pub fn is_c_prefer(&self) -> bool {
        self.stride.is_c_prefer()
    }

    /// Whether this tensor is f-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus f-contiguous.
    pub fn is_f_contig(&self) -> bool {
        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut acc = 1;
        let mut contig = true;
        for (&s, &d) in stride.iter().zip(shape.iter()) {
            if d == 0 {
                return true;
            } else if s != acc {
                contig = false;
            }
            acc *= d as isize;
        }
        return contig;
    }

    /// Whether this tensor is c-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus c-contiguous.
    pub fn is_c_contig(&self) -> bool {
        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut acc = 1;
        let mut contig = true;
        for (&s, &d) in stride.iter().zip(shape.iter()).rev() {
            if d == 0 {
                return true;
            } else if s != acc {
                contig = false;
            }
            acc *= d as isize;
        }
        return contig;
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// This function does not optimized for performance.
    pub fn try_index(&self, index: D) -> Result<usize> {
        let mut pos = self.offset() as isize;
        let index = index.as_ref();
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();

        for (&idx, &shp, &strd) in izip!(index.iter(), shape.iter(), stride.iter()) {
            rstsr_pattern!(idx, 0..shp, ValueOutOfRange)?;
            pos += strd * idx as isize;
        }
        rstsr_pattern!(pos, 0.., ValueOutOfRange)?;
        return Ok(pos as usize);
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// This function does not optimized for performance.
    ///
    /// # Panics
    ///
    /// - Negative index
    /// - Index greater than shape
    pub fn index(&self, index: D) -> usize {
        self.try_index(index).unwrap()
    }

    /// Index range bounds of current layout. This bound is [min, max), which
    /// could be feed into range (min..max). If min == max, then this layout
    /// should not contains any element.
    ///
    /// This function will raise error when minimum index is smaller than zero.
    pub fn bounds_index(&self) -> Result<(usize, usize)> {
        let n = self.ndim();
        let offset = self.offset;
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();

        if n == 0 {
            return Ok((offset, offset));
        }

        let mut min = offset as isize;
        let mut max = offset as isize;

        for i in 0..n {
            if shape[i] == 0 {
                return Ok((offset, offset));
            }
            if stride[i] > 0 {
                max += stride[i] * (shape[i] as isize - 1);
            } else {
                min += stride[i] * (shape[i] as isize - 1);
            }
        }
        rstsr_pattern!(min, 0.., ValueOutOfRange)?;
        return Ok((min as usize, max as usize + 1));
    }

    /// Check if strides is correct (no elemenets can overlap).
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
    /// Correctness of this function is not fully ensured.
    pub fn check_strides(&self) -> Result<()> {
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();
        rstsr_assert_eq!(shape.len(), stride.len(), InvalidLayout)?;
        let n = shape.len();
        if n <= 1 {
            return Ok(());
        }

        let mut indices = (0..n).collect::<Vec<usize>>();
        indices.sort_by_key(|&i| stride[i].abs());
        let shape_sorted = indices.iter().map(|&i| shape[i]).collect::<Vec<_>>();
        let stride_sorted = indices.iter().map(|&i| stride[i].abs() as usize).collect::<Vec<_>>();

        for i in 0..n - 1 {
            rstsr_pattern!(
                shape_sorted[i] * stride_sorted[i],
                0..stride_sorted[i + 1] + 1,
                InvalidLayout
            )?;
        }
        return Ok(());
    }
}

/// Constructors of layout. See also [`DimLayoutContigAPI`] layout from shape
/// directly.
impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Generate new layout by providing everything.
    ///
    /// # Panics
    ///
    /// - Shape and stride length mismatch
    /// - Strides is correct (no elements can overlap)
    /// - Minimum bound is not negative
    pub fn new(shape: Shape<D>, stride: Stride<D>, offset: usize) -> Self {
        let layout = Layout { shape, stride, offset };
        layout.bounds_index().unwrap();
        layout.check_strides().unwrap();
        return layout;
    }

    /// Generate new layout by providing everything, without checking bounds and
    /// strides.
    ///
    /// # Safety
    ///
    /// This function does not check whether layout is valid.
    pub unsafe fn new_unchecked(shape: Shape<D>, stride: Stride<D>, offset: usize) -> Self {
        Layout { shape, stride, offset }
    }
}

pub trait DimLayoutAPI: DimBaseAPI + DimStrideAPI + DimShapeAPI {
    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    unsafe fn index_uncheck_by_ref(layout: &Layout<Self>, index: &Self) -> usize {
        let mut pos = layout.offset as isize;
        let index = index.as_ref();
        let stride = layout.stride.as_ref();
        stride.iter().zip(index.iter()).for_each(|(&s, &i)| pos += s * i as isize);
        return pos as usize;
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    unsafe fn index_uncheck(layout: &Layout<Self>, index: Self) -> usize {
        Self::index_uncheck_by_ref(layout, &index)
    }
}

impl DimLayoutAPI for Ix<0> {
    unsafe fn index_uncheck(_layout: &Layout<Self>, _index: Self) -> usize {
        0
    }
}

impl DimLayoutAPI for Ix<1> {
    unsafe fn index_uncheck(layout: &Layout<Self>, index: Self) -> usize {
        let index = index.as_ref();
        let stride = layout.stride.as_ref();
        (layout.offset as isize + stride[0] * index[0] as isize) as usize
    }
}

impl DimLayoutAPI for Ix<2> {
    unsafe fn index_uncheck(layout: &Layout<Self>, index: Self) -> usize {
        let index = index.as_ref();
        let stride = layout.stride.as_ref();
        (layout.offset as isize + stride[0] * index[0] as isize + stride[1] * index[1] as isize)
            as usize
    }
}

impl DimLayoutAPI for Ix<3> {
    unsafe fn index_uncheck(layout: &Layout<Self>, index: Self) -> usize {
        let index = index.as_ref();
        let stride = layout.stride.as_ref();
        (layout.offset as isize
            + stride[0] * index[0] as isize
            + stride[1] * index[1] as isize
            + stride[2] * index[2] as isize) as usize
    }
}

impl DimLayoutAPI for Ix<4> {
    unsafe fn index_uncheck(layout: &Layout<Self>, index: Self) -> usize {
        let index = index.as_ref();
        let stride = layout.stride.as_ref();
        (layout.offset as isize
            + stride[0] * index[0] as isize
            + stride[1] * index[1] as isize
            + stride[2] * index[2] as isize
            + stride[3] * index[3] as isize) as usize
    }
}

impl DimLayoutAPI for Ix<5> {}
impl DimLayoutAPI for Ix<6> {}
impl DimLayoutAPI for Ix<7> {}
impl DimLayoutAPI for Ix<8> {}
impl DimLayoutAPI for Ix<9> {}
impl DimLayoutAPI for IxD {}

impl<D> Layout<D>
where
    D: DimLayoutAPI,
{
    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    pub unsafe fn index_uncheck(&self, index: D) -> usize {
        D::index_uncheck(self, index)
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    pub unsafe fn index_uncheck_by_ref(&self, index: &D) -> usize {
        D::index_uncheck_by_ref(self, index)
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
            shape: Shape(shape.try_into().map_err(|_| {
                Error::InvalidLayout(format!("Cannot convert IxD to Ix< {:} >", N))
            })?),
            stride: Stride(stride.try_into().map_err(|_| {
                Error::InvalidLayout(format!("Cannot convert IxD to Ix< {:} >", N))
            })?),
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

impl<D> Layout<D>
where
    D: DimBaseAPI,
{
    /// Convert layout to another dimension.
    ///
    /// This is mostly used when converting static dimension to dynamic
    /// dimension or vice versa.
    pub fn into_dim<T>(self) -> Result<Layout<T>>
    where
        T: DimBaseAPI,
        Layout<D>: TryInto<Layout<T>, Error = Error>,
    {
        return self.try_into();
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
