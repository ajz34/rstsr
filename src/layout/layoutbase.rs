//! Layout of tensor.
use crate::prelude_dev::*;
use core::convert::Infallible;
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
#[derive(Clone, PartialEq, Eq)]
pub struct Layout<D>
where
    D: DimBaseAPI,
{
    // essential definitions to layout
    pub(crate) shape: D,
    pub(crate) stride: D::Stride,
    pub(crate) offset: usize,
    size: usize,
}

/* #endregion */

/* #region Layout */

/// Getter functions for layout.
impl<D> Layout<D>
where
    D: DimBaseAPI,
{
    /// Shape of tensor. Getter function.
    #[inline]
    pub fn shape(&self) -> &D {
        &self.shape
    }

    /// Stride of tensor. Getter function.
    #[inline]
    pub fn stride(&self) -> &D::Stride {
        &self.stride
    }

    /// Starting offset of tensor. Getter function.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of dimensions of tensor.
    #[inline]
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
    ///
    /// # Note
    ///
    /// This function uses cached size, instead of evaluating from shape.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Whether this tensor is f-preferred.
    pub fn is_f_prefer(&self) -> bool {
        // always true for 0-dimension or 0-size tensor
        if self.ndim() == 0 || self.size() == 0 {
            return true;
        }

        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut last = 0;
        for (&s, &d) in stride.iter().zip(shape.iter()) {
            if d != 1 {
                if s < last {
                    // latter strides must larger than previous strides
                    return false;
                }
                if last == 0 && s != 1 {
                    // first stride must be 1
                    return false;
                }
                last = s;
            }
        }
        return true;
    }

    /// Whether this tensor is c-preferred.
    pub fn is_c_prefer(&self) -> bool {
        // always true for 0-dimension or 0-size tensor
        if self.ndim() == 0 || self.size() == 0 {
            return true;
        }

        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut last = 0;
        for (&s, &d) in stride.iter().zip(shape.iter()).rev() {
            if d != 1 {
                if s < last {
                    // previous strides must larger than latter strides
                    return false;
                }
                if last == 0 && s != 1 {
                    // last stride must be 1
                    return false;
                }
                last = s;
            }
        }
        return true;
    }

    /// Whether this tensor is f-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus f-contiguous.
    pub fn is_f_contig(&self) -> bool {
        // always true for 0-dimension or 0-size tensor
        if self.ndim() == 0 || self.size() == 0 {
            return true;
        }

        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut acc = 1;
        for (&s, &d) in stride.iter().zip(shape.iter()) {
            if d != 1 && s != acc {
                return false;
            }
            acc *= d as isize;
        }
        return true;
    }

    /// Whether this tensor is c-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is
    ///   not important.
    /// - When length of a dimension is zero, then tensor contains no elements,
    ///   thus c-contiguous.
    pub fn is_c_contig(&self) -> bool {
        // always true for 0-dimension or 0-size tensor
        if self.ndim() == 0 || self.size() == 0 {
            return true;
        }

        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut acc = 1;
        for (&s, &d) in stride.iter().zip(shape.iter()).rev() {
            if d != 1 && s != acc {
                return false;
            }
            acc *= d as isize;
        }
        return true;
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// This function does not optimized for performance.
    pub fn try_index(&self, index: D::Stride) -> Result<usize> {
        let mut pos = self.offset() as isize;
        let index = index.as_ref();
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();

        for (&idx, &shp, &strd) in izip!(index.iter(), shape.iter(), stride.iter()) {
            let idx = if idx < 0 { idx + shp as isize } else { idx };
            rstsr_pattern!(idx, 0..(shp as isize), ValueOutOfRange)?;
            pos += strd * idx;
        }
        rstsr_pattern!(pos, 0.., ValueOutOfRange)?;
        return Ok(pos as usize);
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// This function does not optimized for performance. Negative index
    /// allowed.
    ///
    /// # Panics
    ///
    /// - Index greater than shape
    pub fn index(&self, index: D::Stride) -> usize {
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
            return Ok((offset, offset + 1));
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
    /// Special cases
    /// - if length of tensor is zero, then strides will always be correct.
    /// - if certain dimension is one, then check for this stride will be
    ///   ignored.
    ///
    /// # TODO
    ///
    /// Correctness of this function is not fully ensured.
    pub fn check_strides(&self) -> Result<()> {
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();
        rstsr_assert_eq!(shape.len(), stride.len(), InvalidLayout)?;
        let n = shape.len();

        // unconditionally ok if no elements (length of tensor is zero)
        // unconditionally ok if 0-dimension
        if self.size() == 0 || n == 0 {
            return Ok(());
        }

        let mut indices = (0..n).filter(|&k| shape[k] > 1).collect::<Vec<_>>();
        indices.sort_by_key(|&k| stride[k].abs());
        let shape_sorted = indices.iter().map(|&k| shape[k]).collect::<Vec<_>>();
        let stride_sorted = indices.iter().map(|&k| stride[k].unsigned_abs()).collect::<Vec<_>>();

        for i in 0..indices.len() - 1 {
            // following function also checks that stride could not be zero
            rstsr_pattern!(
                shape_sorted[i] * stride_sorted[i],
                1..stride_sorted[i + 1] + 1,
                InvalidLayout,
                "Either stride be zero, or stride too small that elements in tensor can be overlapped."
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
    #[inline]
    pub fn new(shape: D, stride: D::Stride, offset: usize) -> Self {
        let layout = unsafe { Layout::new_unchecked(shape, stride, offset) };
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
    #[inline]
    pub unsafe fn new_unchecked(shape: D, stride: D::Stride, offset: usize) -> Self {
        let size = shape.shape_size();
        Layout { shape, stride, offset, size }
    }

    /// New zero shape, which number of dimensions are the same to current
    /// layout.
    #[inline]
    pub fn new_shape(&self) -> D {
        self.shape.new_shape()
    }

    /// New zero stride, which number of dimensions are the same to current
    /// layout.
    #[inline]
    pub fn new_stride(&self) -> D::Stride {
        self.shape.new_stride()
    }
}

/// Manuplation of layout.
impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Transpose layout by permutation.
    ///
    /// # See also
    ///
    /// - [`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
    /// - [Python array API: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
    pub fn transpose(&self, axes: &[isize]) -> Result<Self> {
        // check axes and cast to usize
        let n = self.ndim();
        rstsr_assert_eq!(
            axes.len(),
            n,
            InvalidLayout,
            "number of elements in axes should be the same to number of dimensions."
        )?;
        // no elements in axes can be the same
        let mut permut_used = vec![false; n];
        for &p in axes {
            let p = if p < 0 { p + n as isize } else { p };
            rstsr_pattern!(p, 0..n as isize, InvalidLayout)?;
            let p = p as usize;
            permut_used[p] = true;
        }
        rstsr_assert!(
            permut_used.iter().all(|&b| b),
            InvalidLayout,
            "axes should contain all elements from 0 to n-1."
        )?;
        let axes = axes
            .iter()
            .map(|&p| if p < 0 { p + n as isize } else { p } as usize)
            .collect::<Vec<_>>();

        let shape_old = self.shape();
        let stride_old = self.stride();
        let mut shape = self.new_shape();
        let mut stride = self.new_stride();
        for i in 0..self.ndim() {
            shape[i] = shape_old[axes[i]];
            stride[i] = stride_old[axes[i]];
        }
        return unsafe { Ok(Layout::new_unchecked(shape, stride, self.offset)) };
    }

    /// Transpose layout by permutation.
    ///
    /// This is the same function to [`Layout::transpose`]
    pub fn permute_dims(&self, axes: &[isize]) -> Result<Self> {
        self.transpose(axes)
    }

    /// Reverse axes of layout.
    pub fn reverse_axes(&self) -> Self {
        let shape_old = self.shape();
        let stride_old = self.stride();
        let mut shape = self.new_shape();
        let mut stride = self.new_stride();
        for i in 0..self.ndim() {
            shape[i] = shape_old[self.ndim() - i - 1];
            stride[i] = stride_old[self.ndim() - i - 1];
        }
        return unsafe { Layout::new_unchecked(shape, stride, self.offset) };
    }

    /// Swap axes of layout.
    pub fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<Self> {
        let axis1 = if axis1 < 0 { self.ndim() as isize + axis1 } else { axis1 };
        rstsr_pattern!(axis1, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis1 = axis1 as usize;

        let axis2 = if axis2 < 0 { self.ndim() as isize + axis2 } else { axis2 };
        rstsr_pattern!(axis2, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis2 = axis2 as usize;

        let mut shape = self.shape().clone();
        let mut stride = self.stride().clone();
        shape.as_mut().swap(axis1, axis2);
        stride.as_mut().swap(axis1, axis2);
        return unsafe { Ok(Layout::new_unchecked(shape, stride, self.offset)) };
    }
}

impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    #[inline]
    pub unsafe fn index_uncheck(&self, index: &[usize]) -> usize {
        let stride = self.stride.as_ref();
        match self.ndim() {
            0 => self.offset,
            1 => (self.offset as isize + stride[0] * index[0] as isize) as usize,
            2 => {
                (self.offset as isize
                    + stride[0] * index[0] as isize
                    + stride[1] * index[1] as isize) as usize
            },
            3 => {
                (self.offset as isize
                    + stride[0] * index[0] as isize
                    + stride[1] * index[1] as isize
                    + stride[2] * index[2] as isize) as usize
            },
            4 => {
                (self.offset as isize
                    + stride[0] * index[0] as isize
                    + stride[1] * index[1] as isize
                    + stride[2] * index[2] as isize
                    + stride[3] * index[3] as isize) as usize
            },
            _ => {
                let mut pos = self.offset as isize;
                stride.iter().zip(index.iter()).for_each(|(&s, &i)| pos += s * i as isize);
                pos as usize
            },
        }
    }

    /// Index of tensor by list of indexes.
    ///
    /// # Safety
    ///
    /// This function does not check whether index is out of bounds.
    #[inline]
    pub unsafe fn unravel_index_f(&self, index: usize) -> D {
        let mut index = index;
        let mut result = self.new_shape();
        match self.ndim() {
            0 => (),
            1 => {
                result[0] = index;
            },
            2 => {
                result[1] = index % self.shape()[1];
                index /= self.shape()[1];
                result[0] = index;
            },
            3 => {
                result[2] = index % self.shape()[2];
                index /= self.shape()[2];
                result[1] = index % self.shape()[1];
                index /= self.shape()[1];
                result[0] = index;
            },
            4 => {
                result[3] = index % self.shape()[3];
                index /= self.shape()[3];
                result[2] = index % self.shape()[2];
                index /= self.shape()[2];
                result[1] = index % self.shape()[1];
                index /= self.shape()[1];
                result[0] = index;
            },
            _ => {
                for i in 0..self.ndim() {
                    let dim = self.shape()[i];
                    result[i] = index % dim;
                    index /= dim;
                }
            },
        }
        result
    }
}

pub trait DimLayoutContigAPI: DimBaseAPI + DimShapeAPI + DimStrideAPI {
    /// Generate new layout by providing shape and offset; stride fits into
    /// c-contiguous.
    fn new_c_contig(&self, offset: Option<usize>) -> Layout<Self> {
        let shape = self.clone();
        let stride = shape.stride_c_contig();
        unsafe { Layout::new_unchecked(shape, stride, offset.unwrap_or(0)) }
    }

    /// Generate new layout by providing shape and offset; stride fits into
    /// f-contiguous.
    fn new_f_contig(&self, offset: Option<usize>) -> Layout<Self> {
        let shape = self.clone();
        let stride = shape.stride_f_contig();
        unsafe { Layout::new_unchecked(shape, stride, offset.unwrap_or(0)) }
    }

    /// Generate new layout by providing shape and offset; Whether c-contiguous
    /// or f-contiguous depends on cargo feature `c_prefer`.
    fn new_contig(&self, offset: Option<usize>) -> Layout<Self> {
        match TensorOrder::default() {
            TensorOrder::C => self.new_c_contig(offset),
            TensorOrder::F => self.new_f_contig(offset),
        }
    }

    /// Simplified function to generate c-contiguous layout. See also
    /// [DimLayoutContigAPI::new_c_contig].
    fn c(&self) -> Layout<Self> {
        self.new_c_contig(None)
    }

    /// Simplified function to generate f-contiguous layout. See also
    /// [DimLayoutContigAPI::new_f_contig].
    fn f(&self) -> Layout<Self> {
        self.new_f_contig(None)
    }
}

impl<const N: usize> DimLayoutContigAPI for Ix<N> {}
impl DimLayoutContigAPI for IxD {}

/* #endregion Layout */

/* #region Dimension Conversion */

impl<const N: usize> TryFrom<Layout<Ix<N>>> for Layout<IxD> {
    type Error = Error;

    fn try_from(layout: Layout<Ix<N>>) -> Result<Layout<IxD>> {
        let Layout { shape, stride, offset, size } = layout;
        let layout = Layout { shape: shape.to_vec(), stride: stride.to_vec(), offset, size };
        Ok(layout)
    }
}

impl<const N: usize> TryFrom<Layout<IxD>> for Layout<Ix<N>> {
    type Error = Error;

    fn try_from(layout: Layout<IxD>) -> Result<Layout<Ix<N>>> {
        let Layout { shape, stride, offset, size } = layout;
        Ok(Layout {
            shape: shape
                .try_into()
                .map_err(|_| Error::InvalidLayout(format!("Cannot convert IxD to Ix< {:} >", N)))?,
            stride: stride
                .try_into()
                .map_err(|_| Error::InvalidLayout(format!("Cannot convert IxD to Ix< {:} >", N)))?,
            offset,
            size,
        })
    }
}

pub trait LayoutConvertAPI<E>: Sized
where
    E: Into<Error>,
{
    fn into_dim<D2>(self) -> Result<Layout<D2>>
    where
        D2: DimBaseAPI,
        Self: TryInto<Layout<D2>, Error = E>,
    {
        self.try_into().map_err(Into::into)
    }
}

impl<D> LayoutConvertAPI<Error> for Layout<D> where D: DimAPI {}
impl<D> LayoutConvertAPI<Infallible> for Layout<D> where D: DimAPI {}

impl<const N: usize> From<Ix<N>> for Layout<Ix<N>> {
    fn from(shape: Ix<N>) -> Self {
        let stride = shape.stride_contig();
        Layout { shape, stride, offset: 0, size: shape.shape_size() }
    }
}

impl From<IxD> for Layout<IxD> {
    fn from(shape: IxD) -> Self {
        let size = shape.shape_size();
        let stride = shape.stride_contig();
        Layout { shape, stride, offset: 0, size }
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use std::panic::catch_unwind;

    use super::*;

    #[test]
    fn test_layout_new() {
        // a successful layout new
        let shape = [3, 2, 6];
        let stride = [3, -300, 15];
        let layout = Layout::new(shape, stride, 917);
        assert_eq!(layout.shape(), &[3, 2, 6]);
        assert_eq!(layout.stride(), &[3, -300, 15]);
        assert_eq!(layout.offset(), 917);
        assert_eq!(layout.ndim(), 3);
        // unsuccessful layout new (offset underflow)
        let shape = [3, 2, 6];
        let stride = [3, -300, 15];
        let r = catch_unwind(|| Layout::new(shape, stride, 0));
        assert!(r.is_err());
        // unsuccessful layout new (zero stride for non-0/1 shape)
        let shape = [3, 2, 6];
        let stride = [3, -300, 0];
        let r = catch_unwind(|| Layout::new(shape, stride, 1000));
        assert!(r.is_err());
        // unsuccessful layout new (stride too small)
        let shape = [3, 2, 6];
        let stride = [3, 4, 7];
        let r = catch_unwind(|| Layout::new(shape, stride, 1000));
        assert!(r.is_err());
        // successful layout new (zero dim)
        let shape = [];
        let stride = [];
        let r = catch_unwind(|| Layout::new(shape, stride, 1000));
        assert!(r.is_ok());
        // successful layout new (stride 0 for 1-shape)
        let shape = [3, 1, 5];
        let stride = [1, 0, 15];
        let r = catch_unwind(|| Layout::new(shape, stride, 1));
        assert!(r.is_ok());
        // successful layout new (stride 0 for 1-shape)
        let shape = [3, 1, 5];
        let stride = [1, 0, 15];
        let r = catch_unwind(|| Layout::new(shape, stride, 1));
        assert!(r.is_ok());
        // successful layout new (zero-size tensor)
        let shape = [3, 0, 5];
        let stride = [-1, -2, -3];
        let r = catch_unwind(|| Layout::new(shape, stride, 1));
        assert!(r.is_ok());
        // anyway, if one need custom layout, use new_unchecked
        let shape = [3, 2, 6];
        let stride = [3, -300, 0];
        let r = catch_unwind(|| unsafe { Layout::new_unchecked(shape, stride, 1000) });
        assert!(r.is_ok());
    }

    #[test]
    fn test_is_f_prefer() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [1, 10, 100], 0);
        assert!(layout.is_f_prefer());
        let layout = Layout::new(shape, [1, 3, 15], 0);
        assert!(layout.is_f_prefer());
        let layout = Layout::new(shape, [1, 3, -15], 1000);
        assert!(!layout.is_f_prefer());
        let layout = Layout::new(shape, [1, 21, 3], 0);
        assert!(!layout.is_f_prefer());
        let layout = Layout::new(shape, [35, 7, 1], 0);
        assert!(!layout.is_f_prefer());
        let layout = Layout::new(shape, [2, 6, 30], 0);
        assert!(!layout.is_f_prefer());
        // zero dimension
        let layout = Layout::new([], [], 0);
        assert!(layout.is_f_prefer());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0);
        assert!(layout.is_f_prefer());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [1, 1, 2], 0);
        assert!(layout.is_f_prefer());
    }

    #[test]
    fn test_is_c_prefer() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [100, 10, 1], 0);
        assert!(layout.is_c_prefer());
        let layout = Layout::new(shape, [35, 7, 1], 0);
        assert!(layout.is_c_prefer());
        let layout = Layout::new(shape, [-35, 7, 1], 1000);
        assert!(!layout.is_c_prefer());
        let layout = Layout::new(shape, [7, 21, 1], 0);
        assert!(!layout.is_c_prefer());
        let layout = Layout::new(shape, [1, 3, 15], 0);
        assert!(!layout.is_c_prefer());
        let layout = Layout::new(shape, [70, 14, 2], 0);
        assert!(!layout.is_c_prefer());
        // zero dimension
        let layout = Layout::new([], [], 0);
        assert!(layout.is_c_prefer());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0);
        assert!(layout.is_c_prefer());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [4, 1, 1], 0);
        assert!(layout.is_c_prefer());
    }

    #[test]
    fn test_is_f_contig() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [1, 3, 15], 0);
        assert!(layout.is_f_contig());
        let layout = Layout::new(shape, [1, 4, 20], 0);
        assert!(!layout.is_f_contig());
        // zero dimension
        let layout = Layout::new([], [], 0);
        assert!(layout.is_f_contig());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0);
        assert!(layout.is_f_contig());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [1, 1, 2], 0);
        assert!(layout.is_f_contig());
    }

    #[test]
    fn test_is_c_contig() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [35, 7, 1], 0);
        assert!(layout.is_c_contig());
        let layout = Layout::new(shape, [36, 7, 1], 0);
        assert!(!layout.is_c_contig());
        // zero dimension
        let layout = Layout::new([], [], 0);
        assert!(layout.is_c_contig());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0);
        assert!(layout.is_c_contig());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [4, 1, 1], 0);
        assert!(layout.is_c_contig());
    }

    #[test]
    fn test_index() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        assert_eq!(layout.index([0, 0, 0]), 782);
        assert_eq!(layout.index([2, 1, 4]), 668);
        assert_eq!(layout.index([1, -2, -3]), 830);
        // zero-dim
        let layout = Layout::new([], [], 10);
        assert_eq!(layout.index([]), 10);
    }

    #[test]
    fn test_bounds_index() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        // a.min() = 602, a.max() = 863
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        assert_eq!(layout.bounds_index().unwrap(), (602, 864));
        // situation that fails
        let layout = unsafe { Layout::new_unchecked([3, 2, 6], [3, -180, 15], 15) };
        assert!(layout.bounds_index().is_err());
        // zero-dim
        let layout = Layout::new([], [], 10);
        assert_eq!(layout.bounds_index().unwrap(), (10, 11));
    }

    #[test]
    fn test_transpose() {
        // general
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        let trans = layout.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(trans.shape(), &[6, 3, 2]);
        assert_eq!(trans.stride(), &[15, 3, -180]);
        // permute_dims is alias of transpose
        let trans = layout.permute_dims(&[2, 0, 1]).unwrap();
        assert_eq!(trans.shape(), &[6, 3, 2]);
        assert_eq!(trans.stride(), &[15, 3, -180]);
        // negative axis also allowed
        let trans = layout.transpose(&[-1, 0, 1]).unwrap();
        assert_eq!(trans.shape(), &[6, 3, 2]);
        assert_eq!(trans.stride(), &[15, 3, -180]);
        // repeated axis
        let trans = layout.transpose(&[-2, 0, 1]);
        assert!(trans.is_err());
        // non-valid dimension
        let trans = layout.transpose(&[1, 0]);
        assert!(trans.is_err());
        // zero-dim
        let layout = Layout::new([], [], 0);
        let trans = layout.transpose(&[]);
        assert!(trans.is_ok());
    }

    #[test]
    fn test_reverse_axes() {
        // general
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        let trans = layout.reverse_axes();
        assert_eq!(trans.shape(), &[6, 2, 3]);
        assert_eq!(trans.stride(), &[15, -180, 3]);
        // zero-dim
        let layout = Layout::new([], [], 782);
        let trans = layout.reverse_axes();
        assert_eq!(trans.shape(), &[]);
        assert_eq!(trans.stride(), &[]);
    }

    #[test]
    fn test_swapaxes() {
        // general
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        let trans = layout.swapaxes(-1, -2).unwrap();
        assert_eq!(trans.shape(), &[3, 6, 2]);
        assert_eq!(trans.stride(), &[3, 15, -180]);
        // same index is allowed
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
        let trans = layout.swapaxes(-1, -1).unwrap();
        assert_eq!(trans.shape(), &[3, 2, 6]);
        assert_eq!(trans.stride(), &[3, -180, 15]);
    }

    #[test]
    fn test_index_uncheck() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        unsafe {
            // fixed dim
            let layout = Layout::new([3, 2, 6], [3, -180, 15], 782);
            assert_eq!(layout.index_uncheck(&[0, 0, 0]), 782);
            assert_eq!(layout.index_uncheck(&[2, 1, 4]), 668);
            // dynamic dim
            let layout = Layout::new(vec![3, 2, 6], vec![3, -180, 15], 782);
            assert_eq!(layout.index_uncheck(&[0, 0, 0]), 782);
            assert_eq!(layout.index_uncheck(&[2, 1, 4]), 668);
            // zero-dim
            let layout = Layout::new([], [], 10);
            assert_eq!(layout.index_uncheck(&[]), 10);
        }
    }

    #[test]
    fn test_new_contig() {
        let layout = [3, 2, 6].c();
        assert_eq!(layout.shape(), &[3, 2, 6]);
        assert_eq!(layout.stride(), &[12, 6, 1]);
        let layout = [3, 2, 6].f();
        assert_eq!(layout.shape(), &[3, 2, 6]);
        assert_eq!(layout.stride(), &[1, 3, 6]);
        // following code generates contiguous layout
        // c/f-contig depends on cargo feature
        let layout: Layout<_> = [3, 2, 6].into();
        println!("{:?}", layout);
    }

    #[test]
    fn test_layout_cast() {
        let layout = [3, 2, 6].c();
        assert!(layout.clone().into_dim::<IxD>().is_ok());
        assert!(layout.clone().into_dim::<Ix3>().is_ok());
        let layout = vec![3, 2, 6].c();
        assert!(layout.clone().into_dim::<IxD>().is_ok());
        assert!(layout.clone().into_dim::<Ix3>().is_ok());
        assert!(layout.clone().into_dim::<Ix2>().is_err());
        // following usage is not valid
        let layout = unsafe { Layout::new_unchecked(vec![1, 2], vec![3, 4, 5], 1) };
        assert!(layout.clone().into_dim::<IxD>().is_ok());
        assert!(layout.clone().into_dim::<Ix2>().is_err());
        assert!(layout.clone().into_dim::<Ix3>().is_err());
    }
}
