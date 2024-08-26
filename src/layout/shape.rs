use crate::prelude_dev::*;

pub trait DimShapeAPI: DimBaseAPI {
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
    /// let shape = [2, 3];
    /// assert_eq!(shape.shape_size(), 6);
    ///
    /// let shape = vec![];
    /// assert_eq!(shape.shape_size(), 1);
    /// ```
    fn shape_size(&self) -> usize;

    /// Stride for a f-contiguous tensor using this shape.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::prelude_dev::*;
    ///
    /// let stride = [2, 3, 5].stride_f_contig();
    /// assert_eq!(stride, [1, 2, 6]);
    /// ```
    fn stride_f_contig(&self) -> Self::Stride;

    /// Stride for a c-contiguous tensor using this shape.
    ///
    /// # Example
    ///
    /// ```
    /// use rstsr::prelude_dev::*;
    ///
    /// let stride = [2, 3, 5].stride_c_contig();
    /// assert_eq!(stride, [15, 5, 1]);
    /// ```
    fn stride_c_contig(&self) -> Self::Stride;

    /// Stride for contiguous tensor using this shape.
    ///
    /// # Cargo feature dependent
    ///
    /// Whether c-contiguous or f-contiguous will depends on cargo feature
    /// `c_prefer`.
    fn stride_contig(&self) -> Self::Stride;
}

impl<const N: usize> DimShapeAPI for Ix<N> {
    fn shape_size(&self) -> usize {
        self.iter().product()
    }

    fn stride_f_contig(&self) -> [isize; N] {
        let mut stride = [1; N];
        for i in 1..N {
            stride[i] = stride[i - 1] * self[i - 1].max(1) as isize;
        }
        stride
    }

    fn stride_c_contig(&self) -> [isize; N] {
        let mut stride = [1; N];
        if N == 0 {
            return stride;
        }
        for i in (0..N - 1).rev() {
            stride[i] = stride[i + 1] * self[i + 1].max(1) as isize;
        }
        stride
    }

    fn stride_contig(&self) -> [isize; N] {
        match Order::default() {
            Order::C => Self::stride_c_contig(self),
            Order::F => Self::stride_f_contig(self),
        }
    }
}

impl DimShapeAPI for IxD {
    fn shape_size(&self) -> usize {
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
            return stride;
        }
        for i in (0..self.len() - 1).rev() {
            stride[i] = stride[i + 1] * self[i + 1] as isize;
        }
        stride
    }

    fn stride_contig(&self) -> Vec<isize> {
        match Order::default() {
            Order::C => Self::stride_c_contig(self),
            Order::F => Self::stride_f_contig(self),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ndim() {
        // general test
        let shape = [2, 3];
        assert_eq!(shape.ndim(), 2);
        let shape = vec![2, 3];
        assert_eq!(shape.ndim(), 2);
        // empty dimension test
        let shape = [];
        assert_eq!(shape.ndim(), 0);
        let shape = vec![];
        assert_eq!(shape.ndim(), 0);
    }

    #[test]
    fn test_size() {
        // general test
        let shape = [2, 3];
        assert_eq!(shape.shape_size(), 6);
        let shape = vec![];
        assert_eq!(shape.shape_size(), 1);
        // empty dimension test
        let shape = [];
        assert_eq!(shape.shape_size(), 1);
        let shape = vec![];
        assert_eq!(shape.shape_size(), 1);
        // zero element test
        let shape = [1, 2, 0, 4];
        assert_eq!(shape.shape_size(), 0);
    }

    #[test]
    fn test_stride_f_contig() {
        // general test
        let stride = [2, 3, 5].stride_f_contig();
        assert_eq!(stride, [1, 2, 6]);
        // empty dimension test
        let stride = [].stride_f_contig();
        assert_eq!(stride, []);
        let stride = vec![].stride_f_contig();
        assert_eq!(stride, vec![]);
        // zero element test
        let stride = [1, 2, 0, 4].stride_f_contig();
        println!("{stride:?}");
    }

    #[test]
    fn test_stride_c_contig() {
        // general test
        let stride = [2, 3, 5].stride_c_contig();
        assert_eq!(stride, [15, 5, 1]);
        // empty dimension test
        let stride = [].stride_c_contig();
        assert_eq!(stride, []);
        let stride = vec![].stride_c_contig();
        assert_eq!(stride, vec![]);
        // zero element test
        let stride = [1, 2, 0, 4].stride_c_contig();
        println!("{stride:?}");
    }
}
