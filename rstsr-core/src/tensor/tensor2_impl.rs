//! Matrix-oriented helpers on 2-D tensors: shape accessors ([`TensorBase::nrow`],
//! [`TensorBase::ncol`]) and leading-dimension queries ([`TensorBase::ld`],
//! [`TensorBase::ld_row`], [`TensorBase::ld_col`], [`TensorBase::testify_ld`])
//! used by BLAS-style interfaces.

use crate::prelude_dev::*;

impl<S> TensorBase<S, Ix2> {
    /// Number of rows (first dimension of the 2-D shape).
    pub fn nrow(&self) -> usize {
        self.shape()[0]
    }

    /// Number of columns (second dimension of the 2-D shape).
    pub fn ncol(&self) -> usize {
        self.shape()[1]
    }

    /// Leading dimension in row-major order case.
    ///
    /// `None` means the layout has no usable leading dimension: it is not
    /// C-prefer, or its row stride is smaller than the row length
    /// `max(1, ncol)` (such a layout overlaps elements and is not a valid
    /// BLAS operand).
    ///
    /// An axis of length 0 or 1 carries no stride information, so a row
    /// vector or an empty matrix reports the minimal legal leading dimension
    /// `max(1, ncol)`.
    pub fn ld_row(&self) -> Option<usize> {
        if !self.c_prefer() {
            return None;
        }
        let ncol = self.shape()[1].max(1);
        if self.shape()[0] <= 1 {
            // row vector: leading dimension must be larger than row length
            return Some(ncol);
        }
        // usual definition: the leading dimension is the stride of a row
        let stride = self.stride()[0];
        if stride < ncol as isize {
            None
        } else {
            Some(stride as usize)
        }
    }

    /// Leading dimension in column-major order case.
    ///
    /// `None` means the layout has no usable leading dimension: it is not
    /// F-prefer, or its column stride is smaller than the column length
    /// `max(1, nrow)` (such a layout overlaps elements and is not a valid
    /// BLAS operand).
    ///
    /// An axis of length 0 or 1 carries no stride information, so a column
    /// vector or an empty matrix reports the minimal legal leading dimension
    /// `max(1, nrow)`.
    pub fn ld_col(&self) -> Option<usize> {
        if !self.f_prefer() {
            return None;
        }
        let nrow = self.shape()[0].max(1);
        if self.shape()[1] <= 1 {
            // column vector: leading dimension must be larger than column length
            return Some(nrow);
        }
        // usual definition: the leading dimension is the stride of a column
        let stride = self.stride()[1];
        if stride < nrow as isize {
            None
        } else {
            Some(stride as usize)
        }
    }

    /// Leading dimension under the given order; see [`TensorBase::ld_row`] and
    /// [`TensorBase::ld_col`].
    pub fn ld(&self, order: FlagOrder) -> Option<usize> {
        match order {
            ColMajor => self.ld_col(),
            RowMajor => self.ld_row(),
        }
    }

    /// Leading dimension under `order`, or an `InvalidLayout` error when the
    /// layout has no legal one.
    ///
    /// This is the fallible form of [`TensorBase::ld`], used by BLAS/LAPACK
    /// wrappers: `None` (wrong order, or an empty/degenerate layout) becomes an
    /// error that names the offending operand and layout.
    pub fn testify_ld(&self, order: FlagOrder, operand: &str) -> Result<usize> {
        self.ld(order).ok_or_else(|| {
            rstsr_error!(InvalidLayout, "no legal leading dimension for operand {operand}: {:?}", self.layout())
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        let l = Layout::new([6, 1], [100, 10], 0).unwrap();
        println!("{:?}", l.f_prefer());
        println!("{:?}", l.c_prefer());
        println!("{l:?}");
    }

    #[test]
    fn testify_ld_reports_operand_and_layout() {
        let device = DeviceCpuSerial::default();
        let a: Tensor<f64, DeviceCpuSerial, Ix2> = zeros_f(([2usize, 3].c(), &device)).unwrap().into_dim::<Ix2>();
        let msg = format!("{:?}", a.testify_ld(ColMajor, "A").unwrap_err());
        assert!(msg.contains("operand A"), "message = {msg}");
        assert!(msg.contains("shape: [2, 3]"), "message = {msg}");
    }

    /// Build a tensor with an arbitrary layout, bypassing the overlap check of
    /// `Tensor::new_f`; these tensors are only queried for layout properties.
    fn tensor_with(shape: [usize; 2], stride: [isize; 2]) -> Tensor<f64, DeviceCpuSerial, Ix2> {
        let storage = Storage::new(DataOwned { raw: vec![0.0; 64] }, DeviceCpuSerial::default());
        let layout = unsafe { Layout::new_unchecked(shape, stride, 0) };
        TensorBase { storage, layout }
    }

    #[test]
    fn ld_empty_shapes() {
        // [0, 0]: the minimal legal leading dimension in both orders
        let a = tensor_with([0, 0], [1, 1]);
        assert_eq!(a.ld_row(), Some(1));
        assert_eq!(a.ld_col(), Some(1));

        // [m, 0]: ld_row only has to hold an empty row; ld_col must hold m rows
        let a = tensor_with([18, 0], [1, 1]);
        assert_eq!(a.ld_row(), Some(1));
        assert_eq!(a.ld_col(), Some(18));

        // [0, n]: symmetric
        let a = tensor_with([0, 18], [1, 1]);
        assert_eq!(a.ld_row(), Some(18));
        assert_eq!(a.ld_col(), Some(1));
    }

    #[test]
    fn ld_vector_shapes_ignore_stride() {
        // [m, 1]: a single column carries no column-stride information
        let a = tensor_with([18, 1], [1, 25]);
        assert_eq!(a.ld_row(), Some(1));
        assert_eq!(a.ld_col(), Some(18));

        // [1, n]: a single row carries no row-stride information
        let a = tensor_with([1, 18], [25, 1]);
        assert_eq!(a.ld_row(), Some(18));
        assert_eq!(a.ld_col(), Some(1));
    }

    #[test]
    fn ld_rejects_illegal_layouts() {
        // wrong preferred order
        let a = tensor_with([2, 3], [3, 1]);
        assert_eq!(a.ld_row(), Some(3));
        assert_eq!(a.ld_col(), None);

        // overlapping strides: no legal leading dimension
        let a = tensor_with([3, 4], [1, 2]);
        assert_eq!(a.ld_col(), None);

        let a = tensor_with([4, 3], [2, 1]);
        assert_eq!(a.ld_row(), None);
    }
}
