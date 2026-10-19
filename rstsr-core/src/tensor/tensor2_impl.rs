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
    /// This function will not return any value if the layout is not row-major.
    pub fn ld_row(&self) -> Option<usize> {
        if !self.c_prefer() {
            // leading dimension is only defined if not c-prefer
            return None;
        } else if self.shape()[0] == 1 {
            // col-vector, leading dimension must be larger than dimension of col
            return Some(self.shape()[1]);
        } else {
            // usual definition that leading dimension is stride of row
            return Some(self.stride()[0] as usize);
        }
    }

    /// Leading dimension in column-major order case.
    ///
    /// This function will not return any value if the layout is not
    /// column-major.
    pub fn ld_col(&self) -> Option<usize> {
        if !self.f_prefer() {
            // leading dimension is only defined if not f-prefer
            return None;
        } else if self.shape()[1] == 1 {
            // row-vector, leading dimension must be larger than dimension of row
            return Some(self.shape()[0]);
        } else {
            // usual definition that leading dimension is stride of col
            return Some(self.stride()[1] as usize);
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
}
