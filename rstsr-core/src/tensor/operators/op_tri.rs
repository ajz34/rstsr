//! Device-level triangular-matrix pack/unpack operations (`trf`-style, used
//! by LAPACK-backed solvers); backing [`TensorAny::pack_tri`] and
//! [`TensorAny::unpack_tri`].

use crate::prelude_dev::*;
use num::{Float as NumFloat, ToPrimitive};

/* #region pack_tri */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    B: DeviceAPI<T> + OpPackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Packs a triangle of the trailing square matrices into vectors (fallible).
    ///
    /// See also [`pack_tri`](TensorAny::pack_tri).
    pub fn pack_tri_f(&self, uplo: FlagUpLo) -> Result<Tensor<T, B, D::SmallerOne>> {
        // layouts manipulation
        let lb = self.layout().to_dim::<IxD>()?;

        // generate layout for output
        let default_order = self.device().default_order();
        let la_shape = match default_order {
            RowMajor => {
                // check last two dimensions are equal
                let (lb_rest, lb_inner) = lb.dim_split_at(-2)?;
                rstsr_assert_eq!(
                    lb_inner.shape()[0],
                    lb_inner.shape()[1],
                    InvalidLayout,
                    "Last two dimensions should be the same for pack_tri."
                )?;
                let n: usize = lb_inner.shape()[0];
                let n_tp = n * (n + 1) / 2;

                // layouts for output
                let mut la_shape = lb_rest.shape().to_vec();
                la_shape.push(n_tp);
                la_shape
            },
            ColMajor => {
                // check first two dimensions are equal
                let (lb_inner, lb_rest) = lb.dim_split_at(2)?;
                rstsr_assert_eq!(
                    lb_inner.shape()[0],
                    lb_inner.shape()[1],
                    InvalidLayout,
                    "First two dimensions should be the same for pack_tri."
                )?;
                let n: usize = lb_inner.shape()[0];
                let n_tp = n * (n + 1) / 2;

                // layouts for output
                let mut la_shape = vec![n_tp];
                la_shape.append(&mut lb_rest.shape().to_vec());
                la_shape
            },
        };

        let la = match (lb.c_prefer(), lb.f_prefer()) {
            (true, false) => la_shape.c(),
            (false, true) => la_shape.f(),
            _ => match self.device().default_order() {
                RowMajor => la_shape.c(),
                ColMajor => la_shape.f(),
            },
        };

        // device alloc and compute
        let device = self.device();
        let mut storage_a = device.uninit_impl(la.bounds_index()?.1)?;
        device.pack_tri(storage_a.raw_mut(), &la, self.raw(), &lb, uplo)?;
        // SAFETY: `tril`/`triu` op above wrote every element of the fresh storage.
        let storage_a = unsafe { B::assume_init_impl(storage_a)? };
        Tensor::new_f(storage_a, la.into_dim()?)
    }

    /// Packs a triangle of the trailing square matrices into vectors.
    ///
    /// The last two dimensions must be square (`n x n`); the triangle selected by `uplo` is
    /// written into a length-`n * (n + 1) / 2` vector, its entries visited in the device default
    /// order.
    ///
    /// <div class="warning">
    ///
    /// **Row/Column Major Notice**
    ///
    /// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of
    /// device.
    ///
    /// </div>
    ///
    /// The square block is the **trailing** two dimensions under [`RowMajor`] and the **leading**
    /// two under [`ColMajor`], and the packed vector is appended as a new **last** axis under
    /// [`RowMajor`] but prepended as a new **first** axis under [`ColMajor`]. For a 2-D input both
    /// orders give the same length-`n(n+1)/2` vector. See
    /// [`order_semantics`](crate::order_semantics).
    ///
    /// # Parameters
    ///
    /// - `uplo`: [`FlagUpLo`] — [`FlagUpLo::L`] packs the lower triangle, [`FlagUpLo::U`] the
    ///   upper.
    ///
    /// # Returns
    ///
    /// - [`Tensor<T, B, D::SmallerOne>`][Tensor] — a new owned tensor: the input shape with its
    ///   square block replaced by the packed vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::tensor_from_nested!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
    /// println!("{}", a.pack_tri(FlagUpLo::L));
    /// // [ 1 4 5 7 8 9]
    /// # assert_eq!(format!("{}", a.pack_tri(FlagUpLo::L)), "[ 1 4 5 7 8 9]");
    /// ```
    ///
    /// # Elaborated examples
    ///
    /// ## Difference between [`RowMajor`] and [`ColMajor`]
    ///
    /// For an input of rank at least three the packed axis is appended (row-major) or prepended
    /// (column-major):
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
    /// println!("{:?}", a.pack_tril().shape());
    /// // [2, 3]
    ///
    /// device.set_default_order(ColMajor);
    /// let a = rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device);
    /// println!("{:?}", a.pack_tril().shape());
    /// // [3, 2]
    /// # assert_eq!(rt::tensor_from_nested!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &device).pack_tril().shape(), &[3, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the last two dimensions (row-major) or first two dimensions (column-major) are
    ///   not equal.
    ///
    /// For a fallible version, use [`pack_tri_f`](TensorAny::pack_tri_f).
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`unpack_tri`](TensorAny::unpack_tri): the inverse operation.
    ///
    /// ## Variants of this function
    ///
    /// - [`pack_tri_f`](TensorAny::pack_tri_f): fallible version.
    /// - [`pack_tril`](TensorAny::pack_tril) / [`pack_triu`](TensorAny::pack_triu): the lower- and
    ///   upper-triangle convenience forms.
    pub fn pack_tri(&self, uplo: FlagUpLo) -> Tensor<T, B, D::SmallerOne> {
        self.pack_tri_f(uplo).rstsr_unwrap()
    }

    /// Packs the lower triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`](TensorAny::pack_tri).
    pub fn pack_tril_f(&self) -> Result<Tensor<T, B, D::SmallerOne>> {
        self.pack_tri_f(FlagUpLo::L)
    }

    /// Packs the lower triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`](TensorAny::pack_tri).
    pub fn pack_tril(&self) -> Tensor<T, B, D::SmallerOne> {
        self.pack_tril_f().rstsr_unwrap()
    }

    /// Packs the upper triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`](TensorAny::pack_tri).
    pub fn pack_triu_f(&self) -> Result<Tensor<T, B, D::SmallerOne>> {
        self.pack_tri_f(FlagUpLo::U)
    }

    /// Packs the upper triangle of the trailing square matrices into vectors.
    ///
    /// See also [`pack_tri`](TensorAny::pack_tri).
    pub fn pack_triu(&self) -> Tensor<T, B, D::SmallerOne> {
        self.pack_triu_f().rstsr_unwrap()
    }
}

/* #endregion */

/* #region unpack_tri */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    B: DeviceAPI<T> + OpUnpackTriAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Unpacks packed triangular vectors back into square matrices (fallible).
    ///
    /// See also [`unpack_tri`](TensorAny::unpack_tri).
    pub fn unpack_tri_f(&self, uplo: FlagUpLo, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        // layouts manipulation
        let lb = self.layout().to_dim::<IxD>()?;

        let default_order = self.device().default_order();
        let la_shape = match default_order {
            RowMajor => {
                // check last two dimensions are equal
                let (lb_rest, lb_inner) = lb.dim_split_at(-1)?;
                let n_tp: usize = lb_inner.shape()[0];
                let n: usize = NumFloat::floor(NumFloat::sqrt((2 * n_tp).to_f64().unwrap())).to_usize().unwrap();
                rstsr_assert_eq!(
                    n * (n + 1) / 2,
                    n_tp,
                    InvalidLayout,
                    "Last dimension should be triangular number for unpack_tri."
                )?;

                // layouts for output
                let mut la_shape = lb_rest.shape().to_vec();
                la_shape.append(&mut vec![n, n]);
                la_shape
            },
            ColMajor => {
                // check first two dimensions are equal
                let (lb_inner, lb_rest) = lb.dim_split_at(1)?;
                let n_tp: usize = lb_inner.shape()[0];
                let n: usize = NumFloat::floor(NumFloat::sqrt((2 * n_tp).to_f64().unwrap())).to_usize().unwrap();
                rstsr_assert_eq!(
                    n * (n + 1) / 2,
                    n_tp,
                    InvalidLayout,
                    "First dimension should be triangular number for unpack_tri."
                )?;

                // layouts for output
                let mut la_shape = vec![n, n];
                la_shape.append(&mut lb_rest.shape().to_vec());
                la_shape
            },
        };

        let la = match (lb.c_prefer(), lb.f_prefer()) {
            (true, false) => la_shape.c(),
            (false, true) => la_shape.f(),
            _ => match self.device().default_order() {
                RowMajor => la_shape.c(),
                ColMajor => la_shape.f(),
            },
        };

        // device alloc and compute
        let device = self.device();
        let mut storage_a = device.uninit_impl(la.bounds_index()?.1)?;
        device.unpack_tri(storage_a.raw_mut(), &la, self.raw(), &lb, uplo, symm)?;
        // SAFETY: `tril`/`triu` op above wrote every element of the fresh storage.
        let storage_a = unsafe { B::assume_init_impl(storage_a)? };
        Tensor::new_f(storage_a, la.into_dim()?)
    }

    /// Unpacks packed triangular vectors back into square matrices.
    ///
    /// The input carries vectors of length `n * (n + 1) / 2`; each is expanded back into an
    /// `n x n` matrix by writing the vector into the triangle selected by `uplo` and filling the
    /// other triangle according to `symm`.
    ///
    /// <div class="warning">
    ///
    /// **Row/Column Major Notice**
    ///
    /// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of
    /// device.
    ///
    /// </div>
    ///
    /// The packed vector is the **last** dimension under [`RowMajor`] and the **first** under
    /// [`ColMajor`], and the restored `n x n` block is inserted at the back (row-major) or at the
    /// front (column-major). For a 1-D input both orders give the same `n x n` matrix. See
    /// [`order_semantics`](crate::order_semantics).
    ///
    /// # Parameters
    ///
    /// - `uplo`: [`FlagUpLo`] — the triangle the vector is written into ([`FlagUpLo::L`] or
    ///   [`FlagUpLo::U`]).
    /// - `symm`: [`FlagSymm`] — how the opposite triangle is derived from the packed one: `Sy`
    ///   mirrors it (symmetric), `He` mirrors its complex conjugate (Hermitian), `Ay` negates it
    ///   and zeroes the diagonal (anti-symmetric), `Ah` negates the conjugate and zeroes the
    ///   diagonal (anti-Hermitian); `N` writes the triangle selected by `uplo` only.
    ///
    /// # Returns
    ///
    /// - [`Tensor<T, B, D::LargerOne>`][Tensor] — a new owned tensor: the input shape with its
    ///   packed vector replaced by the `n x n` matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let packed = rt::tensor_from_nested!([1.0, 4.0, 5.0, 7.0, 8.0, 9.0], &device);
    /// println!("{}", packed.unpack_tri(FlagUpLo::L, FlagSymm::Sy));
    /// // [[ 1 4 7]
    /// //  [ 4 5 8]
    /// //  [ 7 8 9]]
    /// # assert_eq!(format!("{}", packed.unpack_tri(FlagUpLo::L, FlagSymm::Sy)), "[[ 1 4 7]\n [ 4 5 8]\n [ 7 8 9]]");
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the last dimension (row-major) or first dimension (column-major) is not a
    ///   triangular number of the form `n * (n + 1) / 2`.
    ///
    /// For a fallible version, use [`unpack_tri_f`](TensorAny::unpack_tri_f).
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`pack_tri`](TensorAny::pack_tri): the inverse operation.
    ///
    /// ## Variants of this function
    ///
    /// - [`unpack_tri_f`](TensorAny::unpack_tri_f): fallible version.
    /// - [`unpack_tril`](TensorAny::unpack_tril) / [`unpack_triu`](TensorAny::unpack_triu): the
    ///   lower- and upper-triangle convenience forms.
    pub fn unpack_tri(&self, uplo: FlagUpLo, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        self.unpack_tri_f(uplo, symm).rstsr_unwrap()
    }

    /// Unpacks packed lower-triangular vectors back into square matrices.
    ///
    /// See also [`unpack_tri`](TensorAny::unpack_tri).
    pub fn unpack_tril(&self, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        self.unpack_tri_f(FlagUpLo::L, symm).rstsr_unwrap()
    }

    /// Unpacks packed upper-triangular vectors back into square matrices.
    ///
    /// See also [`unpack_tri`](TensorAny::unpack_tri).
    pub fn unpack_triu(&self, symm: FlagSymm) -> Tensor<T, B, D::LargerOne> {
        self.unpack_tri_f(FlagUpLo::U, symm).rstsr_unwrap()
    }

    /// Unpacks packed lower-triangular vectors back into square matrices (fallible).
    ///
    /// See also [`unpack_tri`](TensorAny::unpack_tri).
    pub fn unpack_tril_f(&self, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        self.unpack_tri_f(FlagUpLo::L, symm)
    }

    /// Unpacks packed upper-triangular vectors back into square matrices (fallible).
    ///
    /// See also [`unpack_tri`](TensorAny::unpack_tri).
    pub fn unpack_triu_f(&self, symm: FlagSymm) -> Result<Tensor<T, B, D::LargerOne>> {
        self.unpack_tri_f(FlagUpLo::U, symm)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_tri() {
        #[cfg(not(feature = "col_major"))]
        {
            let a = {
                let a = arange((48., &DeviceCpuSerial::default()));
                let storage_a = a.into_raw_parts().0;
                Tensor::new(storage_a, [3, 4, 4].f())
            };
            let a_triu = a.pack_tril();
            println!("{a_triu:?}");
            println!("{:?}", a.slice(0));
            println!("{:?}", a_triu.slice(0).to_vec());
            assert_eq!(a_triu.slice(1).to_vec(), [1., 4., 16., 7., 19., 31., 10., 22., 34., 46.]);

            let b = a_triu.unpack_tril(FlagSymm::Sy);
            println!("{b:?}");
            assert_eq!(b.slice((0, 1)).to_vec(), [3., 15., 18., 21.]);
        }
        #[cfg(feature = "col_major")]
        {
            let a = {
                let a = arange((48., &DeviceCpuSerial::default()));
                let storage_a = a.into_raw_parts().0;
                Tensor::new(storage_a, [4, 4, 3].c())
            };
            let a_triu = a.pack_triu();
            println!("{a_triu:?}");
            println!("{:?}", a.slice((.., 0)));
            println!("{:?}", a_triu.slice((.., 0)).to_vec());
            assert_eq!(a_triu.slice((.., 1)).to_vec(), [1., 4., 16., 7., 19., 31., 10., 22., 34., 46.]);

            let b = a_triu.unpack_triu(FlagSymm::Sy);
            println!("{b:?}");
            assert_eq!(b.slice((.., 1, 0)).to_vec(), [3., 15., 18., 21.]);
        }
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_pack_tril_compiles() {
        #[cfg(not(feature = "col_major"))]
        {
            use num::complex::c64;
            let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256)).into_layout([4, 64, 256, 256].f());
            let a_tril = a.pack_tril();
            println!("{a_tril:20.5}");
            let b = a_tril.unpack_tril(FlagSymm::Ah);
            println!("{b:20.5}");
        }
        #[cfg(feature = "col_major")]
        {
            use num::complex::c64;
            let a = linspace((c64(-2.0, 1.5), c64(1.7, -2.3), 256 * 256 * 256)).into_layout([256, 256, 64, 4].c());
            let a_tril = a.pack_tril();
            println!("{a_tril:20.5}");
            let b = a_tril.unpack_tril(FlagSymm::Ah);
            println!("{b:20.5}");
        }
    }

    #[test]
    fn test_correctness() {
        let a = {
            let a = arange((16., &DeviceCpuSerial::default()));
            let storage_a = a.into_raw_parts().0;
            Tensor::new(storage_a, [4, 4].c())
        };
        println!("{a:}");

        let a_tril1 = a.pack_tril();
        println!("{a_tril1:}");

        let a = a.to_contig(FlagOrder::F);
        let a_tril2 = a.pack_tril();
        println!("{a_tril2:}");

        assert!((&a_tril1 - &a_tril2).l2_norm_all() < 1e-6);
    }
}
