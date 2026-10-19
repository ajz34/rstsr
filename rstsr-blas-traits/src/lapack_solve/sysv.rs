use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait SYSVDriverAPI<T, const HERMI: bool> {
    unsafe fn driver_sysv(
        order: FlagOrder,
        uplo: FlagUpLo,
        n: usize,
        nrhs: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
        b: *mut T,
        ldb: usize,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SYSV_<'a, 'b, B, T, const HERMI: bool>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into))]
    pub b: TensorReference<'b, T, B, Ix2>,

    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
}

impl<'a, 'b, B, T, const HERMI: bool> SYSV_<'a, 'b, B, T, HERMI>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + SYSVDriverAPI<T, HERMI>,
{
    pub fn internal_run(
        self,
    ) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>, TensorMutable2<'b, T, B>)> {
        let Self { a, b, uplo } = self;

        let device = a.device().clone();
        rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;
        let mut b = overwritable_convert(b)?;
        let order = get_output_order(&[], &[(b.c_prefer(), b.f_prefer())], device.default_order());
        let mut a = overwritable_convert_with_order(a, order)?;
        let uplo = uplo.unwrap_or_else(|| match device.default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        });

        let [n, nrhs] = *b.view().shape();
        rstsr_assert_eq!(a.view().shape(), &[n, n], InvalidLayout, "SYSV: A shape")?;

        // empty system or no right-hand side: nothing is factored or solved;
        // the driver never receives an empty B
        if n == 0 || nrhs == 0 {
            let ipiv = unsafe { empty_f(([0], &device))?.into_dim::<Ix1>() };
            return Ok((a.clone_to_mut(), ipiv, b.clone_to_mut()));
        }

        let lda = a.view().testify_ld(order, "A")?;
        let ldb = b.view().testify_ld(order, "B")?;
        let mut ipiv = unsafe { empty_f(([n], &device))?.into_dim::<Ix1>() };
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_b = b.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_sysv(order, uplo, n, nrhs, ptr_a, lda, ptr_ipiv, ptr_b, ldb) };
        map_lapack_info(info, "Lapack SYSV")?;

        Ok((a.clone_to_mut(), ipiv, b.clone_to_mut()))
    }

    pub fn run(self) -> Result<(TensorMutable2<'a, T, B>, Tensor<blas_int, B, Ix1>, TensorMutable2<'b, T, B>)> {
        self.internal_run()
    }
}

pub type SYSV<'a, 'b, B, T, const HERMI: bool> = SYSV_Builder<'a, 'b, B, T, HERMI>;
pub type SSYSV<'a, 'b, B> = SYSV<'a, 'b, B, f32, true>;
pub type DSYSV<'a, 'b, B> = SYSV<'a, 'b, B, f64, true>;
pub type CSYSV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f32>, false>;
pub type ZSYSV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f64>, false>;
pub type CHESV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f32>, true>;
pub type ZHESV<'a, 'b, B> = SYSV<'a, 'b, B, Complex<f64>, true>;
