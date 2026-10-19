use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GETRIDriverAPI<T> {
    unsafe fn driver_getri(order: FlagOrder, n: usize, a: *mut T, lda: usize, ipiv: *mut blas_int) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GETRI_<'a, 'ipiv, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T> + DeviceAPI<blas_int>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    pub ipiv: TensorView<'ipiv, blas_int, B, Ix1>,
}

impl<'a, B, T> GETRI_<'a, '_, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GETRIDriverAPI<T>,
{
    pub fn internal_run(self) -> Result<TensorMutable2<'a, T, B>> {
        let Self { a, ipiv } = self;

        let device = a.device().clone();
        let order = get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order());
        let mut a = overwritable_convert_with_order(a, order)?;
        let mut ipiv = ipiv.into_contig_f(ColMajor)?;

        // perform check
        rstsr_assert_eq!(a.view().nrow(), a.view().ncol(), InvalidLayout, "Lapack GETRI: A must be square")?;

        let n = a.view().nrow();
        rstsr_assert!(ipiv.size() >= n, InvalidLayout, "Lapack GETRI: ipiv must have at least n elements")?;

        // empty A: nothing to invert
        if n == 0 {
            return Ok(a.clone_to_mut());
        }

        // rust is 0-indexed
        ipiv += 1;

        let lda = a.view().testify_ld(order, "A")?;
        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_ipiv = ipiv.as_mut_ptr();

        // run driver
        let info = unsafe { B::driver_getri(order, n, ptr_a, lda, ptr_ipiv) };
        map_lapack_info(info, "Lapack GETRI")?;

        Ok(a.clone_to_mut())
    }

    pub fn run(self) -> Result<TensorMutable2<'a, T, B>> {
        self.internal_run()
    }
}

pub type GETRI<'a, 'ipiv, B, T> = GETRI_Builder<'a, 'ipiv, B, T>;
pub type SGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, f32>;
pub type DGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, f64>;
pub type CGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, Complex<f32>>;
pub type ZGETRI<'a, 'ipiv, B> = GETRI<'a, 'ipiv, B, Complex<f64>>;
