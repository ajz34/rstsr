use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GESVDDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_gesvd(
        order: FlagOrder,
        jobu: char,
        jobvt: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut T::Real,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
        superb: *mut T::Real,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GESVD_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(default = "true")]
    pub full_matrices: bool,
    #[builder(default = "true")]
    pub compute_uv: bool,
}

impl<'a, B, T> GESVD_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GESVDDriverAPI<T>,
{
    pub fn internal_run(
        self,
    ) -> Result<(Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>, Option<Tensor<T, B, Ix2>>, Tensor<T::Real, B, Ix1>)>
    {
        let Self { a, full_matrices, compute_uv } = self;
        let device = a.device().clone();

        let order = match (a.c_prefer(), a.f_prefer()) {
            (true, false) => RowMajor,
            (false, true) => ColMajor,
            (false, false) | (true, true) => device.default_order(),
        };

        // gesvd overwrites its input; clone-or-consume in the call order
        let mut a = overwritable_convert_with_order(a, order)?;
        let jobz = match (compute_uv, full_matrices) {
            (false, _) => 'N',
            (true, false) => 'S',
            (true, true) => 'A',
        };
        let [m, n] = *a.view().shape();
        let minmn = m.min(n);
        let lda = a.view().ld(order).unwrap();

        let ([u0, u1], [vt0, vt1]) = match jobz {
            'N' => ([0, 0], [0, 0]),
            'S' => ([m, minmn], [minmn, n]),
            'A' => ([m, m], [n, n]),
            _ => unreachable!(),
        };

        if minmn == 0 {
            // zero-size: LAPACK returns identity sub-blocks for U/VT
            let s = zeros_f(([0], &device))?.into_dim::<Ix1>();
            let superb = zeros_f(([0], &device))?.into_dim::<Ix1>();
            if !compute_uv {
                return Ok((s, None, None, superb));
            }
            let u_eye = eye_f((u0, u1, 0isize, order, &device))?.into_dim::<Ix2>();
            let vt_eye = eye_f((vt0, vt1, 0isize, order, &device))?.into_dim::<Ix2>();
            return Ok((s, Some(u_eye), Some(vt_eye), superb));
        }

        // factor buffers in the call order
        let mut u_f = match jobz {
            'N' => None,
            _ => Some(unsafe { empty_f(([u0, u1], order, &device))?.into_dim::<Ix2>() }),
        };
        let mut vt_f = match jobz {
            'N' => None,
            _ => Some(unsafe { empty_f(([vt0, vt1], order, &device))?.into_dim::<Ix2>() }),
        };
        let mut s = unsafe { empty_f(([minmn], &device))?.into_dim::<Ix1>() };
        let mut superb = unsafe { empty_f(([minmn.saturating_sub(1)], &device))?.into_dim::<Ix1>() };

        let ldu = u_f.as_ref().map_or(1, |u| u.view().ld(order).unwrap());
        let ldvt = vt_f.as_ref().map_or(1, |v| v.view().ld(order).unwrap());

        let info = unsafe {
            B::driver_gesvd(
                order,
                jobz,
                jobz,
                m,
                n,
                a.view_mut().as_mut_ptr(),
                lda,
                s.as_mut_ptr(),
                u_f.as_mut().map_or(std::ptr::null_mut(), |u| u.as_mut_ptr()),
                ldu,
                vt_f.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
                ldvt,
                superb.as_mut_ptr(),
            )
        };
        map_lapack_info(info, "Lapack GESVD")?;

        if compute_uv {
            Ok((s, u_f, vt_f, superb))
        } else {
            Ok((s, None, None, superb))
        }
    }

    pub fn run(
        self,
    ) -> Result<(Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>, Option<Tensor<T, B, Ix2>>, Tensor<T::Real, B, Ix1>)>
    {
        self.internal_run()
    }
}

pub type GESVD<'a, B, T> = GESVD_Builder<'a, B, T>;
pub type SGESVD<'a, B> = GESVD<'a, B, f32>;
pub type DGESVD<'a, B> = GESVD<'a, B, f64>;
pub type CGESVD<'a, B> = GESVD<'a, B, Complex<f32>>;
pub type ZGESVD<'a, B> = GESVD<'a, B, Complex<f64>>;
