//! Elementwise device-op trait impls for `DeviceCuda` (B5): arithmetic,
//! comparison, and unary float families — through the T0 kernel generators.
//!
//! Element bounds follow the house rule: `CudaDType` sub-traits live on the
//! device impls, tensor-tier functions stay bound-free.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaArith, CudaCmp, CudaDType, CudaFloat};
use crate::error::map_driver_result;
use crate::kernels_ops;
use crate::prelude_dev::*;

/// The unit layout `[1]` stride `[0]` offset 0 (broadcast-scalar shape),
/// constructed through the IxD conversion (works for any `D` via `into_dim`).
fn unit_layout<D: DimAPI>() -> Layout<D> {
    let l = Layout::new(vec![1_usize], vec![0_isize], 0).rstsr_unwrap();
    l.into_dim::<D>().rstsr_unwrap()
}

/// One-element device buffer holding `v` (scalar broadcast operand).
fn scalar_buffer<T: CudaDType>(device: &DeviceCuda, v: &T) -> Result<CudaBuffer<T>> {
    let mut buf = CudaBuffer::<T>::new(device.stream(), 1)?;
    let n = core::mem::size_of::<T>();
    let mut host = vec![0u8; n];
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*v) as *const u8, host.as_mut_ptr(), n);
    }
    // blocking H2D of one element (scalar staging is not on the hot path;
    // rt-level scalar ops get a cached scalar channel in a later round)
    let stream = buf.as_bytes().stream().clone();
    let ptr = buf.device_ptr_mut() as u64;
    unsafe {
        map_driver_result(cudarc::driver::result::memcpy_htod_async(
            ptr,
            &host,
            stream.cu_stream(),
        ))?;
        map_driver_result(stream.synchronize())?;
    }
    Ok(buf)
}

/// CUDA C call expression `name<suffix>(arg)`.
fn concat_expr(name: &str, suffix: &str, arg: &str) -> String {
    format!("{name}{suffix}({arg})")
}

// /* #region binary arithmetic */

#[duplicate_item(
    OpAPI   opchar;
    [OpAddAPI] ['+'];
    [OpSubAPI] ['-'];
    [OpMulAPI] ['*'];
    [OpDivAPI] ['/'];
)]
impl<TA, TB, TC, D> OpAPI<TA, TB, TC, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: CudaDType + CudaArith,
    TA: CudaDType + CudaArith,
    TB: CudaDType + CudaArith,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        kernels_ops::binary_arith::<TC, TA, TB>(self, c, lc, a, la, b, lb, opchar)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let b_buf = scalar_buffer(self, &b)?;
        kernels_ops::binary_arith::<TC, TA, TB>(self, c, lc, a, la, &b_buf, &unit_layout::<D>(), opchar)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let a_buf = scalar_buffer(self, &a)?;
        kernels_ops::binary_arith::<TC, TA, TB>(self, c, lc, &a_buf, &unit_layout::<D>(), b, lb, opchar)
    }
}

// /* #endregion */

// /* #region comparisons */

#[duplicate_item(
    OpAPI        opstr;
    [OpEqualAPI        ] ["=="];
    [OpNotEqualAPI     ] ["!="];
    [OpGreaterAPI      ] [">" ];
    [OpGreaterEqualAPI ] [">="];
    [OpLessAPI         ] ["<" ];
    [OpLessEqualAPI    ] ["<="];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
    TA: CudaDType + CudaCmp,
    TB: CudaDType + CudaCmp,
    D: DimAPI,
{
    type TOut = bool;

    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        kernels_ops::binary_cmp::<TA, TB>(self, c, lc, a, la, b, lb, opstr)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let b_buf = scalar_buffer(self, &b)?;
        kernels_ops::binary_cmp::<TA, TB>(self, c, lc, a, la, &b_buf, &unit_layout::<D>(), opstr)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let a_buf = scalar_buffer(self, &a)?;
        kernels_ops::binary_cmp::<TA, TB>(self, c, lc, &a_buf, &unit_layout::<D>(), b, lb, opstr)
    }
}

// /* #endregion */

// /* #region unary float functions */

/// `TOut = T` float unaries; CUDA C function names take an `f`-suffixed
/// form for `float` — the generator picks via the `{F}()` marker.
#[duplicate_item(
    OpAPI       fn_base  is_recip;
    [OpSinAPI     ] ["sin"]   [false];
    [OpCosAPI] ["cos"]         [false];
    [OpTanAPI] ["tan"]         [false];
    [OpAsinAPI] ["asin"]        [false];
    [OpAcosAPI] ["acos"]        [false];
    [OpAtanAPI] ["atan"]        [false];
    [OpSinhAPI] ["sinh"]        [false];
    [OpCoshAPI] ["cosh"]        [false];
    [OpTanhAPI] ["tanh"]        [false];
    [OpAsinhAPI] ["asinh"]       [false];
    [OpAcoshAPI] ["acosh"]       [false];
    [OpAtanhAPI] ["atanh"]       [false];
    [OpExpAPI] ["exp"]         [false];
    [OpExpm1API   ] ["expm1"]  [false];
    [OpLogAPI] ["log"]         [false];
    [OpLog1pAPI   ] ["log1p"]  [false];
    [OpLog2API    ] ["log2"]  [false];
    [OpLog10API   ] ["log10"]  [false];
    [OpSqrtAPI] ["sqrt"]        [false];
    [OpInvAPI     ] ["inv"]  [true];
    [OpReciprocalAPI] ["recip"] [true];
)]
impl<T, D> OpAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaFloat,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        // {F} resolves to the C float-suffix form per dtype at generation
        let cname = T::CNAME;
        let call = if cname == "float" {
            concat_expr(fn_base, "f", "b[ib]")
        } else if cname == "double" {
            concat_expr(fn_base, "", "b[ib]")
        } else {
            rstsr_raise!(UnImplemented, "unary float kernel dtype {cname} not covered")?
        };
        let expr: &str = if is_recip { "(1.0 / b[ib])" } else { &call };
        kernels_ops::unary_float::<T>(self, a, la, b, lb, expr)
    }

    fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<T>>, _la: &Layout<D>) -> Result<()> {
        // in-place variant: out-of-place kernel onto the same buffer is not
        // expressible without aliasing analysis; deferred to the ladder round
        rstsr_raise!(UnImplemented, "in-place unary on CUDA is deferred")?
    }
}

// /* #endregion */
