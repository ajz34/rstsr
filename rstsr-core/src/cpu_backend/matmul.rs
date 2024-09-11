//! Matrix multiplication for CPU backend.
//!
//! **This implementation is not optimized!**

use core::ops::{Add, Mul};

use crate::prelude_dev::*;

impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceGEMMAPI<TA, TB, TC>,
    Self: DeviceGEMVAPI<TA, TB, TC>,
    Self: DeviceInnerDotAPI<TA, TB, TC>,
{
    fn matmul(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<DC>,
        a: &Storage<TA, Self>,
        la: &Layout<DA>,
        b: &Storage<TB, Self>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        match (la.ndim(), lb.ndim(), lc.ndim()) {
            (2, 2, 2) => {
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix2>().unwrap();
                self.gemm(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (2, 1, 1) => {
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let lc = &lc.clone().into_dim::<Ix1>().unwrap();
                self.gemv(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (1, 2, 1) => {
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix1>().unwrap();
                self.gevm(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (1, 1, 0) => {
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let lc = &lc.clone().into_dim::<Ix0>().unwrap();
                self.inner_dot(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (1, 2.., _) => {
                // rule 3: | `        K` | `..., K, N` | `   ..., N` |
                rstsr_assert_eq!(lb.ndim(), lc.ndim() + 1, InvalidLayout)?;
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
                let (lc_rest, lc_matmul) = lc.dim_split_at(-1)?;
                let lb_matmul = &mut lb_matmul.into_dim::<Ix2>()?;
                let lc_matmul = &mut lc_matmul.into_dim::<Ix1>()?;
                let l_rest = translate_to_col_major(&[&lb_rest, &lc_rest], TensorIterOrder::K)?;
                let (lb_rest, lc_rest) = (&l_rest[0], &l_rest[1]);
                let itb_rest = IterLayoutColMajor::new(lb_rest)?;
                let itc_rest = IterLayoutColMajor::new(lc_rest)?;
                for (ib_rest, ic_rest) in izip!(itb_rest, itc_rest) {
                    unsafe { lb_matmul.set_offset(ib_rest) };
                    unsafe { lc_matmul.set_offset(ic_rest) };
                    self.gevm(c, lc_matmul, a, la, b, lb_matmul, alpha.clone(), beta.clone())?;
                }
            },
            _ => {
                rstsr_raise!(
                    UnImplemented,
                    "Invalid ndim for matmul: {}, {}, {}",
                    la.ndim(),
                    lb.ndim(),
                    lc.ndim()
                )?;
            },
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceGEMMAPI<TA, TB, TC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn gemm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix2>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sc = lc.shape();
        let sa = la.shape();
        let sb = lb.shape();
        debug_assert_eq!(sc[0], sa[0]);
        debug_assert_eq!(sa[1], sb[0]);
        debug_assert_eq!(sc[1], sb[1]);
        let (m, n, k) = (sc[0], sc[1], sa[1]);

        // naive iteration: assuming c-prefer
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            for i_n in 0..m {
                for i_m in 0..n {
                    let idx_c = lc.index_uncheck(&[i_m, i_n]);
                    vc[idx_c] = beta.clone() * vc[idx_c].clone();
                }
                for i_k in 0..k {
                    let idx_b = lb.index_uncheck(&[i_k, i_n]);
                    for i_m in 0..n {
                        let idx_c = lc.index_uncheck(&[i_m, i_n]);
                        let idx_a = la.index_uncheck(&[i_m, i_k]);
                        vc[idx_c] = alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone())
                            + vc[idx_c].clone();
                    }
                }
            }
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceGEMVAPI<TA, TB, TC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn gemv(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix1>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sc = lc.shape();
        let sa = la.shape();
        let sb = lb.shape();
        debug_assert_eq!(sc[0], sa[0]);
        debug_assert_eq!(sa[1], sb[0]);
        let (n, k) = (sa[0], sa[1]);

        // naive iteration: assuming c-prefer
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            for i_n in 0..n {
                let idx_c = lc.index_uncheck(&[i_n]);
                vc[idx_c] = beta.clone() * vc[idx_c].clone();
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_n, i_k]);
                    let idx_b = lb.index_uncheck(&[i_k]);
                    vc[idx_c] =
                        alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone()) + vc[idx_c].clone();
                }
            }
        }
        return Ok(());
    }

    fn gevm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix1>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix1>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sc = lc.shape();
        let sa = la.shape();
        let sb = lb.shape();
        debug_assert_eq!(sc[0], sb[1]);
        debug_assert_eq!(sa[0], sb[0]);
        let (n, k) = (sb[1], sb[0]);

        // naive iteration: assuming c-prefer
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            for i_n in 0..n {
                let idx_c = lc.index_uncheck(&[i_n]);
                vc[idx_c] = beta.clone() * vc[idx_c].clone();
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_k]);
                    let idx_b = lb.index_uncheck(&[i_k, i_n]);
                    vc[idx_c] =
                        alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone()) + vc[idx_c].clone();
                }
            }
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceInnerDotAPI<TA, TB, TC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn inner_dot(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix0>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix1>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sa = la.shape();
        let sb = lb.shape();
        rstsr_assert_eq!(sa[0], sb[0], InvalidLayout)?;
        let n = sa[0];

        // naive iteration
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            let idx_c = lc.index_uncheck(&[]);
            let mut sum = beta * vc[idx_c].clone();
            for i in 0..n {
                let idx_a = la.index_uncheck(&[i]);
                let idx_b = lb.index_uncheck(&[i]);
                sum = sum + alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone());
            }
            vc[0] = sum;
        }
        return Ok(());
    }
}
