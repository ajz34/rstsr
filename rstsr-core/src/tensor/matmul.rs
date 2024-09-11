//! Matrix-multiplication for tensor.

use core::ops::{Add, Mul, Rem};

use num::Num;

use crate::prelude_dev::*;

pub fn matmul<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorBase<RC, DC>,
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    // storage and lifetime
    RC: DataMutAPI<Data = Storage<TC, B>>,
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    let device = c.device().clone();
    let la = a.layout();
    let lb = b.layout();
    let lc = c.layout().clone();
    let sa = a.data().storage();
    let sb = b.data().storage();
    let sc = c.data_mut().storage_mut();
    device.matmul(sc, &lc, sa, la, sb, lb, alpha, beta)
}

impl<'a, 'b, RA, RB, TA, TB, TC, D, B> Rem<&TensorBase<RB, D>> for &TensorBase<RA, D>
where
    // storage and
    // lifetime
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    TA: 'a,
    TB: 'b,
    B: 'a + 'b,
    // dimension
    D: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    TC: Num,
    B: DeviceCreationNumAPI<TC>,
    B: DeviceMatMulAPI<TA, TB, TC, D, D, D>,
{
    type Output = Tensor<TC, D, B>;
    fn rem(self, rhs: &TensorBase<RB, D>) -> Tensor<TC, D, B> {
        let a = self.view();
        let b = rhs.view();
        let mut c = Tensor::<TC, D, B>::zeros(a.layout().clone(), a.device());
        matmul(&mut c, &a, &b, TC::one(), TC::zero()).unwrap();
        c
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::linspace_cpu(0.0, 15.0, 16).into_shape_assume_contig([4, 4]).unwrap();
        let b = Tensor::linspace_cpu(0.0, 15.0, 16).into_shape_assume_contig([4, 4]).unwrap();
        let mut c = Tensor::<f64, Ix2>::zeros_cpu([4, 4]);

        matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();
        println!("{c}");

        let d = &a % &b;
        println!("{d}");

        let a = Tensor::linspace_cpu(0.0, 14.0, 15);
        let b = Tensor::linspace_cpu(0.0, 14.0, 15);
        let mut c = Tensor::<f64, Ix0>::zeros_cpu([]);
        matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();
        println!("{c}");
    }
}
