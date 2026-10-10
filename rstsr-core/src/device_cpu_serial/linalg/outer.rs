use crate::prelude_dev::*;

impl<TA, TB, TC> DeviceOuterAPI<TA, TB, TC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TA: Mul<TB, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn outer(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<Ix2>,
        a: &Vec<TA>,
        la: &Layout<Ix1>,
        b: &Vec<TB>,
        lb: &Layout<Ix1>,
    ) -> Result<()> {
        outer_naive_cpu_serial(c, lc, a, la, b, lb)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_outer_row_major() {
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
        let b = rt::tensor_from_nested!([4.0f64, 5.], &device);
        let c = rt::outer(&a, &b);
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(format!("{c}"), "[[ 4 5]\n [ 8 10]\n [ 12 15]]");
    }
}
