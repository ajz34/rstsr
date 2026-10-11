use crate::prelude_dev::*;
use core::ops::{Mul, Sub};

impl<TA, TB, TC, D> DeviceCrossAPI<TA, TB, TC, D> for DeviceCpuSerial
where
    D: DimAPI,
    TA: Clone,
    TB: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Sub<TC, Output = TC>,
    Self: DeviceAPI<TA, Raw = Vec<TA>> + DeviceAPI<TB, Raw = Vec<TB>> + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn cross(
        &self,
        c: &mut Vec<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &Vec<TA>,
        la: &Layout<D>,
        b: &Vec<TB>,
        lb: &Layout<D>,
        axis: usize,
    ) -> Result<()> {
        cross_naive_cpu_serial(c, lc, a, la, b, lb, axis)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_cross_axis_last() {
        // the values are the same under either default order (only the display differs)
        let device = DeviceCpuSerial::default();
        let a = rt::tensor_from_nested!([1.0f64, 2., 3.], &device);
        let b = rt::tensor_from_nested!([4.0f64, 5., 6.], &device);
        let c = rt::cross(&a, &b, -1);
        assert_eq!(c.to_vec(), vec![-3.0, 6.0, -3.0]);
    }
}
