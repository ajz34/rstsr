use crate::prelude_dev::*;
use core::fmt::{Debug, Display, LowerExp, UpperExp};
use core::sync::atomic::AtomicUsize;

pub static MIN_PRINT: AtomicUsize = AtomicUsize::new(3);
pub static MAX_PRINT: AtomicUsize = AtomicUsize::new(8);

/* #region display/lowerexp/upperexp print */

#[duplicate_item(
    Trait Method;
    [Display] [format_display];
    [LowerExp] [format_lower_exp];
    [UpperExp] [format_upper_exp];
)]
impl<R, T, B, D> Trait for TensorAny<R, T, B, D>
where
    T: Trait,
    B: DeviceFormatAPI<T, D>,
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.device().Method(self.raw(), self.layout(), f)
    }
}

/* #endregion */

/* #region debug print */

pub fn print_vec_with_layout_debug<T, D>(
    fmt: &mut core::fmt::Formatter<'_>,
    vec: &[T],
    layout: &Layout<D>,
    max_print: usize,
    min_print: usize,
) -> core::fmt::Result
where
    T: Debug,
    D: DimAPI,
{
    format_tensor_cpu_serial(fmt, vec, layout, max_print, min_print, Debug::fmt)
}

/// Debug print for TensorAny (more expansive than Display print).
///
/// The buffer is rendered through the device's [`DeviceFormatAPI`]
/// implementation. Host-backed devices read it in place without copying; a device
/// implementation that copies internally (e.g. a GPU backend) may still pay that cost.
impl<R, T, B, D> Debug for TensorAny<R, T, B, D>
where
    T: Debug,
    B: DeviceFormatAPI<T, D> + Debug,
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "\n=== Debug Tensor Print ===")?;
        self.device().format_debug(self.raw(), self.layout(), f)?;
        writeln!(f)?;
        Debug::fmt(&self.device(), f)?;
        writeln!(f)?;
        Debug::fmt(&self.layout(), f)?;
        writeln!(f)?;
        let self_type = core::any::type_name::<Self>();
        writeln!(f, "Type: {self_type}")?;
        writeln!(f, "==========================")
    }
}

/* #endregion */

#[cfg(test)]
mod playground {
    use super::*;
    use core::sync::atomic::Ordering;

    #[derive(Debug)]
    struct VL<T, D>(Vec<T>, Layout<D>)
    where
        T: Clone + Display,
        D: DimAPI;

    impl<T, D> Display for VL<T, D>
    where
        T: Clone + Display,
        D: DimAPI,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let vec = self.0.as_slice();
            let layout = &self.1;
            let max_print = MAX_PRINT.load(Ordering::Relaxed);
            let min_print = MIN_PRINT.load(Ordering::Relaxed);
            format_tensor_cpu_serial(f, vec, layout, max_print, min_print, Display::fmt)
        }
    }

    #[test]
    fn playground() {
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [].into();
        println!("{:?}", VL(vec, layout));

        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [2, 0, 4].into();
        println!("{:}", VL(vec, layout));

        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [2, 4, 0].into();
        println!("{:}", VL(vec, layout));

        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<Ix2> = [3, 5].into();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[4:13].reshape(3, 3)
        c = b.T[::2, ::-1]
        */
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new([2, 3], [2, -4], 10).unwrap();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[2:14].reshape(3, 4)
        c = b.T[:, ::-1]
        */
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new([4, 3], [1, -4], 10).unwrap();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[2:14].reshape(3, 4)
        c = b.T[:, ::-1]
        */
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new([4, 3], [1, -4], 10).unwrap();
        println!("{:}", VL(vec, layout));

        let vec = (0..1800).collect::<Vec<usize>>();
        let layout = Layout::<Ix3>::new([15, 12, 10], [1, 150, 15], 0).unwrap();
        println!("{:4}", VL(vec, layout));
    }

    #[test]
    fn playground_lowerexp() {
        use crate::prelude::*;
        let vec = vec![0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.];
        let arr = rt::asarray((vec, [2, 5]));
        println!("{:14.5e}", arr);
    }
}
