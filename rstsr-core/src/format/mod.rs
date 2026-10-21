//! Tensor formatting: [`Display`](core::fmt::Display), [`Debug`],
//! [`LowerExp`](core::fmt::LowerExp) and [`UpperExp`](core::fmt::UpperExp)
//! implementations for tensor types.
//!
//! Formatting is dispatched to the device's [`DeviceFormatAPI`] implementation;
//! the tensor layer does not copy or otherwise materialize a host buffer. Wide
//! axes are truncated at the print limits ([`MIN_PRINT`] / [`MAX_PRINT`]).
//!
//! # Examples
//!
//! ```rust
//! # use rstsr::prelude::*;
//! # let mut device = DeviceCpu::default();
//! # device.set_default_order(RowMajor);
//! let a = rt::arange((6, &device)).into_shape([2, 3]);
//! println!("{a}");
//! // [[ 0 1 2]
//! //  [ 3 4 5]]
//!
//! let b = rt::asarray((vec![0.001, 1.0, 1000.0], &device));
//! println!("{:e}", b);
//! // [ 1e-3 1e0 1e3]
//!
//! let c = rt::arange((100, &device));
//! println!("{c}");
//! // [ 0 1 2 ... 97 98 99]
//!
//! // `Debug` renders through the same device implementation:
//! assert!(format!("{a:?}").contains("=== Debug Tensor Print ==="));
//! ```

pub mod format_tensor;

pub use format_tensor::{MAX_PRINT, MIN_PRINT};

use crate::prelude_dev::*;
use core::fmt::{LowerExp, UpperExp};
use core::sync::atomic::Ordering;

/// Device implementation of tensor formatting.
///
/// The device decides how to render its buffer and reads the print limits
/// ([`MIN_PRINT`], [`MAX_PRINT`]) itself. Devices backed by host memory
/// (`Raw = Vec<T>`) share the blanket implementation below.
pub trait DeviceFormatAPI<T, D>: DeviceAPI<T>
where
    D: DimAPI,
{
    fn format_display(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: Display;

    fn format_debug(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: Debug;

    fn format_lower_exp(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: LowerExp;

    fn format_upper_exp(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: UpperExp;
}

impl<T, D, B> DeviceFormatAPI<T, D> for B
where
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    fn format_display(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: Display,
    {
        let max_print = MAX_PRINT.load(Ordering::Relaxed);
        let min_print = MIN_PRINT.load(Ordering::Relaxed);
        format_tensor_cpu_serial(fmt, raw.as_slice(), layout, max_print, min_print, Display::fmt)
    }

    fn format_debug(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: Debug,
    {
        let max_print = MAX_PRINT.load(Ordering::Relaxed);
        let min_print = MIN_PRINT.load(Ordering::Relaxed);
        format_tensor_cpu_serial(fmt, raw.as_slice(), layout, max_print, min_print, Debug::fmt)
    }

    fn format_lower_exp(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: LowerExp,
    {
        let max_print = MAX_PRINT.load(Ordering::Relaxed);
        let min_print = MIN_PRINT.load(Ordering::Relaxed);
        format_tensor_cpu_serial(fmt, raw.as_slice(), layout, max_print, min_print, LowerExp::fmt)
    }

    fn format_upper_exp(
        &self,
        raw: &Self::Raw,
        layout: &Layout<D>,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result
    where
        T: UpperExp,
    {
        let max_print = MAX_PRINT.load(Ordering::Relaxed);
        let min_print = MIN_PRINT.load(Ordering::Relaxed);
        format_tensor_cpu_serial(fmt, raw.as_slice(), layout, max_print, min_print, UpperExp::fmt)
    }
}
