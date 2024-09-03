//! Flags for the crate.

/* #region changeable default */

pub trait ChangeableDefault {
    /// # Safety
    ///
    /// This function changes static mutable variable.
    /// It is better applying cargo feature instead of using this function.
    unsafe fn change_default(val: Self);
    fn get_default() -> Self;
}

macro_rules! impl_changeable_default {
    ($struct:ty, $val:ident, $default:expr) => {
        static mut $val: $struct = $default;

        impl ChangeableDefault for $struct {
            unsafe fn change_default(val: Self) {
                $val = val;
            }

            fn get_default() -> Self {
                return unsafe { $val };
            }
        }

        impl Default for $struct
        where
            Self: ChangeableDefault,
        {
            fn default() -> Self {
                <$struct>::get_default()
            }
        }
    };
}

/* #endregion */

/* #region TensorOrder */

/// The order of the tensor.
///
/// # Default
///
/// Default order depends on cargo feature `c_prefer`.
/// If `c_prefer` is set, then [`TensorOrder::C`] is applied as default;
/// otherwise [`TensorOrder::F`] is applied as default.
///
/// You may change default value by [`TensorOrder::change_default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorOrder {
    /// row-major order.
    C,
    /// column-major order.
    F,
}

impl_changeable_default!(
    TensorOrder,
    DEFAULT_TENSOR_ORDER,
    if cfg!(feature = "c_prefer") { TensorOrder::C } else { TensorOrder::F }
);

/* #endregion */

/* #region TensorIterOrder */

/// The policy of the tensor iterator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorIterOrder {
    /// Row-major order.
    ///
    /// - absolute safe for array iteration
    C,
    /// Column-major order.
    ///
    /// - absolute safe for array iteration
    F,
    /// Automatically choose row/col-major order.
    ///
    /// - try c/f-contig first (also see [`TensorIterOrder::B`]),
    /// - try c/f-prefer second (also see [`TensorIterOrder::C`],
    ///   [`TensorIterOrder::F`]),
    /// - otherwise [`TensorOrder::default()`], which is defined by crate
    ///   feature `c_prefer`.
    ///
    /// - safe for multi-array iteration like `get_iter(a, b)`
    /// - not safe for cases like `a.iter().zip(b.iter())`
    A,
    /// Greedy when possible (reorder layouts during iteration).
    ///
    /// - safe for multi-array iteration like `get_iter(a, b)`
    /// - not safe for cases like `a.iter().zip(b.iter())`
    /// - if it is used to create a new array, the stride of new array will be
    ///   in K order
    K,
    /// Greedy when possible (reset dimension to 1 if axis is broadcasted).
    ///
    /// - not safe for multi-array iteration like `get_iter(a, b)`
    /// - this is useful for inplace-assign broadcasted array.
    G,
    /// Sequential buffer.
    ///
    /// - not safe for multi-array iteration like `get_iter(a, b)`
    /// - this is useful for reshaping or all-contiguous cases.
    B,
}

/// TODO: this is to be further implemented.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorIterType {
    General,
    Tril,
    Triu,
    Diag,
    UnTril,
    UnTriu,
    UnDiag,
}

impl_changeable_default!(TensorIterType, DEFAULT_TENSOR_ITER_TYPE, TensorIterType::General);

/* #endregion */

/* #region TensorParallelPolicy */

/// The policy of the tensor parallel.
///
/// # Default
///
/// Default parallel policy is [`TensorParallelPolicy::ParallelCPU`].
///
/// You may change default value by [`TensorParallelPolicy::change_default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelPolicy {
    /// always serial
    Serial,
    /// parallel in most cases, serial when inside rayon threads or device is
    /// not CPU (e.g. inside rayon::par_iter)
    ParallelCPU,
    /// parallel in most cases, serial when inside rayon threads
    /// (e.g. inside rayon::par_iter)
    Parallel,
    /// always parallel
    ParallelForce,
}

impl_changeable_default!(
    TensorParallelPolicy,
    DEFAULT_TENSOR_PARALLEL_POLICY,
    TensorParallelPolicy::ParallelCPU
);

/* #endregion */
