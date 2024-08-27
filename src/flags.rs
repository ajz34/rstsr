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

/* #region TensorIterPolicy */

/// The policy of the tensor iterator.
///
/// # Default
///
/// Default iteration policy is [`TensorIterPolicy::K`].
///
/// You may change default value by [`TensorIterPolicy::change_default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorIterType {
    /// row-major order
    C,
    /// column-major order
    F,
    /// try contiguous, otherwise TensorOrder::Default()
    A,
    /// keep similar stride with input tensor
    K,
    /// greedy order (next() offset always larger then next_back())
    G,
}

impl_changeable_default!(TensorIterType, DEFAULT_TENSOR_ITER_POLICY, TensorIterType::K);

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
