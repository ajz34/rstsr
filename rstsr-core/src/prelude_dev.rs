extern crate alloc;
pub use alloc::format;
pub use alloc::string::{String, ToString};
pub use alloc::vec;
pub use alloc::vec::Vec;

pub use core::fmt::{Debug, Display, Write};
pub use core::marker::PhantomData;

pub use itertools::{izip, Itertools};

pub use crate::error::{Error, Result};
pub use crate::flags::*;

pub use crate::layout::*;

pub use crate::storage::assignment::*;
pub use crate::storage::creation::*;
pub use crate::storage::device::*;
pub use crate::storage::matmul::*;
pub use crate::storage::operators::*;

pub use crate::cpu_backend::device::*;

#[allow(unused_imports)]
pub(crate) use crate::dev_utilities::*;

pub use crate::tensor::asarray::*;
pub use crate::tensor::creation::*;
pub use crate::tensor::data::*;
pub use crate::tensor::manuplication::*;
pub use crate::tensor::ownership_conversion::*;

pub use crate::{Tensor, TensorBase, TensorCow, TensorView, TensorViewMut};

pub use crate::slice;
pub use crate::{rstsr_assert, rstsr_assert_eq, rstsr_invalid, rstsr_pattern, rstsr_raise};