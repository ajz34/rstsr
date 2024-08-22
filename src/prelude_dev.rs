extern crate alloc;
pub use alloc::format;
pub use alloc::string::{String, ToString};
pub use alloc::vec;
pub use alloc::vec::Vec;

pub use core::fmt::{Debug, Display, Write};

pub use crate::error::{Error, Result};
pub use crate::layout::*;

pub use crate::storage::creation::*;
pub use crate::storage::data::*;
pub use crate::storage::device::*;

pub use crate::tensor::creation::*;
pub use crate::tensor::layout_manuplication::*;

pub use crate::{Tensor, TensorBase};

pub use crate::slice;
pub use crate::{rstsr_assert, rstsr_assert_eq, rstsr_invalid, rstsr_pattern, rstsr_raise};
