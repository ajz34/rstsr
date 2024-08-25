extern crate alloc;
pub use alloc::format;
pub use alloc::string::{String, ToString};
pub use alloc::vec;
pub use alloc::vec::Vec;

pub use core::fmt::{Debug, Display, Write};
pub use core::marker::PhantomData;

pub use itertools::izip;

pub use crate::error::{Error, Result};
pub use crate::layout::*;

pub use crate::storage::creation::*;
pub use crate::storage::device::*;
pub use crate::storage::op_binary::*;

pub use crate::cpu_backend::device::*;

pub use crate::tensor::data::*;

pub use crate::{Order, Tensor, TensorBase};

pub use crate::slice;
pub use crate::{rstsr_assert, rstsr_assert_eq, rstsr_invalid, rstsr_pattern, rstsr_raise};
