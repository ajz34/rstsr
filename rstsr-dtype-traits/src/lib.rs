#![doc = include_str!("../readme.md")]
#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod c99_complex;
mod ext_complex_float;
mod ext_float;
mod ext_num;
mod ext_real;
mod ext_sort_cmp;
mod ext_zero;
mod isclose;
mod promotion;
mod val_write;

pub use ext_complex_float::*;
pub use ext_float::*;
pub use ext_num::*;
pub use ext_real::*;
pub use ext_sort_cmp::*;
pub use ext_zero::*;
pub use isclose::*;
pub use promotion::*;
pub use val_write::*;
