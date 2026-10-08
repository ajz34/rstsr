#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;
#[cfg(feature = "rayon")]
pub mod cpu_rayon;
pub mod cpu_serial;

pub mod prelude_dev;
