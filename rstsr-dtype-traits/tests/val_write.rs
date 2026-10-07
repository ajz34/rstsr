//! Integration tests for the `val_write` module.

use core::mem::MaybeUninit;
use rstsr_dtype_traits::*;

#[test]
fn playground() {
    fn inner<W, T>(a: &mut W, b: T)
    where
        W: ValWriteAPI<T>,
    {
        a.write(b);
    }

    let mut a = 0.0;
    inner(&mut a, 1.0);
    println!("a {a:?}");

    let mut b = MaybeUninit::uninit();
    inner(&mut b, 1.0);
    println!("b {b:?}");
}
