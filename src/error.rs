#[cfg(feature = "std")]
extern crate std;

use crate::prelude_dev::*;
use core::convert::Infallible;

#[cfg(feature = "cuda")]
use crate::cuda_backend::error::CudaError;

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    ValueOutOfRange(String),
    InvalidValue(String),
    InvalidLayout(String),
    RuntimeError(String),

    TryFromIntError(String),
    Infallible(String),

    Miscellaneous(String),
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        Error::Infallible("Infallible".to_string())
    }
}

#[macro_export]
macro_rules! rstsr_assert {
    ($cond:expr, $errtype:ident) => {
        if $cond {
            Ok(())
        } else {
            use crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : {:}", stringify!($cond)).unwrap();
            Err(Error::$errtype(s))
        }
    };
    ($cond:expr, $errtype:ident, $($arg:tt)*) => {{
        if $cond {
            Ok(())
        } else {
            use crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : ").unwrap();
            write!(s, $($arg)*).unwrap();
            write!(s, " : {:}", stringify!($cond)).unwrap();
            Err(Error::$errtype(s))
        }
    }};
}

#[macro_export]
macro_rules! rstsr_assert_eq {
    ($lhs:expr, $rhs:expr, $errtype:ident) => {
        if $lhs == $rhs {
            Ok(())
        } else {
            use crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(
                s,
                " : {:} = {:?} not equal to {:} = {:?}",
                stringify!($lhs),
                $lhs,
                stringify!($rhs),
                $rhs
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
}

#[macro_export]
macro_rules! rstsr_invalid {
    ($word:expr) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": "));
        write!(s, "Error::InvalidValue");
        write!(s, " : {:?} = {:?}", stringify!($word), $word).unwrap();
        Err(Error::InvalidValue(s))
    }};
}

#[macro_export]
macro_rules! rstsr_raise {
    ($errtype:ident, $($arg:tt)*) => {{
        use crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error::$errtype(s))
    }};
}

#[macro_export]
macro_rules! rstsr_pattern {
    ($value:expr, $pattern:expr, $errtype:ident) => {
        if ($pattern).contains(&($value)) {
            Ok(())
        } else {
            use crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(
                s,
                " : {:?} = {:?} not match to pattern {:} = {:?}",
                stringify!($value),
                $value,
                stringify!($pattern),
                $pattern
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
    ($value:expr, $pattern:expr, $errtype:ident, $($arg:tt)*) => {
        if ($pattern).contains(&($value)) {
            Ok(())
        } else {
            use crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : ").unwrap();
            write!(s, $($arg)*).unwrap();
            write!(
                s,
                " : {:?} = {:?} not match to pattern {:} = {:?}",
                stringify!($value),
                $value,
                stringify!($pattern),
                $pattern
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
}
