pub mod rstsr_traits {
    pub use crate::layout::exports::{
        DimAPI, DimBaseAPI, DimBroadcastableAPI, DimDevAPI, DimIntoAPI, DimLargerOneAPI, DimLayoutContigAPI, DimMaxAPI,
        DimShapeAPI, DimSmallerOneAPI,
    };
    pub use crate::pack_array::{PackArrayAPI, PackableArrayAPI, UnpackArrayAPI};
    pub use core::ops::*;
}

pub mod rstsr_structs {
    pub use crate::flags::{ColMajor, ConjTrans, Lower, NoTrans, NonUnit, RowMajor, Trans, Unit, Upper};
    pub use crate::flags::{
        FlagDiag, FlagOrder, FlagSide, FlagSymm, FlagTrans, FlagUpLo, TensorCopyPolicy, TensorDiag, TensorIterOrder,
        TensorOrder, TensorSide, TensorSymm, TensorTrans, TensorUpLo,
    };
    pub use crate::layout::exports::{Ix, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, Ix7, Ix8, Ix9, IxD, IxDyn, Layout};
    pub use crate::layout::indexer::{Ellipsis, NewAxis};
}

pub mod rstsr_macros {
    pub use crate::{
        rstsr_assert, rstsr_assert_eq, rstsr_errcode, rstsr_error, rstsr_invalid, rstsr_pattern, rstsr_raise,
    };
    pub use crate::{s, slice};
}

// final re-exports

// Only structs and traits are re-exported at this level. Functions and macros may clash with other
// crates / usual functions.
pub use rstsr_structs::*;
pub use rstsr_traits::*;

// For macros, the `slice!` is useful and should be exported at this level. Other macros can be
// called by `rt::macro_name!` instead.
pub use rstsr_macros::slice;

pub mod rt {
    pub use super::rstsr_macros;
    pub use super::rstsr_structs;
    pub use super::rstsr_traits;

    pub use super::rstsr_macros::*;
    pub use super::rstsr_structs::*;
    pub use super::rstsr_traits::*;

    pub use crate::error::{Error, Result};
}
