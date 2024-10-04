//! This module is to declare API for external tensor objects to rstsr object.

pub trait IntoRSTSR {
    type RSTSR;
    fn into_rstsr(self) -> Self::RSTSR;
}
