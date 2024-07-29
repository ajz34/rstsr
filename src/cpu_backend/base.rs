use crate::base_device::TraitDevice;
use core::fmt::Debug;

/* #region Device */

#[derive(Default, Clone, PartialEq)]
pub struct CpuDevice {
    id: isize,
}

impl CpuDevice {
    pub fn get_id(&self) -> isize {
        self.id
    }
}

impl TraitDevice for CpuDevice {
    fn device(&self) -> Self {
        self.clone()
    }
}

impl Debug for CpuDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Device CPU({:})", self.id)
    }
}

/* #endregion Device */

/* #region Storage */

pub struct CpuStorage<T> {
    storage: Vec<T>,
}

impl<T> CpuStorage<T>
where
    T: Clone,
{
    pub fn to_vec(&self) -> Vec<T> {
        self.storage.clone()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage
    }
}

/* #endregion */
