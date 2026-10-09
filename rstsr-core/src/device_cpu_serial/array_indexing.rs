//! Array indexing device impls for [`DeviceCpuSerial`].

use crate::prelude_dev::*;

impl<T> DeviceArrayIndexAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    fn array_index(
        &self,
        c: &mut Vec<MaybeUninit<T>>,
        lc: &Layout<IxD>,
        a: &Vec<T>,
        la: &Layout<IxD>,
        base_layout: &Layout<IxD>,
        indexers: &[ArrayAuxIndexer<'_, Self>],
        consec: usize,
        order: FlagOrder,
    ) -> Result<()> {
        // this device's raw buffer of `usize` is a host vector
        let indexers: Vec<(usize, &[usize], Layout<IxD>)> =
            indexers.iter().map(|ix| (ix.src_axis, ix.indices.as_slice(), ix.layout.clone())).collect();
        array_index_cpu_serial(c, lc, a, la, base_layout, &indexers, consec, order)
    }
}
