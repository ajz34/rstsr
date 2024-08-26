use crate::prelude_dev::*;

/// Determine type of iteration during binary operation.
///
/// - When two tensors are the same contiguous layout, use one-dimension
///   iterator
/// - When two tensors are the same prefer layout, use corresponding iterator
/// - If tensor to be assigned is c/f preferred layout, use corresponding
///   iterator
/// - Otherwise, use default layout iterator
fn iter_type_binary<D1, D2>(la: &Layout<D1>, lb: &Layout<D2>) -> Result<IterLayoutType>
where
    D1: DimAPI,
    D2: DimAPI,
{
    rstsr_assert_eq!(la.size(), lb.size(), InvalidLayout)?;
    if la.is_c_contig() && lb.is_c_contig() || la.is_f_contig() && lb.is_f_contig() {
        // contiguous layout use one-dimension iterator
        Ok(IterLayoutType::MemNonStrided)
    } else {
        // in other situations, try layout of array to be assigned,
        // if it is not c/f-prefer, then use default layout
        if la.is_c_prefer() && lb.is_c_prefer() {
            Ok(IterLayoutType::C)
        } else if la.is_f_prefer() && lb.is_f_prefer() {
            Ok(IterLayoutType::F)
        } else if la.is_c_prefer() {
            Ok(IterLayoutType::C)
        } else if la.is_f_prefer() {
            Ok(IterLayoutType::F)
        } else if Order::default() == Order::C {
            Ok(IterLayoutType::C)
        } else {
            Ok(IterLayoutType::F)
        }
    }
}

impl<T, D1, D2> OpAssignAPI<T, D1, D2> for CpuDevice
where
    T: Clone,
    D1: DimAPI,
    D2: DimAPI,
{
    fn assign_arbitary_layout(
        &self,
        a: &mut Storage<T, CpuDevice>,
        la: &Layout<D1>,
        b: &Storage<T, CpuDevice>,
        lb: &Layout<D2>,
    ) -> Result<()> {
        let it_type = iter_type_binary(la, lb)?;
        let it_a = iter_layout_by_type(it_type, la)?;
        let it_b = iter_layout_by_type(it_type, lb)?;
        for (ia, ib) in it_a.zip(it_b) {
            a.rawvec[ia] = b.rawvec[ib].clone();
        }
        Ok(())
    }
}
