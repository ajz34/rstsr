//! Basic math operations.

use core::ops::Add;

use crate::prelude_dev::*;

/// Determine type of iteration during binary operation.
///
/// - When two tensors are the same contiguous layout, use one-dimension
///   iterator
/// - When two tensors are the same prefer layout, use corresponding iterator
/// - If tensor to be assigned is c/f preferred layout, use corresponding
///   iterator
/// - Otherwise, use default layout iterator
fn iter_type_binary<D1, D2>(l1: &Layout<D1>, l2: &Layout<D2>) -> Result<IterLayoutType>
where
    D1: DimAPI,
    D2: DimAPI,
{
    rstsr_assert_eq!(l1.size(), l2.size(), InvalidLayout)?;
    if l1.is_c_contig() && l2.is_c_contig() || l1.is_f_contig() && l2.is_f_contig() {
        // contiguous layout use one-dimension iterator
        Ok(IterLayoutType::MemNonStrided)
    } else {
        // in other situations, try layout of array to be assigned,
        // if it is not c/f-prefer, then use default layout
        if l1.is_c_prefer() && l2.is_c_prefer() {
            Ok(IterLayoutType::C)
        } else if l1.is_f_prefer() && l2.is_f_prefer() {
            Ok(IterLayoutType::F)
        } else if l1.is_c_prefer() {
            Ok(IterLayoutType::C)
        } else if l1.is_f_prefer() {
            Ok(IterLayoutType::F)
        } else if TensorOrder::default() == TensorOrder::C {
            Ok(IterLayoutType::C)
        } else {
            Ok(IterLayoutType::F)
        }
    }
}

fn iter_type_ternary<D1, D2, D3>(
    l1: &Layout<D1>,
    l2: &Layout<D2>,
    l3: &Layout<D3>,
) -> Result<IterLayoutType>
where
    D1: DimAPI,
    D2: DimAPI,
    D3: DimAPI,
{
    rstsr_assert_eq!(l1.size(), l2.size(), InvalidLayout)?;
    rstsr_assert_eq!(l1.size(), l3.size(), InvalidLayout)?;
    if l1.is_c_contig() && l2.is_c_contig() && l3.is_c_contig()
        || l1.is_f_contig() && l2.is_f_contig() && l3.is_f_contig()
    {
        // contiguous layout use one-dimension iterator
        Ok(IterLayoutType::MemNonStrided)
    } else {
        // in other situations, try layout of array to be assigned,
        // if it is not c/f-prefer, then use default layout
        if l1.is_c_prefer() && l2.is_c_prefer() && l3.is_c_prefer() {
            Ok(IterLayoutType::C)
        } else if l1.is_f_prefer() && l2.is_f_prefer() && l3.is_f_prefer() {
            Ok(IterLayoutType::F)
        } else if l1.is_c_prefer() {
            Ok(IterLayoutType::C)
        } else if l1.is_f_prefer() {
            Ok(IterLayoutType::F)
        } else if TensorOrder::default() == TensorOrder::C {
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

impl<TA, TB, TC, DA, DB, DC> OpAddAPI<TA, TB, TC, DA, DB, DC> for CpuDevice
where
    TA: Add<TB, Output = TC> + Clone,
    TB: Clone,
    TC: Clone,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
{
    fn add(
        &self,
        c: &mut Storage<TC, CpuDevice>,
        lc: &Layout<DC>,
        a: &mut Storage<TA, CpuDevice>,
        la: &Layout<DA>,
        b: &mut Storage<TB, CpuDevice>,
        lb: &Layout<DB>,
    ) -> Result<()> {
        let it_type = iter_type_ternary(la, lb, lc)?;
        let it_a = iter_layout_by_type(it_type, la)?;
        let it_b = iter_layout_by_type(it_type, lb)?;
        let it_c = iter_layout_by_type(it_type, lc)?;
        for (ia, ib, ic) in izip!(it_a, it_b, it_c) {
            c.rawvec[ic] = a.rawvec[ia].clone() + b.rawvec[ib].clone();
        }
        Ok(())
    }
}
