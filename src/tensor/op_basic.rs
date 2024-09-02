use core::ops::Add;

use crate::prelude_dev::*;

/// Operations that changes current tensor.
impl<R, T, D, B> TensorBase<R, D>
where
    R: DataMutAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn add_ternary<RA, RB, DA, DB, TB>(
        &mut self,
        a: &TensorBase<RA, DA>,
        b: &TensorBase<RB, DB>,
    ) -> Result<()>
    where
        // lifetime and data constraints
        RA: DataAPI<Data = Storage<T, B>>,
        RB: DataAPI<Data = Storage<TB, B>>,
        DA: DimAPI,
        DB: DimAPI,
        B: DeviceAPI<TB>,
        // broadcast constraints
        D: DimMaxAPI<DA> + DimMaxAPI<DB>,
        <D as DimMaxAPI<DA>>::Max: DimConvertAPI<D>,
        <D as DimMaxAPI<DB>>::Max: DimConvertAPI<D>,
        // operation constraints
        TB: Clone,
        T: Add<TB, Output = T> + Clone,
        B: OpAddAPI<T, TB, D>,
    {
        let lc = self.layout();
        let la = a.layout();
        let lb = b.layout();
        // all layouts should be broadcastable to lc
        // we can first generate broadcasted shape, then check this
        let (lc_b, la_b) = broadcast_layout_to_first(lc, la)?;
        rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
        let (lc_b, lb_b) = broadcast_layout_to_first(lc, lb)?;
        rstsr_assert_eq!(lc_b, *lc, InvalidLayout)?;
        // add provided by device
        let device = self.device().clone();
        let storage_c = self.data_mut().as_storage_mut();
        let storage_a = a.data().storage();
        let storage_b = b.data().storage();
        device.add_ternary(storage_c, &lc_b, storage_a, &la_b, storage_b, &lb_b)
    }
}

/// Operations that will not change current tensor.
impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn add_binary<RB, DB, TB>(
        &self,
        b: &TensorBase<RB, DB>,
    ) -> Result<Tensor<T, <D as DimMaxAPI<DB>>::Max, B>>
    where
        // lifetime and data constraints
        RB: DataAPI<Data = Storage<TB, B>>,
        DB: DimAPI,
        B: DeviceAPI<TB> + DeviceCreationAnyAPI<T>,
        // broadcast constraints
        D: DimMaxAPI<DB>,
        <D as DimMaxAPI<DB>>::Max: DimAPI,
        // operation constraints
        T: Add<TB, Output = T> + Clone,
        B: OpAddAPI<T, TB, <D as DimMaxAPI<DB>>::Max>,
    {
        let la = self.layout();
        let lb = b.layout();
        let (la_b, lb_b) = broadcast_layout(la, lb)?;
        // generate output layout
        let lc = if la_b.c_contig() && lb_b.c_contig() {
            la_b.shape().c()
        } else if la_b.f_contig() && lb_b.f_contig() {
            la_b.shape().f()
        } else {
            match TensorOrder::default() {
                TensorOrder::C => la_b.shape().c(),
                TensorOrder::F => la_b.shape().f(),
            }
        };
        // generate empty c
        let device = self.device().clone();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index()?.1)? };
        // add provided by device
        let storage_a = self.data().storage();
        let storage_b = b.data().storage();
        device.add_ternary(&mut storage_c, &lc, storage_a, &la_b, storage_b, &lb_b)?;
        // return tensor
        Tensor::new(DataOwned { storage: storage_c }, lc)
    }
}

impl<R, T, D, B, RB, DB, TB> Add<&TensorBase<RB, DB>> for &TensorBase<R, D>
where
    // lifetime and data
    // constraints
    R: DataAPI<Data = Storage<T, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    D: DimAPI,
    DB: DimAPI,
    B: DeviceAPI<T> + DeviceAPI<TB> + DeviceCreationAnyAPI<T>,
    // broadcast constraints
    D: DimMaxAPI<DB>,
    <D as DimMaxAPI<DB>>::Max: DimAPI,
    T: Add<TB, Output = T> + Clone,
    B: OpAddAPI<T, TB, <D as DimMaxAPI<DB>>::Max>,
{
    type Output = Tensor<T, <D as DimMaxAPI<DB>>::Max, B>;

    fn add(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
        self.add_binary(rhs).unwrap()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude_dev::*;

    #[test]
    fn test_add() {
        let a = Tensor::linspace_cpu(0.0, 20.0, 21);
        let b = Tensor::linspace_cpu(0.0, 20.0, 21);
        let c = &a + &b;
        println!("{c:6.3?}");
    }
}
