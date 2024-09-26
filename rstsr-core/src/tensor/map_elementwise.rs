use crate::prelude_dev::*;

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by reference on each element and create a new array with the
    /// new values.
    pub fn map<TOut>(&self, mut f: impl FnMut(&T) -> TOut) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutA_RefB_API<TOut, T, D>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default()).unwrap();
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let f_inner = move |c: &mut TOut, a: &T| *c = f(a);
        device.op_muta_refb_func(&mut storage_c, &lc, storage_a, la, f_inner).unwrap();
        return Tensor::new(DataOwned::from(storage_c), lc).unwrap();
    }

    /// Call `f` by value on each element and create a new array with the new
    /// values.
    pub fn mapv<TOut>(&self, mut f: impl FnMut(T) -> TOut) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        B: DeviceOp_MutA_RefB_API<TOut, T, D>,
    {
        self.map(move |x| f(x.clone()))
    }

    /// Modify the array in place by calling `f` by mutable reference on each
    /// element.
    pub fn map_inplace(&mut self, f: impl FnMut(&mut T))
    where
        R: DataMutAPI<Data = Storage<T, B>>,
        B: DeviceOp_MutA_API<T, D>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let storage_a = self.data_mut().storage_mut();
        device.op_muta_func(storage_a, &la, f).unwrap();
    }

    /// Modify the array in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapv_inplace(&mut self, mut f: impl FnMut(T) -> T)
    where
        R: DataMutAPI<Data = Storage<T, B>>,
        T: Clone,
        B: DeviceOp_MutA_API<T, D>,
    {
        self.map_inplace(move |x| *x = f(x.clone()));
    }

    pub fn map_binary<R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorBase<R2, D2>,
        mut f: impl FnMut(&T, &T2) -> TOut,
    ) -> Tensor<TOut, DOut, B>
    where
        R2: DataAPI<Data = Storage<T2, B>>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut>,
    {
        // get tensor views
        let a = self.view();
        let b = other.view();
        // check device and layout
        rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch).unwrap();
        let la = a.layout();
        let lb = b.layout();
        let (la_b, lb_b) = broadcast_layout(la, lb).unwrap();
        // generate output layout
        let lc_from_a = layout_for_array_copy(&la_b, TensorIterOrder::default()).unwrap();
        let lc_from_b = layout_for_array_copy(&lb_b, TensorIterOrder::default()).unwrap();
        let lc = if lc_from_a == lc_from_b {
            lc_from_a
        } else {
            match TensorOrder::default() {
                TensorOrder::C => la_b.shape().c(),
                TensorOrder::F => la_b.shape().f(),
            }
        };
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let storage_b = other.data().storage();
        let f_inner = move |c: &mut TOut, a: &T, b: &T2| *c = f(a, b);
        device
            .op_mutc_refa_refb_func(
                &mut storage_c,
                &lc,
                storage_a,
                &la_b,
                storage_b,
                &lb_b,
                f_inner,
            )
            .unwrap();
        return Tensor::new(DataOwned::from(storage_c), lc).unwrap();
    }

    pub fn mapv_binary<R2, T2, D2, DOut, TOut>(
        &self,
        other: &TensorBase<R2, D2>,
        mut f: impl FnMut(T, T2) -> TOut,
    ) -> Tensor<TOut, DOut, B>
    where
        R2: DataAPI<Data = Storage<T2, B>>,
        D2: DimAPI,
        DOut: DimAPI,
        D: DimMaxAPI<D2, Max = DOut>,
        T: Clone,
        T2: Clone,
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        B: DeviceOp_MutC_RefA_RefB_API<T, T2, TOut, DOut>,
    {
        self.map_binary(other, move |x, y| f(x.clone(), y.clone()))
    }
}

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by value on each element, update the array with the new values
    /// and return it.
    pub fn mapv_into(mut self, mut f: impl FnMut(T) -> T) -> Tensor<T, D, B>
    where
        T: Clone,
        B: DeviceOp_MutA_API<T, D>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let storage_a = self.data_mut().storage_mut();
        let f_inner = move |x: &mut T| *x = f(x.clone());
        device.op_muta_func(storage_a, &la, f_inner).unwrap();
        return self;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapv() {
        let f = |x| x * 2.0;
        let a = Tensor::from(vec![1., 2., 3., 4.]);
        let b = a.mapv(f);
        assert!(allclose_f64(&b, &vec![2., 4., 6., 8.].into()));
        println!("{:?}", b);
    }

    #[test]
    fn test_mapv_binary() {
        let f = |x, y| 2.0 * x + 3.0 * y;
        let a = Tensor::linspace_cpu(1., 6., 6).into_shape_assume_contig([2, 3]).unwrap();
        let b = Tensor::linspace_cpu(1., 3., 3);
        let c = a.mapv_binary(&b, f);
        assert!(allclose_f64(&c, &vec![5., 10., 15., 11., 16., 21.].into()));
    }
}
