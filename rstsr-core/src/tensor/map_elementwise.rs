use crate::prelude_dev::*;

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by reference on each element and create a new array with the
    /// new values.
    pub fn map<'a, TOut, F>(&'a self, f: F) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        F: Fn(&T) -> TOut + 'a,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'a>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::default()).unwrap();
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let mut f_inner = move |c: &mut TOut, a: &T| *c = f(a);
        device.op_muta_refb_func(&mut storage_c, &lc, storage_a, la, &mut f_inner).unwrap();
        return Tensor::new(DataOwned::from(storage_c), lc).unwrap();
    }

    /// Call `f` by value on each element and create a new array with the new
    /// values.
    pub fn mapv<'a, TOut, F>(&'a self, f: F) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        F: Fn(T) -> TOut + 'a,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'a>,
    {
        self.map(move |x| f(x.clone()))
    }

    /// Modify the array in place by calling `f` by mutable reference on each
    /// element.
    pub fn map_inplace<'a, F>(&'a mut self, mut f: F)
    where
        R: DataMutAPI<Data = Storage<T, B>>,
        F: FnMut(&mut T) + 'a,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'a>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let storage_a = self.data_mut().storage_mut();
        device.op_muta_func(storage_a, &la, &mut f).unwrap();
    }

    /// Modify the array in place by calling `f` by mutable reference on each
    /// element.
    pub fn mapv_inplace<'a, F>(&'a mut self, mut f: F)
    where
        R: DataMutAPI<Data = Storage<T, B>>,
        T: Clone,
        F: FnMut(T) -> T + 'a,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'a>,
    {
        self.map_inplace(move |x| *x = f(x.clone()));
    }
}

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by value on each element, update the array with the new values
    /// and return it.
    pub fn mapv_into<'a, F>(mut self, mut f: F) -> Tensor<T, D, B>
    where
        T: Clone,
        F: FnMut(T) -> T + 'a,
        B: DeviceOp_MutA_API<T, D, dyn FnMut(&mut T) + 'a>,
    {
        let (la, _) = greedy_layout(self.layout(), false);
        let device = self.device().clone();
        let storage_a = self.data_mut().storage_mut();
        let mut f_inner = move |x: &mut T| *x = f(x.clone());
        device.op_muta_func(storage_a, &la, &mut f_inner).unwrap();
        return self;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapv() {
        let f = |x: i32| x * 2;
        let a = Tensor::from(vec![1, 2, 3, 4]);
        let b = a.mapv(f);
        println!("{:?}", b);
    }
}
