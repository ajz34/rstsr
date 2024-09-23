use crate::prelude_dev::*;

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Call `f` by value on each element and create a new array with the new
    /// values.
    pub fn mapv<'a, TOut, F>(&'a self, f: F) -> Tensor<TOut, D, B>
    where
        B: DeviceAPI<TOut> + DeviceCreationAnyAPI<TOut>,
        T: Clone,
        F: Fn(T) -> TOut + 'a,
        B: DeviceOp_MutA_RefB_API<TOut, T, D, dyn FnMut(&mut TOut, &T) + 'a>,
    {
        let la = self.layout();
        let lc = layout_for_array_copy(la, TensorIterOrder::K).unwrap();
        let device = self.device();
        let mut storage_c = unsafe { device.empty_impl(lc.bounds_index().unwrap().1).unwrap() };
        let storage_a = self.data().storage();
        let mut f_inner = move |c: &mut TOut, a: &T| *c = f(a.clone());
        device.op_muta_refb_func(&mut storage_c, &lc, storage_a, la, &mut f_inner).unwrap();
        return Tensor::new(DataOwned::from(storage_c), lc).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapv() {
        let a = Tensor::from(vec![1, 2, 3, 4]);
        let b = a.mapv(|x| x * 2);
        println!("{:?}", b);
    }
}
