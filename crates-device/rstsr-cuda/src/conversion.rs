//! Cross-device transfers for `DeviceCuda`.
//!
//! Host materialization goes through `DeviceToCpuAPI` (post-#164), not the
//! 1-D-only `into_vec`. CUDA->CUDA copies work at the data level
//! (`DataCloneAPI`), so no tensor-level `OpAssignAPI` is needed here.

use crate::buffer::CudaBuffer;
use crate::prelude_dev::*;
use crate::transfer;
use std::sync::Arc;

fn cpu_tensor<T, D>(vec: Vec<T>, layout: Layout<D>) -> TensorAny<DataOwned<Vec<T>>, T, DeviceCpuSerial, D>
where
    T: Clone + Send + Sync + 'static,
    D: DimAPI,
{
    let storage = Storage::new(DataOwned::from(vec), DeviceCpuSerial::default());
    TensorAny::new_f(storage, layout).rstsr_unwrap()
}

/// CPU -> CUDA (copy; a borrowed host view of device memory cannot exist).
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCuda, R, T, D> for DeviceCpuSerial
where
    T: Copy + Send + Sync + 'static,
    R: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
{
    type Repr = DataOwned<CudaBuffer<T>>;
    type ReprTo = DataOwned<CudaBuffer<T>>;

    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCuda, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let vec = storage.into_cpu_vec()?;
        let raw = transfer::h2d_full(device, vec)?;
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
    }

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<DataOwned<CudaBuffer<T>>, T, DeviceCuda, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let vec = storage.into_cpu_vec()?;
        let raw = transfer::h2d_full(device, vec)?;
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
    }

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCuda, D>> {
        let vec = DeviceToCpuAPI::<T>::to_cpu_vec::<R>(tensor.storage())?;
        let raw = transfer::h2d_full(device, vec)?;
        let layout = tensor.layout().clone();
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
    }
}

/// CUDA -> CPU serial (copy).
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCpuSerial, R, T, D> for DeviceCuda
where
    T: Copy + Send + Sync + 'static,
    R: DataCloneAPI<Data = CudaBuffer<T>>,
    D: DimAPI,
{
    type Repr = DataOwned<Vec<T>>;
    type ReprTo = DataOwned<Vec<T>>;

    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        _device: &DeviceCpuSerial,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCpuSerial, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let vec = storage.into_cpu_vec()?;
        Ok(cpu_tensor(vec, layout))
    }

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        _device: &DeviceCpuSerial,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceCpuSerial, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let vec = storage.into_cpu_vec()?;
        Ok(cpu_tensor(vec, layout))
    }

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        _device: &DeviceCpuSerial,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCpuSerial, D>> {
        let vec = DeviceToCpuAPI::<T>::to_cpu_vec::<R>(tensor.storage())?;
        let layout = tensor.layout().clone();
        Ok(cpu_tensor(vec, layout))
    }
}

/// Faer CPU <-> CUDA (copy; same shape as the CPU-serial impls above).
#[cfg(feature = "faer")]
mod faer_impls {
    use super::*;

    impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceCuda, R, T, D> for DeviceFaer
    where
        T: Copy + Send + Sync + 'static,
        D: DimAPI,
        R: DataCloneAPI<Data = Vec<T>>,
    {
        type Repr = DataOwned<CudaBuffer<T>>;
        type ReprTo = DataOwned<CudaBuffer<T>>;

        fn change_device(
            tensor: TensorAny<R, T, Self, D>,
            device: &DeviceCuda,
        ) -> Result<TensorAny<Self::Repr, T, DeviceCuda, D>> {
            let (storage, layout) = tensor.into_raw_parts();
            let vec = DeviceToCpuAPI::<T>::into_cpu_vec(storage)?;
            let raw = transfer::h2d_full(device, vec)?;
            Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
        }

        fn into_device(
            tensor: TensorAny<R, T, Self, D>,
            device: &DeviceCuda,
        ) -> Result<TensorAny<DataOwned<CudaBuffer<T>>, T, DeviceCuda, D>> {
            let (storage, layout) = tensor.into_raw_parts();
            let vec = DeviceToCpuAPI::<T>::into_cpu_vec(storage)?;
            let raw = transfer::h2d_full(device, vec)?;
            Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
        }

        fn to_device(
            tensor: &'a TensorAny<R, T, Self, D>,
            device: &DeviceCuda,
        ) -> Result<TensorAny<Self::ReprTo, T, DeviceCuda, D>> {
            let layout = tensor.layout().clone();
            let vec = DeviceToCpuAPI::<T>::to_cpu_vec::<R>(tensor.storage())?;
            let raw = transfer::h2d_full(device, vec)?;
            Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
        }
    }

    impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceFaer, R, T, D> for DeviceCuda
    where
        T: Copy + Send + Sync + 'static,
        D: DimAPI,
        R: DataCloneAPI<Data = CudaBuffer<T>>,
    {
        type Repr = DataOwned<Vec<T>>;
        type ReprTo = DataOwned<Vec<T>>;

        fn change_device(
            tensor: TensorAny<R, T, Self, D>,
            device: &DeviceFaer,
        ) -> Result<TensorAny<Self::Repr, T, DeviceFaer, D>> {
            let (storage, layout) = tensor.into_raw_parts();
            let vec = storage.into_cpu_vec()?;
            let storage = Storage::new(DataOwned::from(vec), device.clone());
            Ok(TensorAny::new_f(storage, layout).rstsr_unwrap())
        }

        fn into_device(
            tensor: TensorAny<R, T, Self, D>,
            device: &DeviceFaer,
        ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceFaer, D>> {
            Self::change_device(tensor, device)
        }

        fn to_device(
            tensor: &'a TensorAny<R, T, Self, D>,
            device: &DeviceFaer,
        ) -> Result<TensorAny<Self::ReprTo, T, DeviceFaer, D>> {
            let layout = tensor.layout().clone();
            let vec = DeviceToCpuAPI::<T>::to_cpu_vec::<R>(tensor.storage())?;
            let storage = Storage::new(DataOwned::from(vec), device.clone());
            Ok(TensorAny::new_f(storage, layout).rstsr_unwrap())
        }
    }
}

/// CUDA -> CUDA: same-ordinal zero-copy relabel; cross-ordinal peer copy.
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCuda, R, T, D> for DeviceCuda
where
    T: Copy + Send + Sync + 'static,
    R: DataCloneAPI<Data = CudaBuffer<T>>,
    D: DimAPI,
{
    type Repr = R;
    type ReprTo = DataCow<'l, CudaBuffer<T>>;

    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCuda, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        if Arc::ptr_eq(storage.device().ordinal_obj(), device.ordinal_obj()) {
            // same GPU: pure relabel, buffer untouched (zero data movement);
            // order/stream/flags-only differences need no copy
            let (data, _) = storage.into_raw_parts();
            return Ok(TensorAny::new(Storage::new(data, device.clone()), layout));
        }
        // Repr is a passthrough; a cross-ordinal copy cannot rewrap an
        // arbitrary `R` — direct users to `into_device`.
        let (_, _) = (storage, layout);
        Err(rstsr_error!(
            DeviceMismatch,
            "change_device across ordinals cannot preserve the source repr; use into_device"
        ))
    }

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<DataOwned<CudaBuffer<T>>, T, DeviceCuda, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        // Same ordinal: order/flags/stream-only differences need no data
        // movement (the buffer is order-agnostic), so relabel instead of
        // copying; only a cross-ordinal transfer allocates and copies.
        let same_ordinal = Arc::ptr_eq(storage.device().ordinal_obj(), device.ordinal_obj());
        let (data, _) = storage.into_raw_parts();
        let owned = data.into_owned().into_raw();
        if same_ordinal {
            return Ok(TensorAny::new_f(Storage::new(DataOwned::from(owned), device.clone()), layout).rstsr_unwrap());
        }
        let mut raw = CudaBuffer::<T>::new_f(device.stream(), owned.len())?;
        transfer::dtod_full(&owned, &mut raw)?;
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
    }

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCuda, D>> {
        let layout = tensor.layout().clone();
        if Arc::ptr_eq(tensor.device().ordinal_obj(), device.ordinal_obj()) {
            // same GPU: view re-tag, borrowing the same buffer under the new
            // device value (no copy; see `into_device`)
            let storage = Storage::new(DataCow::Ref(DataRef::TrueRef(tensor.raw())), device.clone());
            return Ok(TensorAny::new_f(storage, layout).rstsr_unwrap());
        }
        let src = tensor.raw();
        let mut raw = CudaBuffer::<T>::new_f(device.stream(), src.len())?;
        transfer::dtod_full(src, &mut raw)?;
        let storage = Storage::new(DataCow::Owned(DataOwned::from(raw)), device.clone());
        Ok(TensorAny::new_f(storage, layout).rstsr_unwrap())
    }
}
