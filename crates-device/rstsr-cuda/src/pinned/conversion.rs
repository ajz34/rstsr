//! Cross-device transfers for `DeviceCudaPinned`.
//!
//! Pinned <-> CUDA edges are **non-blocking**: the copy is enqueued on the
//! `DeviceCuda` value's stream and the pinned buffer's event orders every
//! later host access against it. Host-side edges (`DeviceCpuSerial`,
//! `DeviceFaer`) are copies; zero-copy host access is `host_view*`.

use crate::device::DeviceCuda;
use crate::pinned::buffer::{copy_vec_into, PinnedBuffer};
use crate::pinned::device::DeviceCudaPinned;
use crate::pinned::transfer;
use crate::prelude_dev::*;

fn cpu_tensor<T, D>(vec: Vec<T>, layout: Layout<D>) -> TensorAny<DataOwned<Vec<T>>, T, DeviceCpuSerial, D>
where
    D: DimAPI,
{
    let storage = Storage::new(DataOwned::from(vec), DeviceCpuSerial::default());
    TensorAny::new_f(storage, layout).rstsr_unwrap()
}

/// Pinned -> CUDA (non-blocking copy on `device`'s stream).
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCuda, R, T, D> for DeviceCudaPinned
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataCloneAPI<Data = PinnedBuffer<T>>,
{
    type Repr = DataOwned<CudaBuffer<T>>;
    type ReprTo = DataOwned<CudaBuffer<T>>;

    /// Consuming form: the source buffer is dropped after enqueueing, and
    /// dropping it synchronizes the event — effectively blocking. Pipelines
    /// use the borrowing `to_device`.
    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCuda, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let buf = data.into_owned().into_raw();
        let raw = transfer::h2d_pinned_async(device, &buf)?;
        drop(buf);
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
    }

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<DataOwned<CudaBuffer<T>>, T, DeviceCuda, D>> {
        Self::change_device(tensor, device)
    }

    /// Borrowing form: returns after enqueue; the pinned source stays alive
    /// and untouched until the copy completes.
    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        device: &DeviceCuda,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCuda, D>> {
        let raw = transfer::h2d_pinned_async(device, tensor.raw())?;
        let layout = tensor.layout().clone();
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(raw), device.clone()), layout).rstsr_unwrap())
    }
}

/// CUDA -> pinned (non-blocking copy on `Self`'s stream; the result is host-ready
/// after its event sync, which every host access performs).
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCudaPinned, R, T, D> for DeviceCuda
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataCloneAPI<Data = CudaBuffer<T>>,
{
    type Repr = DataOwned<PinnedBuffer<T>>;
    type ReprTo = DataOwned<PinnedBuffer<T>>;

    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCudaPinned,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCudaPinned, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, dev) = storage.into_raw_parts();
        let src = data.into_owned().into_raw();
        let mut dst = PinnedBuffer::<T>::new_f(device.context(), src.len())?;
        transfer::d2h_pinned_async(&dev, &src, &mut dst)?;
        drop(src);
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(dst), device.clone()), layout).rstsr_unwrap())
    }

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCudaPinned,
    ) -> Result<TensorAny<DataOwned<PinnedBuffer<T>>, T, DeviceCudaPinned, D>> {
        Self::change_device(tensor, device)
    }

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        device: &DeviceCudaPinned,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCudaPinned, D>> {
        let mut dst = PinnedBuffer::<T>::new_f(device.context(), tensor.raw().len())?;
        transfer::d2h_pinned_async(tensor.storage().device(), tensor.raw(), &mut dst)?;
        let layout = tensor.layout().clone();
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(dst), device.clone()), layout).rstsr_unwrap())
    }
}

/// Pinned -> CPU serial (copy into a `Vec`; zero-copy is `host_view`).
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCpuSerial, R, T, D> for DeviceCudaPinned
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataCloneAPI<Data = PinnedBuffer<T>>,
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
        device: &DeviceCpuSerial,
    ) -> Result<TensorAny<DataOwned<Vec<T>>, T, DeviceCpuSerial, D>> {
        Self::change_device(tensor, device)
    }

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        _device: &DeviceCpuSerial,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCpuSerial, D>> {
        let vec = tensor.raw().to_vec_f()?;
        let layout = tensor.layout().clone();
        Ok(cpu_tensor(vec, layout))
    }
}

/// CPU serial -> pinned (copy of a `Vec`).
impl<'l, R, T, D> DeviceChangeAPI<'l, DeviceCudaPinned, R, T, D> for DeviceCpuSerial
where
    T: Copy + Send + Sync + 'static,
    D: DimAPI,
    R: DataCloneAPI<Data = Vec<T>>,
{
    type Repr = DataOwned<PinnedBuffer<T>>;
    type ReprTo = DataOwned<PinnedBuffer<T>>;

    fn change_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCudaPinned,
    ) -> Result<TensorAny<Self::Repr, T, DeviceCudaPinned, D>> {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, _) = storage.into_raw_parts();
        let vec = data.into_owned().into_raw();
        let mut buf = PinnedBuffer::<T>::new_f(device.context(), vec.len())?;
        copy_vec_into(&mut buf, &vec)?;
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(buf), device.clone()), layout).rstsr_unwrap())
    }

    fn into_device(
        tensor: TensorAny<R, T, Self, D>,
        device: &DeviceCudaPinned,
    ) -> Result<TensorAny<DataOwned<PinnedBuffer<T>>, T, DeviceCudaPinned, D>> {
        Self::change_device(tensor, device)
    }

    fn to_device(
        tensor: &'l TensorAny<R, T, Self, D>,
        device: &DeviceCudaPinned,
    ) -> Result<TensorAny<Self::ReprTo, T, DeviceCudaPinned, D>> {
        let buf = device.from_vec_f(tensor.raw().clone())?;
        let layout = tensor.layout().clone();
        Ok(TensorAny::new_f(Storage::new(DataOwned::from(buf), device.clone()), layout).rstsr_unwrap())
    }
}

/// Faer CPU <-> pinned (copies; same shape as the CPU-serial impls above).
#[cfg(feature = "faer")]
mod faer_impls {
    use super::*;

    impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceCudaPinned, R, T, D> for DeviceFaer
    where
        T: Copy + Send + Sync + 'static,
        D: DimAPI,
        R: DataCloneAPI<Data = Vec<T>>,
    {
        type Repr = DataOwned<PinnedBuffer<T>>;
        type ReprTo = DataOwned<PinnedBuffer<T>>;

        fn change_device(
            tensor: TensorAny<R, T, Self, D>,
            device: &DeviceCudaPinned,
        ) -> Result<TensorAny<Self::Repr, T, DeviceCudaPinned, D>> {
            let (storage, layout) = tensor.into_raw_parts();
            let (data, _) = storage.into_raw_parts();
            let vec = data.into_owned().into_raw();
            let mut buf = PinnedBuffer::<T>::new_f(device.context(), vec.len())?;
            copy_vec_into(&mut buf, &vec)?;
            Ok(TensorAny::new_f(Storage::new(DataOwned::from(buf), device.clone()), layout).rstsr_unwrap())
        }

        fn into_device(
            tensor: TensorAny<R, T, Self, D>,
            device: &DeviceCudaPinned,
        ) -> Result<TensorAny<DataOwned<PinnedBuffer<T>>, T, DeviceCudaPinned, D>> {
            Self::change_device(tensor, device)
        }

        fn to_device(
            tensor: &'a TensorAny<R, T, Self, D>,
            device: &DeviceCudaPinned,
        ) -> Result<TensorAny<Self::ReprTo, T, DeviceCudaPinned, D>> {
            let buf = device.from_vec_f(tensor.raw().clone())?;
            let layout = tensor.layout().clone();
            Ok(TensorAny::new_f(Storage::new(DataOwned::from(buf), device.clone()), layout).rstsr_unwrap())
        }
    }

    impl<'a, R, T, D> DeviceChangeAPI<'a, DeviceFaer, R, T, D> for DeviceCudaPinned
    where
        T: Copy + Send + Sync + 'static,
        D: DimAPI,
        R: DataCloneAPI<Data = PinnedBuffer<T>>,
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
            let vec = tensor.raw().to_vec_f()?;
            let storage = Storage::new(DataOwned::from(vec), device.clone());
            let layout = tensor.layout().clone();
            Ok(TensorAny::new_f(storage, layout).rstsr_unwrap())
        }
    }
}
