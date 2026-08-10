use crate::prelude_dev::*;

impl DeviceHDF5 {
    pub fn to_tensor<T>(&self) -> Tensor<T, DeviceHDF5, IxD>
    where
        T: H5Type + Copy,
    {
        self.to_tensor_f().rstsr_unwrap()
    }

    pub fn to_tensor_mut<T>(&mut self) -> TensorMut<'_, T, DeviceHDF5, IxD>
    where
        T: H5Type + Copy,
    {
        self.to_tensor_mut_f().rstsr_unwrap()
    }

    pub fn to_tensor_view<T>(&self) -> TensorView<'_, T, DeviceHDF5, IxD>
    where
        T: H5Type + Copy,
    {
        self.to_tensor_view_f().rstsr_unwrap()
    }

    pub fn to_tensor_f<T>(&self) -> Result<Tensor<T, DeviceHDF5, IxD>>
    where
        T: H5Type + Copy,
    {
        // owned tensor must be writable
        if self.h5file().is_read_only() {
            return Err(rstsr_error!(IOError, "Cannot create an owned tensor from a read-only HDF5 file"));
        }
        let dataset = self.dataset_f()?;
        verify_dataset_type::<T>(&dataset)?;
        let shape = dataset.shape().to_vec();
        let layout = shape.c();

        let data = DataOwned::from(dataset);
        let storage = Storage::new(data, self.clone());
        Tensor::new_f(storage, layout)
    }

    pub fn to_tensor_mut_f<T>(&mut self) -> Result<TensorMut<'_, T, DeviceHDF5, IxD>>
    where
        T: H5Type + Copy,
    {
        // owned tensor must be writable
        if self.h5file().is_read_only() {
            return Err(rstsr_error!(IOError, "Cannot create an owned tensor from a read-only HDF5 file"));
        }
        let dataset = self.dataset_f()?;
        verify_dataset_type::<T>(&dataset)?;
        let shape = dataset.shape().to_vec();
        let layout = shape.c();

        use core::mem::ManuallyDrop;
        let data = DataMut::from_manually_drop(ManuallyDrop::new(dataset));
        let storage = Storage::new(data, self.clone());
        TensorMut::new_f(storage, layout)
    }

    pub fn to_tensor_view_f<T>(&self) -> Result<TensorView<'_, T, DeviceHDF5, IxD>>
    where
        T: H5Type + Copy,
    {
        let dataset = self.dataset_f()?;
        verify_dataset_type::<T>(&dataset)?;
        let shape = dataset.shape().to_vec();
        let layout = shape.c();

        use core::mem::ManuallyDrop;
        let data = DataRef::from_manually_drop(ManuallyDrop::new(dataset));
        let storage = Storage::new(data, self.clone());
        TensorView::new_f(storage, layout)
    }
}

#[test]
fn playground_hdf5_tensor() {
    let device = DeviceHDF5::new_f("/home/a/rstsr_pack/tmp/play.h5", H5OpenMode::ReadWrite, "/b/0").unwrap();
    let tensor = device.to_tensor::<i64>();
    println!("Tensor shape: {:?}", tensor.shape());
    let device = DeviceHDF5::new_f("/home/a/rstsr_pack/tmp/play.h5", H5OpenMode::ReadWrite, "/a").unwrap();
    let tensor = device.to_tensor::<i64>();
    println!("Tensor shape: {:?}", tensor);

    let device = DeviceHDF5::default();
    println!("Default device: {:?}", device);
    println!("Full path: {:?}", device.h5file().filename());
    println!("Full path exists: {:?}", std::path::Path::new(&device.h5file().filename()).exists());
    device.h5file().create_group("a").unwrap();
    println!("{:?}", device.h5file().group("a").unwrap().is_valid());
    println!("Full path exists: {:?}", std::path::Path::new(&device.h5file().filename()).exists());
}

#[test]
fn temporary_device_path_valid_until_drop() {
    // A temporary (default) device must expose a real on-disk path that exists for its whole
    // lifetime, and must be auto-cleaned (unlinked) once the last clone is dropped.
    let device = DeviceHDF5::default();
    let path = device.h5file().filename();
    assert!(
        std::path::Path::new(&path).exists(),
        "temp file path should exist while the device is alive, got {path:?}",
    );

    // HDF5 writes go through its own file descriptor, but the path must remain reachable so
    // the file can also be reopened by name.
    device.h5file().create_group("a").unwrap();
    assert!(device.h5file().group("a").unwrap().is_valid());
    assert!(std::path::Path::new(&path).exists(), "path still exists after HDF5 writes");

    // `filename()` returns an owned String, so `path` is independent of `device` and survives
    // the drop; assert the directory entry is removed once the device is gone.
    drop(device);
    assert!(
        !std::path::Path::new(&path).exists(),
        "temp file should be auto-cleaned after the device is dropped, still at {path:?}",
    );
}
