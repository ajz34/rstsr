use crate::prelude_dev::*;

/* #region base rayon device */

/// This is base device for Parallel CPU device.
///
/// This device is not intended to be used directly, but to be used as a base.
/// Possible inherited devices could be Faer or Blas.
/// 
/// This device is intended not to implement `DeviceAPI<T>`.
#[derive(Clone, Debug)]
pub struct DeviceCpuRayon {
    num_threads: usize,
}

impl DeviceCpuRayon {
    pub fn new(num_threads: usize) -> Self {
        DeviceCpuRayon { num_threads }
    }

    pub fn var_num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
    }

    pub fn get_num_threads(&self) -> usize {
        match self.num_threads {
            0 => rayon::current_num_threads(),
            _ => rayon::current_num_threads().max(self.num_threads),
        }
    }

    pub fn get_pool(&self, n: usize) -> Result<rayon::ThreadPool> {
        rstsr_pattern!(n, 0..self.get_num_threads(), RayonError, "Specified too much threads.")?;
        let nthreads = if n == 0 { self.get_num_threads() } else { n };
        rayon::ThreadPoolBuilder::new().num_threads(nthreads).build().map_err(Error::from)
    }
}

impl Default for DeviceCpuRayon {
    fn default() -> Self {
        DeviceCpuRayon::new(0)
    }
}

impl DeviceBaseAPI for DeviceCpuRayon {
    fn same_device(&self, other: &Self) -> bool {
        self.num_threads == other.num_threads
    }
}

/* #endregion */

/* #region device Faer */

#[derive(Clone, Debug)]
pub struct DeviceCpuFaer {
    base: DeviceCpuRayon,
}

impl DeviceCpuFaer {
    pub fn new(num_threads: usize) -> Self {
        DeviceCpuFaer { base: DeviceCpuRayon::new(num_threads) }
    }

    pub fn var_num_threads(&self) -> usize {
        self.base.var_num_threads()
    }

    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.base.set_num_threads(num_threads);
    }

    pub fn get_num_threads(&self) -> usize {
        self.base.get_num_threads()
    }

    pub fn get_pool(&self, n: usize) -> Result<rayon::ThreadPool> {
        self.base.get_pool(n)
    }
}

impl Default for DeviceCpuFaer {
    fn default() -> Self {
        DeviceCpuFaer::new(0)
    }
}

impl DeviceBaseAPI for DeviceCpuFaer {
    fn same_device(&self, other: &Self) -> bool {
        self.var_num_threads() == other.var_num_threads()
    }
}

impl<T> DeviceRawVecAPI<T> for DeviceCpuFaer
where
    T: Clone,
{
    type RawVec = Vec<T>;
}

impl<T> DeviceStorageAPI<T> for DeviceCpuFaer
where
    T: Clone,
{
    fn new(vector: Self::RawVec, device: Self) -> Storage<T, Self> {
        Storage::<T, Self> { rawvec: vector, device }
    }

    fn len(storage: &Storage<T, Self>) -> usize {
        storage.rawvec.len()
    }

    fn to_cpu_vec(storage: &Storage<T, Self>) -> Result<Vec<T>> {
        Ok(storage.rawvec.clone())
    }

    fn into_cpu_vec(storage: Storage<T, Self>) -> Result<Vec<T>> {
        Ok(storage.rawvec)
    }

    #[inline]
    fn get_index(storage: &Storage<T, Self>, index: usize) -> T {
        storage.rawvec[index].clone()
    }

    #[inline]
    fn get_index_ptr(storage: &Storage<T, Self>, index: usize) -> *const T {
        &storage.rawvec[index] as *const T
    }

    #[inline]
    fn get_index_mut_ptr(storage: &mut Storage<T, Self>, index: usize) -> *mut T {
        &mut storage.rawvec[index] as *mut T
    }

    #[inline]
    fn set_index(storage: &mut Storage<T, Self>, index: usize, value: T) {
        storage.rawvec[index] = value;
    }
}

impl<T> DeviceAPI<T> for DeviceCpuFaer where T: Clone {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_conversion() {
        let device_serial = DeviceCpuSerial {};
        let device_faer = DeviceCpuFaer::new(0);
        let a = Tensor::linspace(1.0, 5.0, 5, &device_serial);
        let b = a.into_device(&device_faer).unwrap();
        println!("{:?}", b);
        let a = Tensor::linspace(1.0, 5.0, 5, &device_serial);
        let b = a.view().into_device(&device_faer).unwrap();
        println!("{:?}", b);
    }
}
