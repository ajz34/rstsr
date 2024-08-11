use crate::cpu_backend::device::CpuDevice;
use crate::layout::{DimAPI, Layout};
use crate::storage::{DataAPI, DataOwned, DataRef, DataRefMut, Storage, StorageBaseAPI};
use crate::{Error, Result};

#[derive(Clone)]
pub struct TensorBase<S, D>
where
    D: DimAPI,
{
    data: S,
    layout: Layout<D>,
}

/// Basic definitions for tensor object.
impl<S, D> TensorBase<S, D>
where
    D: DimAPI,
{
    /// Initialize tensor object.
    ///
    /// # Safety
    ///
    /// This function will not check whether data meets the standard of
    /// [Storage<T, B>], or whether layout may exceed pointer bounds of data.
    pub unsafe fn new_unchecked(data: S, layout: Layout<D>) -> Self {
        Self { data, layout }
    }

    pub fn new(data: S, layout: Layout<D>) -> Result<Self>
    where
        S: DataAPI,
        S::Data: StorageBaseAPI,
        D: DimAPI,
    {
        // check stride sanity
        layout.check_strides()?;

        // check pointer exceed
        let len_data = data.as_storage().len();
        let (_, idx_max) = layout.bounds_index()?;
        if idx_max < len_data {
            return Err(Error::IndexOutOfBound {
                index: idx_max as isize,
                bound: len_data as isize,
            });
        }

        return Ok(Self { data, layout });
    }

    pub fn data(&self) -> &S {
        &self.data
    }

    pub fn layout(&self) -> &Layout<D> {
        &self.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.layout().shape_ref().as_ref()
    }

    pub fn raw_shape(&self) -> D::Shape {
        self.layout().shape().0
    }

    pub fn stride(&self) -> &[isize] {
        self.layout().stride_ref().as_ref()
    }

    pub fn raw_stride(&self) -> D::Stride {
        self.layout().stride().0
    }

    pub fn offset(&self) -> usize {
        self.layout().offset()
    }

    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    pub fn size(&self) -> usize {
        self.layout().size()
    }
}

pub type Tensor<T, D, B = CpuDevice> = TensorBase<DataOwned<Storage<T, B>>, D>;
pub type TensorView<'a, T, D, B = CpuDevice> = TensorBase<DataRef<'a, Storage<T, B>>, D>;
pub type TensorViewMut<'a, T, D, B = CpuDevice> = TensorBase<DataRefMut<'a, Storage<T, B>>, D>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        use crate::layout::*;
        let a = Tensor::<f64, Ix<2>> {
            data: Storage { rawvec: vec![1.12345, 2.0], device: CpuDevice }.into(),
            layout: [1, 2].new_c_contig(0),
        };
        println!("{a:6.3?}");
    }
}
