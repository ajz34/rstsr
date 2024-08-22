use crate::cpu_backend::device::CpuDevice;
use crate::prelude_dev::*;

pub trait TensorBaseAPI {}

#[derive(Clone)]
pub struct TensorBase<R, D>
where
    D: DimAPI,
{
    pub(crate) data: R,
    pub(crate) layout: Layout<D>,
}

impl<R, D> TensorBaseAPI for TensorBase<R, D> where D: DimAPI {}

/// Basic definitions for tensor object.
impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{
    /// Initialize tensor object.
    ///
    /// # Safety
    ///
    /// This function will not check whether data meets the standard of
    /// [Storage<T, B>], or whether layout may exceed pointer bounds of data.
    pub unsafe fn new_unchecked(data: R, layout: Layout<D>) -> Self {
        Self { data, layout }
    }

    pub fn data(&self) -> &R {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut R {
        &mut self.data
    }

    pub fn layout(&self) -> &Layout<D> {
        &self.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.layout().shape_ref().as_ref()
    }

    pub fn raw_shape(&self) -> D {
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

impl<T, D, B, R> TensorBase<R, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    R: DataAPI<Data = Storage<T, B>>,
{
    pub fn new(data: R, layout: Layout<D>) -> Result<Self> {
        // check stride sanity
        layout.check_strides()?;

        // check pointer exceed
        let len_data = data.as_storage().len();
        let (_, idx_max) = layout.bounds_index()?;
        rstsr_pattern!(idx_max, len_data.., ValueOutOfRange)?;
        return Ok(Self { data, layout });
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
