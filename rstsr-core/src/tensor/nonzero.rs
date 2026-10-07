//! Nonzero tensor API: [`nonzero`] returning one index tensor per dimension.

use crate::prelude_dev::*;

/// Returns the indices of the elements that are non-zero.
///
/// See also [`nonzero`].
pub fn nonzero_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<Vec<Tensor<usize, B, IxD>>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpNonzeroAPI<T, D>,
{
    let tensor = tensor.view();
    let device = tensor.device().clone();
    let ndim = tensor.ndim();
    rstsr_assert!(ndim > 0, InvalidLayout, "nonzero requires ndim > 0.")?;

    // pass 1: count (the output length is data-dependent)
    let count = device.nonzero_count(tensor.raw(), tensor.layout())?;
    // pass 2: one output buffer per dimension, filled with the coordinates of
    // every nonzero element in row-major visit order
    let mut storages = (0..ndim).map(|_| device.uninit_impl(count)).collect::<Result<Vec<_>>>()?;
    {
        let mut buffers: Vec<_> = storages.iter_mut().map(|s| s.raw_mut()).collect();
        device.nonzero_fill(&mut buffers, tensor.raw(), tensor.layout())?;
    }
    let layout = vec![count].new_contig(None, device.default_order());
    storages
        .into_iter()
        .map(|storage| {
            // SAFETY: `nonzero_fill` wrote exactly `count` coordinates into
            // every buffer.
            let storage = unsafe { <B as DeviceCreationAnyAPI<usize>>::assume_init_impl(storage)? };
            Tensor::new_f(storage, layout.clone())
        })
        .collect()
}

/// Returns the indices of the elements that are non-zero, one 1-D index
/// tensor per dimension.
///
/// Together the returned tensors locate every element that compares unequal
/// to zero (`!= 0`; booleans: `true`; complex: either component nonzero;
/// NaN is nonzero). The k-th tensor holds the k-th coordinate of each
/// nonzero element, in strict row-major element order. All index tensors
/// have dtype [`usize`] and equal (data-dependent) lengths; a 0-d input
/// raises (there is no axis to index).
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (The element visit order is row-major regardless of the
/// device default order; the memory arrangement of the new tensors follows
/// the device default order.)
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor (ndim ≥ 1, any dtype).
///
/// # Returns
///
/// - `Vec<[Tensor<usize, B, IxD>][Tensor]>`: one coordinate tensor per dimension, each 1-D of
///   length = the number of nonzero elements.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 0, 2], [0, 3, 0]], &device);
/// let coords = rt::nonzero(&a);
/// println!("{}", coords[0]);
/// // [ 0 0 1]
/// println!("{}", coords[1]);
/// // [ 0 2 1]
/// # assert_eq!(coords[0].to_vec(), vec![0, 0, 1]);
/// # assert_eq!(coords[1].to_vec(), vec![0, 2, 1]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `nonzero(x, /)` returning a tuple of rank(`x`) index arrays ([`nonzero`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.nonzero.html))
/// - NumPy: `numpy.nonzero(x)`
/// - RSTSR: `rt::nonzero(tensor)` (a `Vec` of tensors; the python wrapper builds the tuple)
///
/// # Panics
///
/// - Panics if the input is 0-dimensional.
///
/// For a fallible version, use [`nonzero_f`].
///
/// # See also
///
/// ## Variants of this function
///
/// - [`nonzero_f`]: fallible version.
pub fn nonzero<Inp>(inp: Inp) -> Inp::Out
where
    Inp: NonzeroAPI,
{
    Inp::nonzero(inp)
}

/// API trait backing [`nonzero`].
pub trait NonzeroAPI {
    type Out;

    fn nonzero_f(self) -> Result<Self::Out>;
    fn nonzero(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::nonzero_f(self).rstsr_unwrap()
    }
}

impl<R, T, B, D> NonzeroAPI for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceAPI<usize>
        + DeviceRawAPI<MaybeUninit<usize>>
        + DeviceCreationAnyAPI<usize>
        + OpNonzeroAPI<T, D>,
{
    type Out = Vec<Tensor<usize, B, IxD>>;

    fn nonzero_f(self) -> Result<Self::Out> {
        nonzero_f(self)
    }
}
