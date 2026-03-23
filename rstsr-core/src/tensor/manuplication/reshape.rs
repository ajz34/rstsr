use crate::prelude_dev::*;

/* #region reshape args */

/// Reshape arguments.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReshapeArgs {
    /// The indexing order for **reading**. This also affects the order for writing.
    /// By default, the device's default order is used.
    pub order: Option<TensorOrder>,

    /// Whether to clone data when the new shape is not compatible with the original shape.
    ///
    /// - True: the tensor will always be copied, with order specified.
    /// - False: panic if the new shape is not compatible with the original shape.
    /// - None: the tensor will be copied only if necessary.
    pub copy: Option<bool>,
}

impl From<TensorOrder> for ReshapeArgs {
    fn from(order: TensorOrder) -> Self {
        Self { order: Some(order), copy: None }
    }
}

impl From<bool> for ReshapeArgs {
    fn from(copy: bool) -> Self {
        Self { order: None, copy: Some(copy) }
    }
}

#[duplicate_item(
    T1 T2 expr1 expr2;
    [TensorOrder        ] [bool        ] [Some(order)] [Some(copy)];
    [TensorOrder        ] [Option<bool>] [Some(order)] [copy      ];
    [Option<TensorOrder>] [bool        ] [order      ] [Some(copy)];
    [Option<TensorOrder>] [Option<bool>] [order      ] [copy      ];
)]
#[allow(clippy::redundant_field_names)]
impl From<(T1, T2)> for ReshapeArgs {
    fn from(args: (T1, T2)) -> Self {
        let (order, copy) = args;
        Self { order: expr1, copy: expr2 }
    }
}

impl From<Option<TensorOrder>> for ReshapeArgs {
    fn from(order: Option<TensorOrder>) -> Self {
        Self { order, copy: None }
    }
}

/* #endregion */

/* #region reshapeable */

/// Check if this tensor can be reshaped to a new shape without explicitly copying underlying data.
///
/// Please note this function returns `Result` instead of boolean.
///
/// - If shape not match, this function will raise error.
/// - If shape match but data need to be copied, return `Ok(None)`.
/// - If everything is fine, return `Ok(Some(layout_out))`.
///
/// For order, row-major and col-major behaves differently.
///
/// # See also
///
/// - [`reshape`]: the actual function for tensor reshaping.
/// - [`layout_reshapeable`]: The underlying function for checking layout compatibility for
///   reshaping, input by shape instead of tensor.
pub fn reshapeable_without_copy<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    order: Option<TensorOrder>,
) -> Result<Option<Layout<IxD>>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    let shape = reshape_substitute_negatives(shape.try_into().map_err(Into::into)?.as_ref(), tensor.size())?;
    let order = order.unwrap_or_else(|| tensor.device().default_order());
    layout_reshapeable(&tensor.layout().to_dim()?, &shape, order)
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Check if this tensor can be reshaped to a new shape without explicitly copying underlying
    /// data.
    ///
    /// See also [`reshapeable_without_copy`].
    pub fn reshapeable_without_copy(
        &self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        order: Option<TensorOrder>,
    ) -> Result<Option<Layout<IxD>>> {
        reshapeable_without_copy(self, shape, order)
    }
}

/* #endregion */

/* #region reshape_with_args */

/// Reshapes the given tensor to the specified shape, with argument specifying the order and whether
/// to copy data.
///
/// See also [`reshape_with_args`].
pub fn change_shape_with_args_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    args: impl Into<ReshapeArgs>,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    // own shape, this is cheap operation
    let shape_new = reshape_substitute_negatives(shape.try_into().map_err(Into::into)?.as_ref(), tensor.size())?;
    let ReshapeArgs { order, copy } = args.into();
    let order = order.unwrap_or(tensor.device().default_order());

    // rust 2021 does not allow chain if let
    if copy.is_none() || copy == Some(false) {
        if let Some(layout_new) = layout_reshapeable(&tensor.layout().to_dim()?, &shape_new, order)? {
            // shape does not need to be changed
            let (storage, _) = tensor.into_raw_parts();
            let layout = layout_new.into_dim::<IxD>()?;
            return unsafe { Ok(TensorBase::new_unchecked(storage, layout).into_cow()) };
        }
    }

    // if not allow copy, but layout is not compatible, raise error
    if copy == Some(false) {
        rstsr_raise!(
            InvalidValue,
            "copy is set to false in reshape, but layout {:?} is not compatible with shape {shape_new:?} and order {order:?}",
            tensor.layout(),
        )?
    }

    // clone underlying data by assign_arbitary
    // dev note: assign_arbitary_uninit depends on the iteration order of device
    let (storage, layout) = tensor.into_raw_parts();
    let device = storage.device();
    let layout_new = match order {
        RowMajor => shape_new.new_c_contig(None),
        ColMajor => shape_new.new_f_contig(None),
    };
    let mut storage_new = device.uninit_impl(layout_new.size())?;
    if device.default_order() == order {
        device.assign_arbitary_uninit(storage_new.raw_mut(), &layout_new, storage.raw(), &layout)?;
    } else {
        let mut device = device.clone();
        device.set_default_order(order);
        device.assign_arbitary_uninit(storage_new.raw_mut(), &layout_new, storage.raw(), &layout)?;
    }
    let storage_new = unsafe { B::assume_init_impl(storage_new)? };
    return unsafe { Ok(TensorBase::new_unchecked(storage_new, layout_new).into_cow()) };
}

/// Reshapes the given tensor to the specified shape, with argument specifying the order and whether
/// to copy data.
///
/// See also [`reshape_with_args`].
pub fn change_shape_with_args<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    args: impl Into<ReshapeArgs>,
) -> TensorCow<'a, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_with_args_f(tensor, shape, args).rstsr_unwrap()
}

/// Reshapes the given tensor to the specified shape, with argument specifying the order and whether
/// to copy data.
///
/// See also [`reshape_with_args`].
pub fn into_shape_with_args_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    args: impl Into<ReshapeArgs>,
) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    change_shape_with_args_f(tensor, shape, args).map(|v| v.into_owned())
}

/// Reshapes the given tensor to the specified shape, with argument specifying the order and whether
/// to copy data.
///
/// See also [`reshape_with_args`].
pub fn into_shape_with_args<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    args: impl Into<ReshapeArgs>,
) -> Tensor<T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_shape_with_args_f(tensor, shape, args).rstsr_unwrap()
}

/// Reshapes the given tensor to the specified shape, with argument specifying the order and whether
/// to copy data.
///
/// See also [`reshape_with_args`].
pub fn reshape_with_args_f<'a, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    args: impl Into<ReshapeArgs>,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_with_args_f(tensor.view(), shape, args)
}

/// Reshapes the given tensor to the specified shape, with argument specifying the order and whether
/// to copy data.
///
/// For usual users, please consider using [`reshape`] (take reference of tensor) or [`into_shape`]
/// (take ownership of tensor) instead, which are simpler interfaces to reshaping.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`TensorAny<R, T, B, D>`]
///
///   - The input tensor to be reshaped.
///
/// - `shape`: TryInto [`AxesIndex<isize>`]
///
///   - The new shape of the tensor.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///   - Overloads:
///
///     - integer: 1-D shape with a single dimension.
///     - vector/array/tuple of integers: N-D shape with N dimensions. For tuples,
///       mixed-signed/unsigned integers are supported.
///
/// - `args`: Into [`ReshapeArgs`]
///
///   - `order`: The indexing order for **reading** (similar to changing the default-order of
///     device). This also affects the order for writing. [`RowMajor`] and [`ColMajor`] are
///     supported. By default, the device's default order is used.
///   
///   - `copy`: Whether to clone data when the new shape is not compatible with the original shape.
///
///     - True: The tensor will always be copied. The output tensor will be contiguous with the
///       specified order.
///     - False: Panic if the new shape is not compatible with the original shape.
///     - None (default): The tensor will be copied only if necessary. If copied, the output tensor
///       will be contiguous with the specified order. Copy will be avoided if the new shape is
///       compatible with the original layout, even if the tensor is not contiguous.
///   
///   - Overloads:
///
///     - copy: [`bool`]
///     - copy: [`Option<bool>`] (None means default behavior)
///     - order: [`TensorOrder`]
///     - (order: [`TensorOrder`], copy: [`bool`])
///     - (order: [`TensorOrder`], copy: [`Option<bool>`])
///
/// # Examples
///
/// You can specify the order for reading the tensor by argument `order`.
///
/// Following is an example of row-major reshape. This is independent to the original default-layout
/// of device.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// println!("{a}");
/// // [[ 0 1 2]
/// //  [ 3 4 5]]
/// let a_row = rt::tensor_from_nested!([[0, 1], [2, 3], [4, 5]], &device);
/// println!("{a_row}");
/// // [[ 0 1]
/// //  [ 2 3]
/// //  [ 4 5]]
/// ```
///
/// And here is an example of col-major reshape.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
/// println!("{a}");
/// // [[ 0 1 2]
/// //  [ 3 4 5]]
/// let a_col = rt::tensor_from_nested!([[0, 4], [3, 2], [1, 5]], &device);
/// println!("{a_col}");
/// // [[ 0 4]
/// //  [ 3 2]
/// //  [ 1 5]]
/// ```
///
/// The following example shows that if `copy = false`, then an error will be raised when the new
/// shape is not compatible with the original shape. Given a strided tensor:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// assert_eq!(a.shape(), &[4, 6, 9]);
/// assert_eq!(a.stride(), &[72, 9, 1]);
/// assert!(!a.c_contig());
/// ```
///
/// The following example shows the reshaping does not explicitly clones data, and `copy = false`
/// does not raise error.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// // split a single dimension into multiple dimensions
/// assert!(a.reshape_with_args_f([2, 2, 6, 9], false).is_ok()); // (4, 6, 9) -> ([2, 2], 6, 9)
/// assert!(a.reshape_with_args_f([4, 3, 2, 9], false).is_ok()); // (4, 6, 9) -> (4, [3, 2], 9)
/// assert!(a.reshape_with_args_f([4, 2, 3, 3, 3], false).is_ok()); // (4, 6, 9) -> (4, [2, 3], [3, 3])
///
/// // merge contiguous dimensions into a single dimension
/// assert!(a.reshape_with_args_f([4, 54], false).is_ok()); // (4, 6, 9) -> (4, 6 * 9)
///
/// // merge contiguous dimensions and then split
/// assert!(a.reshape_with_args_f([4, 3, 6, 3], false).is_ok()); // (4, [6, 9]) -> (4, [3, 6, 3])
/// ```
///
/// However, the following example will raise error due to shape-incompatible. Using `copy = None`
/// or `copy = true` will work, but the data will be cloned.
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// // merge non-contiguous dimensions
/// assert!(a.reshape_with_args_f([24, 9], false).is_err()); // (4, 6, 9) -> (4 * 6, 9)
/// assert!(a.reshape_with_args_f([-1], false).is_err()); // (4, 6, 9) -> (4 * 6 * 9)
/// assert!(a.reshape_with_args_f([12, 2, 9], false).is_err()); // (4, 6, 9) -> (4 * [3, 2], 9)
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `reshape(x, /, shape, *, copy=None)` ([`reshape`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html))
/// - NumPy: `reshape(a, /, shape, order='C', *, copy=False)` ([`numpy.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)):
/// - RSTSR: `rt::reshape_with_args(tensor, shape, (order, copy))`
/// - RSTSR: `rt::reshape(tensor, shape)`
///
/// Please note that the `order` argument in RSTSR does not support NumPy's `'A'` (order='A' means
/// 'F' if the array is Fortran contiguous, 'C' otherwise in NumPy).
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - Python Array API standard: [`reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html)
/// - NumPy: [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
/// - ndarray: [`to_shape`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.to_shape)
///
/// ## Related functions in RSTSR
///
/// - [`reshape`]: simpler interface for reshaping.
/// - [`reshapeable_without_copy`]: Check whether the layout is compatible with the new shape.
/// - [`to_layout`]: Return a tensor with the specified layout.
/// - [`to_contig`]: Return an owned contiguous tensor.
///
/// ## Variants of this function
///
/// - [`reshape_with_args`] / [`reshape_with_args_f`]: Taking reference and returning Cow.
/// - [`into_shape_with_args`] / [`into_shape_with_args_f`]: Taking ownership and returning owned
///   tensor.
/// - [`change_shape_with_args`] / [`change_shape_with_args_f`]: Taking ownership and returning Cow.
/// - [`to_shape_with_args`] / [`to_shape_with_args_f`]: Alias to [`reshape_with_args`] /
///   [`reshape_with_args_f`].
/// - Associated methods on [`TensorAny`]:
///
///   - [`Tensor::reshape_with_args`] / [`Tensor::reshape_with_args_f`]
///   - [`Tensor::into_shape_with_args`] / [`Tensor::into_shape_with_args_f`]
///   - [`Tensor::change_shape_with_args`] / [`Tensor::change_shape_with_args_f`]
///   - [`Tensor::to_shape_with_args`] / [`Tensor::to_shape_with_args_f`]
pub fn reshape_with_args<'a, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    args: impl Into<ReshapeArgs>,
) -> TensorCow<'a, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    reshape_with_args_f(tensor, shape, args).rstsr_unwrap()
}

pub use reshape_with_args as to_shape_with_args;
pub use reshape_with_args_f as to_shape_with_args_f;

impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
    T: Clone,
{
    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn change_shape_with_args_f(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        change_shape_with_args_f(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn change_shape_with_args(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> TensorCow<'a, T, B, IxD> {
        change_shape_with_args(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn into_shape_with_args_f(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> Result<Tensor<T, B, IxD>>
    where
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape_with_args_f(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn into_shape_with_args(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> Tensor<T, B, IxD>
    where
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape_with_args(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn reshape_with_args(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> TensorCow<'a, T, B, IxD> {
        reshape_with_args(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn reshape_with_args_f(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        reshape_with_args_f(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn to_shape_with_args(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> TensorCow<'a, T, B, IxD> {
        to_shape_with_args(self, shape, args)
    }

    /// Reshapes the given tensor to the specified shape, with argument specifying the order and
    /// whether to copy data.
    ///
    /// # See also [`reshape_with_args`].
    pub fn to_shape_with_args_f(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
        args: impl Into<ReshapeArgs>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        to_shape_with_args_f(self, shape, args)
    }
}

/* #endregion */

/* #region reshape */

/// Reshapes the given tensor to the specified shape.
///
/// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
pub fn change_shape_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_with_args_f(tensor, shape, None)
}

/// Reshapes the given tensor to the specified shape.
///
/// This function is not intended to be used by usual users. Please consider using
/// [`reshape`] (take reference of tensor) or [`into_shape`] (take ownership of tensor)
/// instead.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`TensorAny<R, T, B, D>`]
///
///   - The input tensor to be reshaped.
///   - Ownership of input tensor is taken.
///
/// - `shape`: TryInto [`AxesIndex<isize>`]
///
///   - Position in the expanded axes where the new axis (or axes) is placed.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///
/// # Returns
///
/// - [`TensorCow<'a, T, B, IxD>`](TensorCow)
///
///   - The reshaped tensor.
///   - This function will try to avoid data cloning if possible.
///
///     - If layout-compatible, depending on whether the input tensor is owned or other cases,
///       either a view or owned tensor will be returned.
///     - If layout-not-compatible, an owned tensor will be returned, cloning the data.
///     - Cow (Clone-on-Write) semantics is used for representing either view or owned tensor.
///
/// This function is different to [`reshape`], in that it takes ownership of the input
/// tensor.
///
/// This function is also different to [`into_shape`], in that it may return a view, if the input
/// tensor also have the ownership of tensor view, and the layout is compatible.
///
/// # See also
///
/// Refer to [`reshape`] for more details and examples.
pub fn change_shape<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorCow<'a, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    change_shape_with_args(tensor, shape, None)
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
pub fn into_shape_f<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_shape_with_args_f(tensor, shape, None)
}

/// Reshapes the given tensor to the specified shape.
///
/// <div class="warning">
///
/// **Row/Column Major Notice**
///
/// This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.
///
/// </div>
///
/// # Parameters
///
/// - `tensor`: [`TensorAny<R, T, B, D>`]
///
///   - The input tensor to be reshaped.
///   - Ownership of input tensor is taken.
///
/// - `shape`: TryInto [`AxesIndex<isize>`]
///
///   - The new shape of the tensor.
///   - Can be a single integer, or a list/tuple of integers.
///   - Negative values are supported and indicate counting dimensions from the back.
///   - Overloads:
///     - integer: 1-D shape with a single dimension.
///     - vector/array/tuple of integers: N-D shape with N dimensions. For tuples,
///       mixed-signed/unsigned integers are supported.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`]
///
///   - The reshaped tensor.
///   - This function will try to avoid data cloning if possible, but with strict conditions:
///
///     - Layout-compatible after reshaping;
///     - Input tensor owns the underlying data (i.e., not a view);
///     - The input tensor is compact in memory (i.e., the underlying data does not have redundant
///       elements; size of tensor exactly matches the length of underlying data).
///
/// This function is different to [`change_shape`](change_shape()) and [`reshape`], in
/// that it takes ownership of the input tensor, and always returns an owned tensor.
///
/// # Examples
///
/// ```rust
/// use rstsr::prelude::*;
/// let a = rt::arange(6).into_shape([2, 3]);
/// ```
///
/// # Elaborated examples
///
/// Here is some showcases that demonstrate when data cloning happens or not. All examples are
/// row-major.
///
/// A first case is a tensor that is not fully contiguous (containing negative strides), but the
/// tensor is compact (size of tensor is the same to the length of underlying data). In this case,
/// if the new shape is compatible, no data cloning happens:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (-54, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]); the first dimension is reversed
/// let a = rt::arange((216, &device)).into_shape([4, 6, 9]).into_flip(0);
/// let a_ptr = a.raw().as_ptr();
/// let b = a.into_shape([4, 54]);
/// let b_ptr = b.raw().as_ptr();
/// assert_eq!(a_ptr, b_ptr); // contiguous dims merged, no data clone happened
/// ```
///
/// However, if the new shape is not compatible, data cloning will happen:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (-54, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]); the first dimension is reversed
/// let a = rt::arange((216, &device)).into_shape([4, 6, 9]).into_flip(0);
/// let a_ptr = a.raw().as_ptr();
/// let b = a.into_shape([24, 9]);
/// let b_ptr = b.raw().as_ptr();
/// assert_ne!(a_ptr, b_ptr); // layout not compatible, data clone happened
/// ```
///
/// Another case is a tensor that is not compact (size of tensor is less than the length of
/// underlying data). In this case, even if the new shape is compatible, data cloning will happen:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// // shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
/// // contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
/// let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
/// let a_ptr = a.raw().as_ptr();
/// let b = a.into_shape([4, 54]);
/// let b_ptr = b.raw().as_ptr();
/// assert_ne!(a_ptr, b_ptr); // layout-compatible, but input tensor is not compact (216 < 288)
/// ```
///
/// # See also
///
/// Refer to [`reshape`] for more details and examples.
pub fn into_shape<'a, R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Tensor<T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
{
    into_shape_with_args(tensor, shape, None)
}

/// Reshapes the given tensor to the specified shape.
///
/// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
pub fn reshape_f<'a, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> Result<TensorCow<'a, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    reshape_with_args_f(tensor, shape, None)
}

#[doc = include_str!("doc_reshape.md")]
pub fn reshape<'a, R, T, B, D>(
    tensor: &'a TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
) -> TensorCow<'a, T, B, IxD>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
{
    reshape_with_args(tensor, shape, None)
}

pub use reshape as to_shape;
pub use reshape_f as to_shape_f;

/// Reshapes the given tensor to the specified shape.
///
/// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
    T: Clone,
{
    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn change_shape_f(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        change_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn change_shape(self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorCow<'a, T, B, IxD> {
        change_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn into_shape_f(self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> Result<Tensor<T, B, IxD>>
    where
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn into_shape(self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> Tensor<T, B, IxD>
    where
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn to_shape_f(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        to_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn to_shape(&'a self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorCow<'a, T, B, IxD> {
        to_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn reshape_f(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        reshape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`], [`into_shape`], [`change_shape`] and [`reshape_with_args`].
    pub fn reshape(&'a self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorCow<'a, T, B, IxD> {
        reshape(self, shape)
    }
}

/* #endregion */

/* #region macro */

#[doc = include_str!("doc_reshape.md")]
#[macro_export]
macro_rules! reshape {
    // tensor, shape, order, copy
    ($tensor:expr, shape = $shape:expr, order = $order:expr, copy = $copy:expr) => {
        $tensor.reshape_with_args($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, copy = $copy:expr) => {
        $tensor.reshape_with_args($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, $order:expr, copy = $copy:expr) => {
        $tensor.reshape_with_args($shape, ($order, $copy))
    };

    // tensor, shape, copy
    ($tensor:expr, shape = $shape:expr, copy = $copy:expr) => {
        $tensor.reshape_with_args($shape, (None, $copy))
    };
    ($tensor:expr, $shape:expr, copy = $copy:expr) => {
        $tensor.reshape_with_args($shape, (None, $copy))
    };

    // tensor, shape, order
    ($tensor:expr, shape = $shape:expr, order = $order:expr) => {
        $tensor.reshape_with_args($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, order = $order:expr) => {
        $tensor.reshape_with_args($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, $order:expr) => {
        $tensor.reshape_with_args($shape, ($order, None))
    };

    // tensor, shape
    ($tensor:expr, shape = $shape:expr) => {
        $tensor.reshape($shape)
    };
    ($tensor:expr, $shape:expr) => {
        $tensor.reshape($shape)
    };
}

/* #endregion */
