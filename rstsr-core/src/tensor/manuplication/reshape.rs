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
/// See also [`reshape`].
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
/// See also [`reshape`].
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
/// See also [`reshape`].
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
/// See also [`reshape`].
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
/// See also [`reshape`].
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
/// See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
    /// # See also [`reshape`].
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
/// # See also [`reshape`].
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
/// # See also [`reshape`].
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
/// # See also [`reshape`].
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
/// # See also [`reshape`].
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
/// # See also [`reshape`].
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
/// # See also [`reshape`].
impl<'a, R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, D>,
    T: Clone,
{
    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn change_shape_f(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        change_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn change_shape(self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorCow<'a, T, B, IxD> {
        change_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn into_shape_f(self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> Result<Tensor<T, B, IxD>>
    where
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn into_shape(self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> Tensor<T, B, IxD>
    where
        <B as DeviceRawAPI<T>>::Raw: Clone + 'a,
        B: OpAssignAPI<T, IxD>,
    {
        into_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn to_shape_f(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        to_shape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn to_shape(&'a self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorCow<'a, T, B, IxD> {
        to_shape(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn reshape_f(
        &'a self,
        shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>,
    ) -> Result<TensorCow<'a, T, B, IxD>> {
        reshape_f(self, shape)
    }

    /// Reshapes the given tensor to the specified shape.
    ///
    /// # See also [`reshape`].
    pub fn reshape(&'a self, shape: impl TryInto<AxesIndex<isize>, Error: Into<Error>>) -> TensorCow<'a, T, B, IxD> {
        reshape(self, shape)
    }
}

/* #endregion */

/* #region macro */

#[doc = include_str!("doc_reshape.md")]
#[macro_export]
macro_rules! reshape {
    /* #region change fallible */
    // tensor, shape, order, copy
    ($tensor:expr, shape = $shape:expr, order = $order:expr, copy = $copy:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, copy = $copy:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, $order:expr, copy = $copy:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, ($order, $copy))
    };

    // tensor, shape, copy
    ($tensor:expr, shape = $shape:expr, copy = $copy:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, (None, $copy))
    };
    ($tensor:expr, $shape:expr, copy = $copy:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, (None, $copy))
    };

    // tensor, shape, order
    ($tensor:expr, shape = $shape:expr, order = $order:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, $order:expr, change, fallible) => {
        $tensor.change_shape_with_args_f($shape, ($order, None))
    };
    /* #endregion */

    /* #region into fallible */
    // tensor, shape, order, copy
    ($tensor:expr, shape = $shape:expr, order = $order:expr, copy = $copy:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, copy = $copy:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, $order:expr, copy = $copy:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, ($order, $copy))
    };

    // tensor, shape, copy
    ($tensor:expr, shape = $shape:expr, copy = $copy:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, (None, $copy))
    };
    ($tensor:expr, $shape:expr, copy = $copy:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, (None, $copy))
    };

    // tensor, shape, order
    ($tensor:expr, shape = $shape:expr, order = $order:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, $order:expr, into, fallible) => {
        $tensor.into_shape_with_args_f($shape, ($order, None))
    };
    /* #endregion */

    /* #region to fallible */
    // tensor, shape, order, copy
    ($tensor:expr, shape = $shape:expr, order = $order:expr, copy = $copy:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, copy = $copy:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, $order:expr, copy = $copy:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, ($order, $copy))
    };

    // tensor, shape, copy
    ($tensor:expr, shape = $shape:expr, copy = $copy:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, (None, $copy))
    };
    ($tensor:expr, $shape:expr, copy = $copy:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, (None, $copy))
    };

    // tensor, shape, order
    ($tensor:expr, shape = $shape:expr, order = $order:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, $order:expr, fallible) => {
        $tensor.reshape_with_args_f($shape, ($order, None))
    };

    // tensor, shape
    ($tensor:expr, shape = $shape:expr, fallible) => {
        $tensor.reshape_f($shape)
    };
    ($tensor:expr, $shape:expr, fallible) => {
        $tensor.reshape_f($shape)
    };
    /* #endregion */

    /* #region change */
    // tensor, shape, order, copy
    ($tensor:expr, shape = $shape:expr, order = $order:expr, copy = $copy:expr, change) => {
        $tensor.change_shape_with_args($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, copy = $copy:expr, change) => {
        $tensor.change_shape_with_args($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, $order:expr, copy = $copy:expr, change) => {
        $tensor.change_shape_with_args($shape, ($order, $copy))
    };

    // tensor, shape, copy
    ($tensor:expr, shape = $shape:expr, copy = $copy:expr, change) => {
        $tensor.change_shape_with_args($shape, (None, $copy))
    };
    ($tensor:expr, $shape:expr, copy = $copy:expr, change) => {
        $tensor.change_shape_with_args($shape, (None, $copy))
    };

    // tensor, shape, order
    ($tensor:expr, shape = $shape:expr, order = $order:expr, change) => {
        $tensor.change_shape_with_args($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, change) => {
        $tensor.change_shape_with_args($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, $order:expr, change) => {
        $tensor.change_shape_with_args($shape, ($order, None))
    };
    /* #endregion */

    /* #region into */
    // tensor, shape, order, copy
    ($tensor:expr, shape = $shape:expr, order = $order:expr, copy = $copy:expr, into) => {
        $tensor.into_shape_with_args($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, copy = $copy:expr, into) => {
        $tensor.into_shape_with_args($shape, ($order, $copy))
    };
    ($tensor:expr, $shape:expr, $order:expr, copy = $copy:expr, into) => {
        $tensor.into_shape_with_args($shape, ($order, $copy))
    };

    // tensor, shape, copy
    ($tensor:expr, shape = $shape:expr, copy = $copy:expr, into) => {
        $tensor.into_shape_with_args($shape, (None, $copy))
    };
    ($tensor:expr, $shape:expr, copy = $copy:expr, into) => {
        $tensor.into_shape_with_args($shape, (None, $copy))
    };

    // tensor, shape, order
    ($tensor:expr, shape = $shape:expr, order = $order:expr, into) => {
        $tensor.into_shape_with_args($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, order = $order:expr, into) => {
        $tensor.into_shape_with_args($shape, ($order, None))
    };
    ($tensor:expr, $shape:expr, $order:expr, into) => {
        $tensor.into_shape_with_args($shape, ($order, None))
    };
    /* #endregion */

    /* #region to */
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
    }; /* #endregion */
}

/* #endregion */
