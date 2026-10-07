//! diff tensor API: n-th order discrete differences along an axis.

use crate::prelude_dev::*;

/// Returns the n-th order discrete differences along the given axis.
///
/// See also [`diff`].
pub fn diff_f<R, T, B, D>(
    x: &TensorAny<R, T, B, D>,
    axis: impl TryInto<AxisIndex<isize>, Error: Into<Error>>,
    n: usize,
    prepend: Option<&TensorAny<R, T, B, D>>,
    append: Option<&TensorAny<R, T, B, D>>,
) -> Result<Tensor<T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataCloneAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    T: Clone + Default + core::ops::Sub<Output = T>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, D>
        + OpAssignAPI<T, Vec<usize>>
        + OpSubAPI<T, T, T, D>,
{
    let axis = axis.try_into().map_err(Into::into)?.into_inner();
    let device = x.device().clone();
    let _ = &device;
    let axis = rstsr_check_axis!(axis, x.ndim())?;

    // prepend/append must match x's shape outside the axis (tensor-level
    // validation before any concat)
    let check_side = |side: Option<&TensorAny<R, T, B, D>>, name: &str| -> Result<()> {
        if let Some(side) = side {
            rstsr_assert!(
                device.same_device(side.device()),
                DeviceMismatch,
                "diff requires {name} on the same device."
            )?;
            rstsr_assert_eq!(side.ndim(), x.ndim(), InvalidLayout, "diff requires {name} with the same ndim as x.")?;
            for i in 0..x.ndim() {
                if i != axis {
                    rstsr_assert_eq!(
                        side.shape()[i],
                        x.shape()[i],
                        InvalidLayout,
                        "diff requires {name} to match x's shape outside the axis."
                    )?;
                }
            }
        }
        Ok(())
    };
    check_side(prepend, "prepend")?;
    check_side(append, "append")?;

    // concat prepend + x + append along the axis (omitting None sides)
    let mut parts: Vec<&TensorAny<R, T, B, D>> = Vec::new();
    if let Some(pre) = prepend {
        parts.push(pre);
    }
    parts.push(x);
    if let Some(app) = append {
        parts.push(app);
    }
    let concatenated: Tensor<T, B, IxD> = if parts.len() == 1 {
        // single input: copy to a fresh owned tensor (gathers in K order
        // for non-compact layouts)
        TensorIntoOwnedAPI::into_owned(parts[0].view()).into_dim::<IxD>()
    } else {
        concat_f((parts, axis as isize))?
    };

    // n passes of adjacent-difference slicing along the axis: diff = hi - lo
    // via the generic binary subtraction on sliced views
    let mut current = concatenated;
    for _ in 0..n {
        let size = current.shape()[axis];
        if size == 0 {
            break;
        }
        // hi - lo via the device operator directly (views share the shape;
        // no broadcast needed)
        let lo_view = slice_pair_view(&current, axis, 0, size - 1)?;
        let hi_view = slice_pair_view(&current, axis, 1, size)?;
        let cur_shape: &Vec<usize> = current.shape();
        let mut shape_out: Vec<usize> = cur_shape.clone();
        shape_out[axis] = size - 1;
        let layout_out: Layout<IxD> = shape_out.new_contig(None, device.default_order());
        let (_, idx_max) = layout_out.bounds_index()?;
        let mut storage = device.uninit_impl(idx_max)?;
        let layout_out_d: Layout<D> = layout_out.to_dim()?;
        let hi_layout_d: Layout<D> = hi_view.layout().to_dim()?;
        let lo_layout_d: Layout<D> = lo_view.layout().to_dim()?;
        device.op_mutc_refa_refb(
            storage.raw_mut(),
            &layout_out_d,
            hi_view.raw(),
            &hi_layout_d,
            lo_view.raw(),
            &lo_layout_d,
        )?;
        // SAFETY: `op_mutc_refa_refb` above wrote every element of `layout_out`
        // (a broadcast-free same-shape elementwise op covers it exactly).
        let storage = unsafe { <B as DeviceCreationAnyAPI<T>>::assume_init_impl(storage)? };
        current = Tensor::new_f(storage, layout_out)?;
    }
    Ok(current)
}

/// Two slices of `t` along `axis` at `[start, stop)`, all other axes full.
fn slice_pair_view<'a, R, T, B, D>(
    t: &'a TensorAny<R, T, B, D>,
    axis: usize,
    start: usize,
    stop: usize,
) -> Result<TensorView<'a, T, B, IxD>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    let ndim = t.ndim();
    let mut indexers: Vec<Indexer> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        if i == axis {
            indexers.push(Indexer::Slice(Slice::new(start as isize, stop as isize, 1_isize)));
        } else {
            indexers.push(Indexer::Slice(Slice::new(None, None, None)));
        }
    }
    slice_f(t, indexers)
}

/// Returns the n-th order discrete differences along the given axis.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device
/// default orders. (Only the memory arrangement of the new tensor follows the
/// device default order.)
///
/// # Parameters
///
/// - `x`: [`&TensorAny<R, T, B, D>`](TensorAny): the input tensor.
///
/// - `axis`: TryInto [`AxisIndex<isize>`]: the axis along which to difference (negative counts from
///   the back).
///
/// - `n`: the number of difference passes; the axis shrinks by `n`. Passes on an already-empty axis
///   stop early (NumPy parity).
///
/// - `prepend` / `append`: optional [`&TensorAny<R, T, B, D>`](TensorAny) values concatenated along
///   `axis` before differencing; matching shapes outside `axis`, any size along it.
///
/// # Returns
///
/// - [`Tensor<T, B, IxD>`][`Tensor`]: owned tensor of the differenced shape.
///
/// # Examples
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 4, 9, 16], &device);
/// println!("{}", rt::diff((&a, -1, 1, None, None)));
/// // [ 3 5 7]
/// # let d1 = rt::diff((&a, -1, 1, None, None));
/// # assert_eq!(d1.to_vec(), vec![3, 5, 7]);
/// let b = rt::tensor_from_nested!([[1, 3, 6]], &device);
/// println!("{}", rt::diff((&b, -1, 1, None, None)));
/// // [[ 2 3]]
/// # let d2 = rt::diff((&b, -1, 1, None, None));
/// # assert_eq!(d2.reshape([-1]).to_vec(), vec![2, 3]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `diff(x, /, *, axis=-1, n=1, prepend=None, append=None)` ([`diff`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.diff.html))
/// - NumPy: `numpy.diff(a, n=1, axis=-1, prepend=None, append=None)` ([`numpy.diff`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html))
/// - RSTSR: `rt::diff((x, axis, n, prepend, append))`
///
/// # Panics
///
/// - Panics if `axis` is out of range, or `prepend`/`append` shapes mismatch outside `axis`. An `n`
///   pass on an empty axis stops early (the axis is already empty), matching NumPy.
///
/// For a fallible version, use [`diff_f`].
pub fn diff<Args, Inp>(args: Args) -> Args::Out
where
    Args: DiffAPI<Inp>,
{
    Args::diff(args)
}

/// API trait backing [`diff`].
pub trait DiffAPI<Inp> {
    type Out;

    fn diff_f(self) -> Result<Self::Out>;
    fn diff(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::diff_f(self).rstsr_unwrap()
    }
}

impl<'a, R, T, B, D, AArg> DiffAPI<()>
    for (&'a TensorAny<R, T, B, D>, AArg, usize, Option<&'a TensorAny<R, T, B, D>>, Option<&'a TensorAny<R, T, B, D>>)
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
    R: DataCloneAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    T: Clone + Default + core::ops::Sub<Output = T>,
    AArg: TryInto<AxisIndex<isize>, Error: Into<Error>>,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, D>
        + OpAssignAPI<T, Vec<usize>>
        + OpSubAPI<T, T, T, D>,
{
    type Out = Tensor<T, B, IxD>;

    fn diff_f(self) -> Result<Self::Out> {
        let (x, axis, n, prepend, append) = self;
        diff_f(x, axis, n, prepend, append)
    }
}
