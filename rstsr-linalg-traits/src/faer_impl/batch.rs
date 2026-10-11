//! Shared batch walk for the stacked (n-dimensional) faer linalg entries.
//!
//! Mirrors the per-slice loop of [`crate::faer_impl::cholesky`]: the two matrix
//! axes are the last two under `RowMajor` and the first two under `ColMajor`
//! (the device default order); every other axis is a batch axis. Each input
//! matrix slice is paired with the corresponding slice of one or more
//! pre-allocated outputs, in the same batch order.

use faer::prelude::*;
use rstsr_core::prelude_dev::*;

/// Run `f` with faer's global parallelism pinned to the device's thread pool,
/// restoring the previous setting afterwards.
pub(crate) fn with_parallel<T>(device: &DeviceFaer, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    if let Some(pool) = pool {
        faer::set_global_parallelism(Par::rayon(pool.current_num_threads()));
    }
    let result = f();
    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig);
    }
    result
}

/// Shape of a stacked result whose per-slice shape is `inner_shape`: batch then
/// inner under `RowMajor`, inner then batch under `ColMajor`.
pub(crate) fn stack_shape(batch_shape: &[usize], inner_shape: &[usize], order: FlagOrder) -> Vec<usize> {
    match order {
        RowMajor => [batch_shape, inner_shape].concat(),
        ColMajor => [inner_shape, batch_shape].concat(),
    }
}

/// Split a stacked shape into its batch shape and 2-D matrix shape. The matrix
/// axes are the last two under `RowMajor` and the first two under `ColMajor`.
pub(crate) fn batch_and_matrix_shape(shape: &[usize], order: FlagOrder) -> Result<(Vec<usize>, [usize; 2])> {
    let ndim = shape.len();
    rstsr_assert!(ndim >= 2, InvalidLayout, "linalg: expected at least 2 dimensions, got {ndim}")?;
    Ok(match order {
        RowMajor => (shape[..ndim - 2].to_vec(), [shape[ndim - 2], shape[ndim - 1]]),
        ColMajor => (shape[2..].to_vec(), [shape[0], shape[1]]),
    })
}

/// Split a stacked layout into its (batch, inner) parts. The `inner_ndim` axes
/// are trailing under `RowMajor`, leading under `ColMajor`.
fn split_inner(layout: &Layout<IxD>, order: FlagOrder, inner_ndim: usize) -> Result<(Layout<IxD>, Layout<IxD>)> {
    let axis = inner_ndim as isize;
    match order {
        RowMajor => layout.dim_split_at(-axis),
        ColMajor => {
            let (inner, batch) = layout.dim_split_at(axis)?;
            Ok((batch, inner))
        },
    }
}

/// Batch-offset iterator and inner layout of a stacked tensor.
fn batch_parts(layout: &Layout<IxD>, order: FlagOrder, inner_ndim: usize) -> Result<(IterLayout<IxD>, Layout<IxD>)> {
    let (batch, inner) = split_inner(layout, order, inner_ndim)?;
    Ok((IterLayout::new(&batch, TensorIterOrder::C)?, inner))
}

/// Apply `f` to every matrix slice of the stack `a`, writing results into the
/// pre-allocated `out`. `out` must share `a`'s batch shape and hold one inner
/// slice per matrix.
pub(crate) fn map_stack_slices<TA, TB, DO, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    mut out: TensorMut<'_, TB, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<()>
where
    DO: DimAPI,
    F: FnMut(TensorView<'_, TA, DeviceFaer, Ix2>, TensorMut<'_, TB, DeviceFaer, DO>) -> Result<()>,
{
    let inner_ndim = out.ndim() + 2 - a.ndim();
    let (a_iters, a_inner) = batch_parts(a.layout(), order, 2)?;
    let (o_iters, o_inner) = batch_parts(out.layout(), order, inner_ndim)?;
    for (off_a, off_o) in izip!(a_iters, o_iters) {
        let mut a_i = a_inner.clone().into_dim::<Ix2>()?;
        let mut o_i = o_inner.clone().into_dim::<DO>()?;
        unsafe { a_i.set_offset(off_a) };
        unsafe { o_i.set_offset(off_o) };
        let a_slice = {
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, a_i) }
        };
        let o_slice = {
            let (storage, _) = out.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o_i) }
        };
        f(a_slice, o_slice)?;
    }
    Ok(())
}

/// As [`map_stack_slices`], for two outputs computed together from one slice.
pub(crate) fn map_stack_slices2<TA, TB1, DB1, TB2, DB2, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    mut out1: TensorMut<'_, TB1, DeviceFaer, IxD>,
    mut out2: TensorMut<'_, TB2, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<()>
where
    DB1: DimAPI,
    DB2: DimAPI,
    F: FnMut(
        TensorView<'_, TA, DeviceFaer, Ix2>,
        TensorMut<'_, TB1, DeviceFaer, DB1>,
        TensorMut<'_, TB2, DeviceFaer, DB2>,
    ) -> Result<()>,
{
    let inner_ndim1 = out1.ndim() + 2 - a.ndim();
    let inner_ndim2 = out2.ndim() + 2 - a.ndim();
    let (a_iters, a_inner) = batch_parts(a.layout(), order, 2)?;
    let (o1_iters, o1_inner) = batch_parts(out1.layout(), order, inner_ndim1)?;
    let (o2_iters, o2_inner) = batch_parts(out2.layout(), order, inner_ndim2)?;
    for (off_a, off1, off2) in izip!(a_iters, o1_iters, o2_iters) {
        let mut a_i = a_inner.clone().into_dim::<Ix2>()?;
        let mut o1_i = o1_inner.clone().into_dim::<DB1>()?;
        let mut o2_i = o2_inner.clone().into_dim::<DB2>()?;
        unsafe { a_i.set_offset(off_a) };
        unsafe { o1_i.set_offset(off1) };
        unsafe { o2_i.set_offset(off2) };
        let a_slice = {
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, a_i) }
        };
        let o1_slice = {
            let (storage, _) = out1.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o1_i) }
        };
        let o2_slice = {
            let (storage, _) = out2.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o2_i) }
        };
        f(a_slice, o1_slice, o2_slice)?;
    }
    Ok(())
}

/// As [`map_stack_slices`], for three outputs computed together from one slice.
pub(crate) fn map_stack_slices3<TA, TB1, DB1, TB2, DB2, TB3, DB3, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    mut out1: TensorMut<'_, TB1, DeviceFaer, IxD>,
    mut out2: TensorMut<'_, TB2, DeviceFaer, IxD>,
    mut out3: TensorMut<'_, TB3, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<()>
where
    DB1: DimAPI,
    DB2: DimAPI,
    DB3: DimAPI,
    F: FnMut(
        TensorView<'_, TA, DeviceFaer, Ix2>,
        TensorMut<'_, TB1, DeviceFaer, DB1>,
        TensorMut<'_, TB2, DeviceFaer, DB2>,
        TensorMut<'_, TB3, DeviceFaer, DB3>,
    ) -> Result<()>,
{
    let inner_ndim1 = out1.ndim() + 2 - a.ndim();
    let inner_ndim2 = out2.ndim() + 2 - a.ndim();
    let inner_ndim3 = out3.ndim() + 2 - a.ndim();
    let (a_iters, a_inner) = batch_parts(a.layout(), order, 2)?;
    let (o1_iters, o1_inner) = batch_parts(out1.layout(), order, inner_ndim1)?;
    let (o2_iters, o2_inner) = batch_parts(out2.layout(), order, inner_ndim2)?;
    let (o3_iters, o3_inner) = batch_parts(out3.layout(), order, inner_ndim3)?;
    for (off_a, off1, off2, off3) in izip!(a_iters, o1_iters, o2_iters, o3_iters) {
        let mut a_i = a_inner.clone().into_dim::<Ix2>()?;
        let mut o1_i = o1_inner.clone().into_dim::<DB1>()?;
        let mut o2_i = o2_inner.clone().into_dim::<DB2>()?;
        let mut o3_i = o3_inner.clone().into_dim::<DB3>()?;
        unsafe { a_i.set_offset(off_a) };
        unsafe { o1_i.set_offset(off1) };
        unsafe { o2_i.set_offset(off2) };
        unsafe { o3_i.set_offset(off3) };
        let a_slice = {
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, a_i) }
        };
        let o1_slice = {
            let (storage, _) = out1.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o1_i) }
        };
        let o2_slice = {
            let (storage, _) = out2.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o2_i) }
        };
        let o3_slice = {
            let (storage, _) = out3.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o3_i) }
        };
        f(a_slice, o1_slice, o2_slice, o3_slice)?;
    }
    Ok(())
}

/// As [`map_stack_slices`], for scalar-per-matrix outputs: `out1`/`out2` have
/// the batch shape and `f` writes one scalar into each per matrix.
pub(crate) fn map_stack_scalars2<TA, TB1, TB2, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    mut out1: TensorMut<'_, TB1, DeviceFaer, IxD>,
    mut out2: TensorMut<'_, TB2, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<()>
where
    F: FnMut(TensorView<'_, TA, DeviceFaer, Ix2>, &mut TB1, &mut TB2) -> Result<()>,
{
    let (a_batch, a_inner) = split_inner(a.layout(), order, 2)?;
    let a_iters = IterLayout::new(&a_batch, TensorIterOrder::C)?;
    let o1_iters = IterLayout::new(out1.layout(), TensorIterOrder::C)?;
    let o2_iters = IterLayout::new(out2.layout(), TensorIterOrder::C)?;
    for (off_a, off1, off2) in izip!(a_iters, o1_iters, o2_iters) {
        let a_slice = {
            let mut inner = a_inner.clone().into_dim::<Ix2>()?;
            unsafe { inner.set_offset(off_a) };
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, inner) }
        };
        f(a_slice, &mut out1.raw_mut()[off1], &mut out2.raw_mut()[off2])?;
    }
    Ok(())
}
