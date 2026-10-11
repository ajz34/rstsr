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

/// As [`map_stack_slices`], for two read-only inputs and two outputs computed
/// together from matching slices (generalized `eigh`: operands `a`, `b` ->
/// eigenvalues and eigenvectors).
pub(crate) fn map_stack_slices2x2<TA, TB, TC1, DC1, TC2, DC2, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    b: TensorView<'_, TB, DeviceFaer, IxD>,
    mut out1: TensorMut<'_, TC1, DeviceFaer, IxD>,
    mut out2: TensorMut<'_, TC2, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<()>
where
    DC1: DimAPI,
    DC2: DimAPI,
    F: FnMut(
        TensorView<'_, TA, DeviceFaer, Ix2>,
        TensorView<'_, TB, DeviceFaer, Ix2>,
        TensorMut<'_, TC1, DeviceFaer, DC1>,
        TensorMut<'_, TC2, DeviceFaer, DC2>,
    ) -> Result<()>,
{
    let inner1 = out1.ndim() + 2 - a.ndim();
    let inner2 = out2.ndim() + 2 - a.ndim();
    let (a_iters, a_inner) = batch_parts(a.layout(), order, 2)?;
    let (b_iters, b_inner) = batch_parts(b.layout(), order, 2)?;
    let (o1_iters, o1_inner) = batch_parts(out1.layout(), order, inner1)?;
    let (o2_iters, o2_inner) = batch_parts(out2.layout(), order, inner2)?;
    for (off_a, off_b, off1, off2) in izip!(a_iters, b_iters, o1_iters, o2_iters) {
        let mut a_i = a_inner.clone().into_dim::<Ix2>()?;
        let mut b_i = b_inner.clone().into_dim::<Ix2>()?;
        let mut o1_i = o1_inner.clone().into_dim::<DC1>()?;
        let mut o2_i = o2_inner.clone().into_dim::<DC2>()?;
        unsafe { a_i.set_offset(off_a) };
        unsafe { b_i.set_offset(off_b) };
        unsafe { o1_i.set_offset(off1) };
        unsafe { o2_i.set_offset(off2) };
        let a_slice = {
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, a_i) }
        };
        let b_slice = {
            let (storage, _) = b.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, b_i) }
        };
        let o1_slice = {
            let (storage, _) = out1.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o1_i) }
        };
        let o2_slice = {
            let (storage, _) = out2.view_mut().into_raw_parts();
            unsafe { TensorMut::new_unchecked(storage, o2_i) }
        };
        f(a_slice, b_slice, o1_slice, o2_slice)?;
    }
    Ok(())
}

/// Solve every system of the stack `a` (`(..., M, M)`) in place into the
/// matching slice of `b` (`(..., M, K)`, or `(..., M)` for the vector form,
/// widened to `(..., M, 1)` per slice). `a` is read-only; `b` is the (mutable)
/// right-hand side, so an owned/contiguous `b` is solved without any copy. The
/// `b` is returned so a caller wrapping it in
/// [`TensorMutable::ToBeCloned`] can finalize with `clone_to_mut`.
pub(crate) fn map_stack_slices_inplace<'b, TA, TB, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    mut b: TensorMutable<'b, TB, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<TensorMutable<'b, TB, DeviceFaer, IxD>>
where
    F: FnMut(TensorView<'_, TA, DeviceFaer, Ix2>, TensorMut<'_, TB, DeviceFaer, Ix2>) -> Result<()>,
{
    // inner ndim of b: 2 for a matrix stack, 1 for a vector stack
    let b_inner_ndim = b.view().ndim() + 2 - a.ndim();
    let (a_iters, a_inner) = batch_parts(a.layout(), order, 2)?;
    let (b_iters, b_inner) = batch_parts(b.view().layout(), order, b_inner_ndim)?;
    for (off_a, off_b) in izip!(a_iters, b_iters) {
        let mut a_i = a_inner.clone().into_dim::<Ix2>()?;
        let mut b_i = b_inner.clone().into_dim::<IxD>()?;
        unsafe { a_i.set_offset(off_a) };
        unsafe { b_i.set_offset(off_b) };
        let a_slice = {
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, a_i) }
        };
        let (storage, _) = b.view_mut().into_raw_parts();
        let mut b_slice = unsafe { TensorMut::new_unchecked(storage, b_i) };
        // a 1-D vector slice is widened to (M, 1)
        match b_inner_ndim {
            1 => f(a_slice, b_slice.i_mut((.., None)).into_dim::<Ix2>())?,
            _ => f(a_slice, b_slice.view_mut().into_dim::<Ix2>())?,
        }
    }
    Ok(b)
}

/// Broadcast plan for a batched `solve`: `a` is `(..., M, M)`, `b` is
/// `(..., M, K)` or the vector form `(..., M)`, and the two batch shapes
/// broadcast to `batch_out`. Matrix axes follow `order` (last two under
/// `RowMajor`, first two under `ColMajor`).
pub(crate) struct SolvePlan {
    pub batch_out: Vec<usize>,
    pub m: usize,
    pub k: usize,
    pub is_vec: bool,
}

impl SolvePlan {
    /// Shape of the broadcast solution: `batch_out ++ [M, K]` (`batch_out ++ [M]`
    /// for the vector form), placed per `order` (batch then inner under
    /// `RowMajor`, inner then batch under `ColMajor`).
    pub fn out_shape(&self, order: FlagOrder) -> Vec<usize> {
        let inner = if self.is_vec { vec![self.m] } else { vec![self.m, self.k] };
        stack_shape(&self.batch_out, &inner, order)
    }

    /// Shape of the broadcast matrix stack: `batch_out ++ [M, M]`, per `order`.
    pub fn a_shape(&self, order: FlagOrder) -> Vec<usize> {
        stack_shape(&self.batch_out, &[self.m, self.m], order)
    }
}

pub(crate) fn solve_plan(a: &[usize], b: &[usize], order: FlagOrder, op: &str) -> Result<SolvePlan> {
    let ndim_a = a.len();
    let ndim_b = b.len();
    rstsr_assert!(ndim_a >= 2, InvalidLayout, "{op}: a must be at least 2-D, got {ndim_a}")?;
    rstsr_assert!(ndim_b >= 1, InvalidLayout, "{op}: b must have at least 1 dimension")?;
    // array-API rule: b is a vector iff it is 1-D; otherwise it is a matrix stack
    // `(..., M, K)` (numpy / array-api-tests `solve_args`). Only the batch dims
    // broadcast.
    let is_vec = ndim_b == 1;
    let (batch_a, [m, n]) = batch_and_matrix_shape(a, order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "{op}: a must be square, got {m}x{n}")?;

    let (batch_b, k) = if is_vec {
        rstsr_assert_eq!(b[0], m, InvalidLayout, "{op}: vector b length must match a, got {} vs {m}", b[0])?;
        (Vec::new(), 1)
    } else {
        let (batch, [bm, bk]) = batch_and_matrix_shape(b, order)?;
        rstsr_assert_eq!(bm, m, InvalidLayout, "{op}: b's row dimension must match a, got {bm} vs {m}")?;
        (batch, bk)
    };
    let batch_out = broadcast_shapes_f(&[batch_a, batch_b], order)?;
    Ok(SolvePlan { batch_out, m, k, is_vec })
}

/// As [`map_stack_slices`], for a scalar-per-matrix output: `out` has the batch
/// shape and `f` writes one scalar into it per matrix slice.
pub(crate) fn map_stack_scalars<TA, TB, F>(
    a: TensorView<'_, TA, DeviceFaer, IxD>,
    mut out: TensorMut<'_, TB, DeviceFaer, IxD>,
    order: FlagOrder,
    mut f: F,
) -> Result<()>
where
    F: FnMut(TensorView<'_, TA, DeviceFaer, Ix2>, &mut TB) -> Result<()>,
{
    let (a_batch, a_inner) = split_inner(a.layout(), order, 2)?;
    let a_iters = IterLayout::new(&a_batch, TensorIterOrder::C)?;
    let o_iters = IterLayout::new(out.layout(), TensorIterOrder::C)?;
    for (off_a, off) in izip!(a_iters, o_iters) {
        let a_slice = {
            let mut inner = a_inner.clone().into_dim::<Ix2>()?;
            unsafe { inner.set_offset(off_a) };
            let (storage, _) = a.view().into_raw_parts();
            unsafe { TensorView::new_unchecked(storage, inner) }
        };
        f(a_slice, &mut out.raw_mut()[off])?;
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
