//! Shared helpers for batched linalg entry points.

use rstsr_core::prelude_dev::*;

/// Split a stack-of-matrices shape into the batch shape and the 2-D matrix
/// shape, optionally requiring square matrices.
fn split_shape(shape: &[usize], order: FlagOrder, square: bool) -> Result<(Vec<usize>, [usize; 2])> {
    let ndim = shape.len();
    rstsr_assert!(ndim >= 2, InvalidLayout, "linalg: expected at least 2 dimensions, got {ndim}")?;
    let (batch_shape, mat) = match order {
        RowMajor => (shape[..ndim - 2].to_vec(), [shape[ndim - 2], shape[ndim - 1]]),
        ColMajor => (shape[2..].to_vec(), [shape[0], shape[1]]),
    };
    if square {
        let [m, n] = mat;
        rstsr_assert_eq!(m, n, InvalidLayout, "linalg: expected square matrices, got {m}x{n}")?;
    }
    Ok((batch_shape, mat))
}

/// Split the shape of a stack of matrices into the batch shape and the 2-D
/// matrix shape.
///
/// The two matrix axes are the last two under `RowMajor` and the first two
/// under `ColMajor` (the device order rule); all remaining dims are the batch.
/// Errors when `ndim < 2`.
pub fn batch_and_matrix_shape(shape: &[usize], order: FlagOrder) -> Result<(Vec<usize>, [usize; 2])> {
    split_shape(shape, order, false)
}

/// As [`batch_and_matrix_shape`], additionally requiring the matrix to be square;
/// returns the batch shape.
pub fn batch_and_square_shape(shape: &[usize], order: FlagOrder) -> Result<Vec<usize>> {
    split_shape(shape, order, true).map(|(batch_shape, _)| batch_shape)
}

/// Walk the batch dims of a matrix stack and apply `kernel` to every 2-D matrix
/// slice, in the device default-order flat sequence.
///
/// Returns the batch shape, the 2-D matrix shape, and the per-slice results in
/// that same order, ready to be assembled by [`batch_tensor_f`] /
/// [`assemble_batch_matrices_f`].
///
/// The batch dims are peeled from the leading side under `RowMajor` (matrix axes
/// last) and from the trailing side under `ColMajor` (matrix axes first), so the
/// 2-D slices keep their matrix orientation and the sequence follows the device
/// default-order flat sequence.
pub fn map_batch_matrices<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, [usize; 2], Vec<O>)>
where
    B: DeviceAPI<T>,
{
    walk_batch_matrices(a, order, false, kernel)
}

/// As [`map_batch_matrices`], additionally requiring the matrices to be square
/// (the single-operand square entries: `det`, `cholesky`, `inv`, `eigh`,
/// `eigvalsh`, `slogdet`). Non-square input yields `InvalidLayout` instead of a
/// device-side panic.
pub fn map_batch_square_matrices<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, [usize; 2], Vec<O>)>
where
    B: DeviceAPI<T>,
{
    walk_batch_matrices(a, order, true, kernel)
}

fn walk_batch_matrices<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    square: bool,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, [usize; 2], Vec<O>)>
where
    B: DeviceAPI<T>,
{
    let (batch_shape, matrix) = split_shape(a.shape(), order, square)?;
    let mut out = Vec::with_capacity(batch_shape.iter().product::<usize>());
    walk_matrices(a, order == ColMajor, kernel, &mut out)?;
    Ok((batch_shape, matrix, out))
}

fn walk_matrices<T, B, O>(
    v: TensorView<'_, T, B, IxD>,
    from_end: bool,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>) -> Result<O>,
    out: &mut Vec<O>,
) -> Result<()>
where
    B: DeviceAPI<T>,
{
    if v.ndim() == 2 {
        out.push(kernel(v.into_dim::<Ix2>())?);
        return Ok(());
    }
    let axis = if from_end { v.ndim() - 1 } else { 0 };
    for i in 0..v.shape()[axis] {
        let sub = if from_end { v.i((Ellipsis, i)) } else { v.i(i) };
        walk_matrices(sub, from_end, kernel, out)?;
    }
    Ok(())
}

/// Assemble default-order flat data into a tensor of `shape` on `device`.
pub fn batch_tensor_f<T, B>(data: Vec<T>, shape: Vec<usize>, device: &B) -> Result<Tensor<T, B, IxD>>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    asarray_f((data, shape, device))
}

/// As [`map_batch_matrices`], for two operands sharing one batch: `a` and `b`
/// are walked in lockstep, `kernel` receiving both 2-D slices.
pub fn map_batch_matrices2<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    b: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, [usize; 2], Vec<O>)>
where
    B: DeviceAPI<T>,
{
    let (batch_shape, matrix) = batch_and_matrix_shape(a.shape(), order)?;
    let mut out = Vec::with_capacity(batch_shape.iter().product::<usize>());
    walk_matrices2(a, b, order == ColMajor, kernel, &mut out)?;
    Ok((batch_shape, matrix, out))
}

fn walk_matrices2<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    b: TensorView<'_, T, B, IxD>,
    from_end: bool,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorView<'_, T, B, Ix2>) -> Result<O>,
    out: &mut Vec<O>,
) -> Result<()>
where
    B: DeviceAPI<T>,
{
    if a.ndim() == 2 {
        out.push(kernel(a.into_dim::<Ix2>(), b.into_dim::<Ix2>())?);
        return Ok(());
    }
    let axis = if from_end { a.ndim() - 1 } else { 0 };
    for i in 0..a.shape()[axis] {
        if from_end {
            walk_matrices2(a.i((Ellipsis, i)), b.i((Ellipsis, i)), from_end, kernel, out)?;
        } else {
            walk_matrices2(a.i(i), b.i(i), from_end, kernel, out)?;
        }
    }
    Ok(())
}

/// Validate that two matrix stacks share one batch shape and the same square
/// matrix dims (used by the two-operand entries such as generalized `eigh`).
pub fn check_same_batch<T, B>(
    a: &TensorView<'_, T, B, IxD>,
    b: &TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    op: &str,
) -> Result<()>
where
    B: DeviceAPI<T>,
{
    let (batch_a, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "{op}: matrix must be square, got {m}x{n}")?;
    let (batch_b, [bm, bn]) = batch_and_matrix_shape(b.shape(), order)?;
    rstsr_assert_eq!(bm, m, InvalidLayout, "{op}: operands must share the matrix row dimension")?;
    rstsr_assert_eq!(bn, n, InvalidLayout, "{op}: operands must share the matrix column dimension")?;
    rstsr_assert_eq!(batch_a, batch_b, InvalidLayout, "{op}: operands must share one batch shape")?;
    Ok(())
}

/// Broadcast-aware, in-place `solve`: each solution slice is written into `b`,
/// using the same broadcast rule as [`map_batch_solve_into_output`]. Because an
/// in-place solve cannot allocate, it errors whenever the solution does not fit
/// `b`'s shape exactly (e.g. a 1-D `b` under a stacked `a`) — pass a view of `b`
/// to the allocating form instead.
pub fn map_batch_solve_inplace<'a, 'b, T, B>(
    a: TensorView<'a, T, B, IxD>,
    mut b: TensorMut<'b, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorMut<'_, T, B, Ix2>) -> Result<()>,
) -> Result<()>
where
    B: DeviceAPI<T>,
{
    let (batch_out, m, k, is_vec) = solve_plan(&a, &b.view(), order)?;
    // b must already have the solution shape (the `(..., M)` form for a vector)
    let mut want_shape = batch_out.clone();
    want_shape.push(m);
    if !is_vec {
        want_shape.push(k);
    }
    rstsr_assert_eq!(
        want_shape,
        b.shape().to_vec(),
        InvalidLayout,
        "solve: an in-place solve needs the solution shape to equal b's shape; \
         pass a view of b to allocate a new result"
    )?;
    // a's matrix slices broadcast over the batch; b stays unbatched
    let mut a_shape = batch_out;
    a_shape.extend_from_slice(&[m, m]);
    let a_b = a.broadcast_to(a_shape);
    let b2d = if is_vec { b.i_mut((.., None)) } else { b };
    walk_matrices2_mut(a_b, b2d, order == ColMajor, kernel)
}

fn walk_matrices2_mut<'a, 'b, T, B>(
    a: TensorView<'a, T, B, IxD>,
    mut b: TensorMut<'b, T, B, IxD>,
    from_end: bool,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorMut<'_, T, B, Ix2>) -> Result<()>,
) -> Result<()>
where
    B: DeviceAPI<T>,
{
    if a.ndim() == 2 {
        kernel(a.into_dim::<Ix2>(), b.into_dim::<Ix2>())?;
        return Ok(());
    }
    let axis = if from_end { a.ndim() - 1 } else { 0 };
    for i in 0..a.shape()[axis] {
        if from_end {
            walk_matrices2_mut(a.i((Ellipsis, i)), b.i_mut((Ellipsis, i)), from_end, kernel)?;
        } else {
            walk_matrices2_mut(a.i(i), b.i_mut(i), from_end, kernel)?;
        }
    }
    Ok(())
}

/// Validate the operands of a batched `solve` and return the broadcast plan
/// `(batch_out, m, k, is_vec)`.
///
/// `a` is `(..., M, M)` and `b` is `(..., M)` (`is_vec`) or `(..., M, K)`; the
/// batch dims of `a` and `b` are broadcast against each other. A 1-D `b` is
/// treated as an `(..., M, 1)` stack.
pub fn solve_plan<T, B>(
    a: &TensorView<'_, T, B, IxD>,
    b: &TensorView<'_, T, B, IxD>,
    order: FlagOrder,
) -> Result<(Vec<usize>, usize, usize, bool)>
where
    B: DeviceAPI<T>,
{
    let is_vec = b.ndim() == 1;
    let (batch_a, [m, m2]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, m2, InvalidLayout, "solve: matrix a must be square, got {m}x{m2}")?;
    let (batch_b, k) = if is_vec {
        rstsr_assert_eq!(b.shape()[0], m, InvalidLayout, "solve: vector b length must match a")?;
        (Vec::new(), 1)
    } else {
        let (batch_b, [bm, bk]) = batch_and_matrix_shape(b.shape(), order)?;
        rstsr_assert_eq!(bm, m, InvalidLayout, "solve: b's row dimension must match a")?;
        (batch_b, bk)
    };
    let batch_out = broadcast_shapes_f(&[batch_a, batch_b], order)?.to_vec();
    // col-major keeps the matrix axes leading, so the batch broadcast would need
    // left-alignment; only the unbatched case is supported there.
    if order == ColMajor {
        rstsr_assert!(batch_out.is_empty(), InvalidLayout, "solve: n-dim batching is not supported under ColMajor")?;
    }
    Ok((batch_out, m, k, is_vec))
}

/// Solve into a freshly allocated output: the broadcast `b` is copied into the
/// output buffer once, then `kernel` solves each slice in place — no per-slice
/// temporaries and no separate assembly copy. The result is `(..., M, K)`, or
/// `(..., M)` for a 1-D `b`.
pub fn map_batch_solve_into_output<T, B>(
    a: TensorView<'_, T, B, IxD>,
    b: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorMut<'_, T, B, Ix2>) -> Result<()>,
) -> Result<Tensor<T, B, IxD>>
where
    T: Clone,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + OpAssignAPI<T, Vec<usize>>
        + OpAssignArbitaryAPI<T, IxD, IxD>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    let (batch_out, m, k, is_vec) = solve_plan(&a, &b, order)?;
    let mut a_shape = batch_out.clone();
    a_shape.extend_from_slice(&[m, m]);
    let mut out_shape = batch_out;
    out_shape.push(m);
    out_shape.push(k);
    let b2d = if is_vec { b.i((.., None)) } else { b };
    let mut out = b2d.broadcast_to(out_shape).to_owned();
    let a_b = a.broadcast_to(a_shape);
    walk_matrices2_mut(a_b, out.view_mut(), order == ColMajor, kernel)?;
    if is_vec {
        // drop the trailing length-1 axis of the (..., M, 1) buffer
        let mut shape = out.shape().to_vec();
        shape.pop();
        Ok(out.into_shape(shape))
    } else {
        Ok(out)
    }
}

/// Flatten an owned 1-D/2-D per-slice result into a `Vec` in device default order.
pub fn into_default_vec<T, B, D>(t: Tensor<T, B, D>) -> Vec<T>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, Ix1>
        + OpAssignAPI<T, Vec<usize>>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    t.into_shape(-1).into_dim::<Ix1>().into_vec()
}

/// Assemble per-slice owned 1-D/2-D results into one batched tensor, placing the
/// matrix/vector axes per the device order.
pub fn assemble_batch_matrices_f<T, B, D>(
    mats: Vec<Tensor<T, B, D>>,
    batch_shape: &[usize],
    matrix: &[usize],
    order: FlagOrder,
    device: &B,
) -> Result<Tensor<T, B, IxD>>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, D>
        + OpAssignAPI<T, Ix1>
        + OpAssignAPI<T, Vec<usize>>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    let shape = batch_matrix_output_shape(batch_shape, matrix, order);
    let mut data = Vec::with_capacity(mats.len() * matrix.iter().product::<usize>());
    for t in mats {
        data.extend(into_default_vec(t));
    }
    batch_tensor_f(data, shape, device)
}

/// Output shape of a batched matrix result: matrix axes last under `RowMajor`,
/// first under `ColMajor` (the device order rule).
pub fn batch_matrix_output_shape(batch_shape: &[usize], matrix: &[usize], order: FlagOrder) -> Vec<usize> {
    let mut shape = Vec::with_capacity(batch_shape.len() + matrix.len());
    match order {
        RowMajor => {
            shape.extend_from_slice(batch_shape);
            shape.extend_from_slice(matrix);
        },
        ColMajor => {
            shape.extend_from_slice(matrix);
            shape.extend_from_slice(batch_shape);
        },
    }
    shape
}
