//! Shared helpers for batched linalg entry points.

use rstsr_core::prelude_dev::*;

/// Split the shape of a stack of matrices into the batch shape and the 2-D
/// matrix shape.
///
/// The two matrix axes are the last two under `RowMajor` and the first two
/// under `ColMajor` (the device order rule); all remaining dims are the batch.
/// Errors when `ndim < 2`.
pub(crate) fn batch_and_matrix_shape(shape: &[usize], order: FlagOrder) -> Result<(Vec<usize>, [usize; 2])> {
    let ndim = shape.len();
    rstsr_assert!(ndim >= 2, InvalidLayout, "linalg: expected at least 2 dimensions, got {ndim}")?;
    let (batch_shape, mat) = match order {
        RowMajor => (shape[..ndim - 2].to_vec(), [shape[ndim - 2], shape[ndim - 1]]),
        ColMajor => (shape[2..].to_vec(), [shape[0], shape[1]]),
    };
    Ok((batch_shape, mat))
}

/// As [`batch_and_matrix_shape`], additionally requiring the matrix to be square;
/// returns the batch shape.
pub(crate) fn batch_and_square_shape(shape: &[usize], order: FlagOrder) -> Result<Vec<usize>> {
    let (batch_shape, [m, n]) = batch_and_matrix_shape(shape, order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "linalg: expected square matrices, got {m}x{n}")?;
    Ok(batch_shape)
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
pub(crate) fn map_batch_matrices<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, [usize; 2], Vec<O>)>
where
    B: DeviceAPI<T>,
{
    let (batch_shape, matrix) = batch_and_matrix_shape(a.shape(), order)?;
    let mut out = Vec::new();
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
pub(crate) fn batch_tensor_f<T, B>(data: Vec<T>, shape: Vec<usize>, device: &B) -> Result<Tensor<T, B, IxD>>
where
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    asarray_f((data, shape, device))
}

/// As [`map_batch_matrices`], for two operands sharing one batch: `a` and `b`
/// are walked in lockstep, `kernel` receiving both 2-D slices.
pub(crate) fn map_batch_matrices2<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    b: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, [usize; 2], Vec<O>)>
where
    B: DeviceAPI<T>,
{
    let (batch_shape, matrix) = batch_and_matrix_shape(a.shape(), order)?;
    let mut out = Vec::new();
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
pub(crate) fn check_same_batch<T, B>(
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

/// As [`map_batch_matrices2`], with a mutable `b`: the kernel receives each of
/// `b`'s slices as a `TensorMut` so it can write the solution in place.
pub(crate) fn map_batch_matrices2_mut<'a, 'b, T, B, O>(
    a: TensorView<'a, T, B, IxD>,
    mut b: TensorMut<'b, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorMut<'_, T, B, Ix2>) -> Result<O>,
    out: &mut Vec<O>,
) -> Result<()>
where
    B: DeviceAPI<T>,
{
    if a.ndim() == 2 {
        out.push(kernel(a.into_dim::<Ix2>(), b.into_dim::<Ix2>())?);
        return Ok(());
    }
    let from_end = order == ColMajor;
    let axis = if from_end { a.ndim() - 1 } else { 0 };
    for i in 0..a.shape()[axis] {
        if from_end {
            map_batch_matrices2_mut(a.i((Ellipsis, i)), b.i_mut((Ellipsis, i)), order, kernel, out)?;
        } else {
            map_batch_matrices2_mut(a.i(i), b.i_mut(i), order, kernel, out)?;
        }
    }
    Ok(())
}

/// Broadcast a `(..., M, M)` stack `a` and a `(..., M)` (vector) or
/// `(..., M, K)` (matrix) stack `b` into a common batch and apply `kernel` to
/// every 2-D slice pair.
///
/// Returns the output batch shape, `m`, `k` (1 for a vector), the vector flag,
/// and the per-slice results. A 1-D `b` is treated as an `(..., M, 1)` stack.
#[allow(clippy::type_complexity)]
pub(crate) fn map_batch_solve<T, B, O>(
    a: TensorView<'_, T, B, IxD>,
    b: TensorView<'_, T, B, IxD>,
    order: FlagOrder,
    kernel: &mut impl FnMut(TensorView<'_, T, B, Ix2>, TensorView<'_, T, B, Ix2>) -> Result<O>,
) -> Result<(Vec<usize>, usize, usize, bool, Vec<O>)>
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
    let mut a_shape = batch_out.clone();
    a_shape.extend_from_slice(&[m, m]);
    let mut b_shape = batch_out.clone();
    b_shape.extend_from_slice(&[m, k]);
    let a_b = a.broadcast_to(a_shape);
    let b2d = if is_vec { b.i((.., None)) } else { b };
    let b_b = b2d.broadcast_to(b_shape);
    let mut out = Vec::new();
    walk_matrices2(a_b, b_b, order == ColMajor, kernel, &mut out)?;
    Ok((batch_out, m, k, is_vec, out))
}

/// Flatten an owned 1-D/2-D per-slice result into a `Vec` in device default order.
pub(crate) fn into_default_vec<T, B, D>(t: Tensor<T, B, D>) -> Vec<T>
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
pub(crate) fn assemble_batch_matrices_f<T, B, D>(
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
    let mut data = Vec::new();
    for t in mats {
        data.extend(into_default_vec(t));
    }
    batch_tensor_f(data, shape, device)
}

/// Output shape of a batched matrix result: matrix axes last under `RowMajor`,
/// first under `ColMajor` (the device order rule).
pub(crate) fn batch_matrix_output_shape(batch_shape: &[usize], matrix: &[usize], order: FlagOrder) -> Vec<usize> {
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
