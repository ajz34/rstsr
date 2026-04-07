use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::*;

/* #region TensorMutable */

pub type TensorMutable1<'a, T, B> = TensorMutable<'a, T, B, Ix1>;
pub type TensorMutable2<'a, T, B> = TensorMutable<'a, T, B, Ix2>;

/// Convert a view/mut tensor reference to a mutable tensor.
///
/// # Note on preferred/contiguous layout
///
/// Given matrix A of shape `[m, n]`,
/// - C-contiguous stride is `[n, 1]`, and F-contiguous stride is `[1, m]`.
/// - C-preferred stride is `[lda, 1]` (where `lda >= n`), and F-preferred stride is `[1, lda]`
///   (where `lda >= m`). `lda` is the leading dimension.
///
/// The mutable tensor will be in either row-major or col-major.
/// - If input is view (non-mutable), it will always be converted to contiguous owned tensor with
///   the default layout;
/// - If input is mutable, and already in the preferred layout (either C/F-prefer), it will be used
///   as is;
/// - If input is mutable, but not in the preferred layout, a temporary contiguous owned tensor with
///   the preferred layout will be created. The computation will use this temporary tensor, and the
///   result will be copied back to the original tensor (manually by caller).
pub fn overwritable_convert<T, B, D>(a: TensorReference<'_, T, B, D>) -> Result<TensorMutable<'_, T, B, D>>
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
    D: DimAPI,
{
    let order = match (a.f_prefer(), a.c_prefer()) {
        (true, false) => ColMajor,
        (false, true) => RowMajor,
        _ => a.device().default_order(),
    };
    let a = if a.is_ref() {
        TensorMutable::Owned(TensorView::from(a).into_contig_f(order)?)
    } else {
        let a = TensorMut::from(a);
        if a.f_prefer() || a.c_prefer() {
            TensorMutable::Mut(a)
        } else {
            let a_buffer = a.to_contig_f(order)?.into_owned();
            TensorMutable::ToBeCloned(a, a_buffer)
        }
    };
    Ok(a)
}

/// Same as `overwritable_convert`, but allows caller to specify the preferred layout (row-major or
/// column-major).
pub fn overwritable_convert_with_order<T, B, D>(
    a: TensorReference<'_, T, B, D>,
    order: FlagOrder,
) -> Result<TensorMutable<'_, T, B, D>>
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
    D: DimAPI,
{
    let a = if a.is_ref() {
        TensorMutable::Owned(TensorView::from(a).into_contig_f(order)?)
    } else {
        let a = TensorMut::from(a);
        if (order == ColMajor && a.f_prefer()) || (order == RowMajor && a.c_prefer()) {
            TensorMutable::Mut(a)
        } else {
            let a_buffer = a.to_contig_f(order)?.into_owned();
            TensorMutable::ToBeCloned(a, a_buffer)
        }
    };
    Ok(a)
}

/* #endregion */

/* #region flip */

/// Helper function to flip the transpose flag and tensor layout.
///
/// This function is intended to be used in BLAS/LAPACK operations that have `trans` option, to
/// avoid unnecessary memory allocation and transposition.
///
/// - If the input tensor is already in the preferred layout, it will be used as is.
/// - If the input tensor is not in the preferred layout, it will perform flip to both the tensor
///   and the transpose flag, try to see if the flipped layout is in the preferred layout.
///   - If the flipped layout is in the preferred layout, it will return view of the original tensor
///     with flipped layout (without allocating new tensor).
///   - If the flipped layout is still not in the preferred layout, it will allocate a new tensor
///     with the preferred layout, and return it.
pub fn flip_trans<T, B>(
    order: FlagOrder,
    trans: FlagTrans,
    view: TensorView<'_, T, B, Ix2>,
    hermi: bool,
) -> Result<(FlagTrans, TensorCow<'_, T, B, Ix2>)>
where
    T: Clone,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpConjAPI<T, Ix2, TOut = T>,
{
    // row-major
    if (order == FlagOrder::C && view.c_prefer()) || (order == FlagOrder::F && view.f_prefer()) {
        // tensor is already in the preferred order
        Ok((trans, view.into_cow()))
    } else {
        // otherwise, flip both the tensor and flag, and allocate new tensor if
        // necessary
        match trans {
            FlagTrans::N => Ok((trans.flip(hermi)?, match hermi {
                true => view.into_reverse_axes().change_prefer(order).conj().into_cow(),
                false => view.into_reverse_axes().change_prefer(order),
            })),
            FlagTrans::T => Ok((trans.flip(hermi)?, view.into_reverse_axes().change_prefer(order))),
            FlagTrans::C => Ok((trans.flip(hermi)?, view.into_reverse_axes().change_prefer(order).conj().into_cow())),
            _ => rstsr_invalid!(trans),
        }
    }
}

/* #endregion */
