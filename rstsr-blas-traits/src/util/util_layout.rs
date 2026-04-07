use rstsr_core::prelude::rstsr_structs::*;

/// Get the preferred layout order for output tensor.
///
/// This function is mostly useful for multi-tensor involved BLAS3/LAPACK operations (gemm, symm,
/// trsm).
///
/// - `by_first`: given list of (c_prefer, f_prefer) for each input tensor, if the first encountered
///   input tensor has a clear preference (either row-major or column-major), return that layout.
/// - `by_all`: if all input tensors have the same preference (either all row-major or all
///   column-major), return that layout.
pub fn get_output_order(
    by_first: &[Option<(bool, bool)>],
    by_all: &[(bool, bool)],
    default_order: FlagOrder,
) -> FlagOrder {
    // inputs are in the form of (c_prefer, f_prefer)
    for x in by_first {
        match x {
            Some((true, false)) => return RowMajor,
            Some((false, true)) => return ColMajor,
            _ => continue,
        }
    }

    let row_all = by_all.iter().all(|&(c_prefer, _)| c_prefer);
    let col_all = by_all.iter().all(|&(_, f_prefer)| f_prefer);
    match (row_all, col_all) {
        (true, false) => RowMajor,
        (false, true) => ColMajor,
        _ => default_order,
    }
}
