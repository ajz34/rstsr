use rstsr_core::prelude_dev::*;

/// Require a legal leading dimension from [`TensorBase::ld_row`] or
/// [`TensorBase::ld_col`].
///
/// `None` means the layout has no legal leading dimension for the requested
/// order (wrong order, or an empty/degenerate layout), which the BLAS/LAPACK
/// wrappers report as `InvalidLayout`. The offending layout is included in the
/// error message.
pub fn require_ld<D>(ld: Option<usize>, layout: &Layout<D>, operand: &str) -> Result<usize>
where
    D: DimDevAPI,
{
    ld.ok_or_else(|| rstsr_error!(InvalidLayout, "no legal leading dimension for operand {operand}: {layout:?}"))
}

/// Determine the single order a BLAS/LAPACK call should run in, from the
/// layout preferences of its tensor arguments.
///
/// Tags and votes take the form `(c_prefer, f_prefer)`:
///
/// - `by_first`: first-class tags, in priority order — an explicit output buffer (`out=`) goes
///   here, unwrapped as `Some`; its order wins outright. `None` entries are skipped.
/// - `by_all`: input votes — any exactly-one-prefer tag elects its order (c-prefer → `RowMajor`,
///   f-prefer → `ColMajor`). Both-prefer and neither-prefer tags are neutral (a size-0 or
///   single-row/column input reports both `true`).
/// - With no decisive entry, `default_order` is returned.
///
/// ```rust
/// use rstsr_blas_traits::util::util_layout::get_output_order;
/// use rstsr_core::prelude::rstsr_structs::*;
/// // explicit output in row-major; an f-prefer input is outvoted
/// assert_eq!(get_output_order(&[Some((true, false))], &[(false, true)], FlagOrder::ColMajor), FlagOrder::RowMajor);
/// // no output; one f-prefer input decides among otherwise-neutral inputs
/// assert_eq!(get_output_order(&[], &[(true, true), (false, true)], FlagOrder::RowMajor), FlagOrder::ColMajor);
/// // everything neutral: device default
/// assert_eq!(get_output_order(&[], &[(true, true), (false, false)], FlagOrder::RowMajor), FlagOrder::RowMajor);
/// ```
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

    // only exactly-one-prefer inputs vote; both-prefer (e.g. size-0 or
    // single-row/column) and neither-prefer are neutral
    let c_votes = by_all.iter().any(|&(c_prefer, f_prefer)| c_prefer && !f_prefer);
    let f_votes = by_all.iter().any(|&(c_prefer, f_prefer)| f_prefer && !c_prefer);
    if c_votes {
        RowMajor
    } else if f_votes {
        ColMajor
    } else {
        default_order
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn require_ld_reports_operand_and_layout() {
        let layout: Layout<Ix2> = [0usize, 18].f();
        let msg = format!("{:?}", require_ld(None, &layout, "B").unwrap_err());
        assert!(msg.contains("operand B"), "message = {msg}");
        assert!(msg.contains("shape: [0, 18]"), "message = {msg}");
    }
}
