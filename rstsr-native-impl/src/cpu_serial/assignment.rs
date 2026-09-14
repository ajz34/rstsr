use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

#[duplicate_item(
    func_name
        TypeC TypeA Types
        func_clone
        oc_r2c oc_c2r
    ;
    [assign_arbitary_cpu_serial]
        [T] [T] [T: Clone]
        [*ci = ai.clone()]
        [orderchange_out_r2c_ix2_promote_cpu_serial] [orderchange_out_c2r_ix2_promote_cpu_serial]
    ;
    [assign_arbitary_uninit_cpu_serial]
        [MaybeUninit<T>] [T] [T: Clone]
        [ci.write(ai.clone())]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_serial] [orderchange_out_c2r_ix2_uninit_promote_cpu_serial]
    ;
    [assign_arbitary_promote_cpu_serial]
        [TC] [TA] [TC: Clone, TA: Clone + DTypeCastAPI<TC>]
        [*ci = ai.clone().into_cast()]
        [orderchange_out_r2c_ix2_promote_cpu_serial] [orderchange_out_c2r_ix2_promote_cpu_serial]
    ;
    [assign_arbitary_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA] [TC: Clone, TA: Clone + DTypeCastAPI<TC>]
        [ci.write(ai.clone().into_cast())]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_serial] [orderchange_out_c2r_ix2_uninit_promote_cpu_serial]
    ;
)]
pub fn func_name<Types, DC, DA>(
    c: &mut [TypeC],
    lc: &Layout<DC>,
    a: &[TypeA],
    la: &Layout<DA>,
    order: FlagOrder,
) -> Result<()>
where
    DC: DimAPI,
    DA: DimAPI,
{
    let contig = match order {
        RowMajor => lc.c_contig() && la.c_contig(),
        ColMajor => lc.f_contig() && la.f_contig(),
    };
    if contig {
        // contiguous case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)].iter_mut().zip(a[offset_a..(offset_a + size)].iter()).for_each(|(ci, ai)| {
            func_clone;
        });
    } else {
        // 2-D order-change fast path: when both layouts are 2-D and match the
        // transpose pattern (one fast on axis 0, the other on axis 1, fast
        // strides exactly +1), use the blocked orderchange kernels
        // (BLOCK_SIZE = 64). Guards reject everything else — zero/negative
        // fast-axis strides (broadcast / flip), sliced fast axes, ndim != 2 —
        // which fall through to the generic iterator path unchanged. A copy
        // writes every output element exactly once from one input element,
        // so the tiled visit order is bit-identical to the iterator order.
        // shape identity is required: shape-changing assigns (e.g. a reshape
        // that re-lays out data) pass through the same kernels with different
        // shapes, which the orderchange kernels cannot serve
        if let (Ok(lc2), Ok(la2)) = (lc.to_dim::<Ix2>(), la.to_dim::<Ix2>()) {
            if lc2.shape() == la2.shape() {
                let sc = *lc2.stride();
                let sa = *la2.stride();
                if sa[1] == 1 && sc[0] == 1 {
                    return oc_r2c(c, &lc2, a, &la2);
                } else if sa[0] == 1 && sc[1] == 1 {
                    return oc_c2r(c, &lc2, a, &la2);
                }
            }
        }
        // determine order by layout preference
        let order = match order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        layout_col_major_dim_dispatch_2diff(&lc, &la, |(idx_c, idx_a)| {
            let ci = &mut c[idx_c];
            let ai = &a[idx_a];
            func_clone;
        })?;
    }
    Ok(())
}

#[duplicate_item(
    func_name
        TypeC TypeA Types
        func_clone
        oc_r2c oc_c2r
    ;
    [assign_cpu_serial]
        [T] [T] [T: Clone]
        [*ci = ai.clone()]
        [orderchange_out_r2c_ix2_promote_cpu_serial] [orderchange_out_c2r_ix2_promote_cpu_serial]
    ;
    [assign_uninit_cpu_serial]
        [MaybeUninit<T>] [T] [T: Clone]
        [ci.write(ai.clone())]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_serial] [orderchange_out_c2r_ix2_uninit_promote_cpu_serial]
    ;
    [assign_promote_cpu_serial]
        [TC] [TA] [TC: Clone, TA: Clone + DTypeCastAPI<TC>]
        [*ci = ai.clone().into_cast()]
        [orderchange_out_r2c_ix2_promote_cpu_serial] [orderchange_out_c2r_ix2_promote_cpu_serial]
    ;
    [assign_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA] [TC: Clone, TA: Clone + DTypeCastAPI<TC>]
        [ci.write(ai.clone().into_cast())]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_serial] [orderchange_out_c2r_ix2_uninit_promote_cpu_serial]
    ;
)]
pub fn func_name<Types, D>(c: &mut [TypeC], lc: &Layout<D>, a: &[TypeA], la: &Layout<D>) -> Result<()>
where
    D: DimAPI,
{
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    if size_contig >= CONTIG_SWITCH {
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            c[idx_c..(idx_c + size_contig)].iter_mut().zip(a[idx_a..(idx_a + size_contig)].iter()).for_each(
                |(ci, ai)| {
                    func_clone;
                },
            );
        })?;
    } else {
        // 2-D order-change fast path (see assign_arbitary_cpu_serial): both
        // layouts 2-D, transpose stride pattern, fast axes exactly +1; a
        // fully/partially contiguous pair never reaches here (the contig
        // branch above takes it), so this never shadows the memcpy path.
        // shape identity is required: shape-changing assigns (e.g. a reshape
        // that re-lays out data) pass through the same kernels with different
        // shapes, which the orderchange kernels cannot serve
        if let (Ok(lc2), Ok(la2)) = (lc.to_dim::<Ix2>(), la.to_dim::<Ix2>()) {
            if lc2.shape() == la2.shape() {
                let sc = *lc2.stride();
                let sa = *la2.stride();
                if sa[1] == 1 && sc[0] == 1 {
                    return oc_r2c(c, &lc2, a, &la2);
                } else if sa[0] == 1 && sc[1] == 1 {
                    return oc_c2r(c, &lc2, a, &la2);
                }
            }
        }
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            let ci = &mut c[idx_c];
            let ai = &a[idx_a];
            func_clone;
        })?;
    }
    Ok(())
}

pub fn fill_cpu_serial<T, D>(c: &mut [T], lc: &Layout<D>, fill: T) -> Result<()>
where
    T: Clone,
    D: DimAPI,
{
    fill_promote_cpu_serial(c, lc, fill)
}

pub fn fill_promote_cpu_serial<TC, TA, D>(c: &mut [TC], lc: &Layout<D>, fill: TA) -> Result<()>
where
    TA: Clone + DTypeCastAPI<TC>,
    TC: Clone,
    D: DimAPI,
{
    let fill = fill.clone().into_cast();

    let layouts_full = [translate_to_col_major_unary(lc, TensorIterOrder::G)?];
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    if size_contig > CONTIG_SWITCH {
        layout_col_major_dim_dispatch_1(&layouts_contig[0], |idx_c| {
            for i in 0..size_contig {
                c[idx_c + i] = fill.clone();
            }
        })?;
    } else {
        layout_col_major_dim_dispatch_1(&layouts_full[0], |idx_c| {
            c[idx_c] = fill.clone();
        })?;
    }
    Ok(())
}
