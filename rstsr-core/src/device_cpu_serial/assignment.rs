use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

impl<T, DC, DA> OpAssignArbitaryAPI<T, DC, DA> for DeviceCpuSerial
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()> {
        if lc.c_contig() && la.c_contig() || lc.f_contig() && la.f_contig() {
            // contiguous case
            let offset_c = lc.offset();
            let offset_a = la.offset();
            let size = lc.size();
            for i in 0..size {
                c.rawvec[offset_c + i] = a.rawvec[offset_a + i].clone();
            }
        } else {
            // determine order by layout preference
            let order = {
                if lc.c_prefer() && la.c_prefer() {
                    TensorIterOrder::C
                } else if lc.f_prefer() && la.f_prefer() {
                    TensorIterOrder::F
                } else {
                    match TensorOrder::default() {
                        TensorOrder::C => TensorIterOrder::C,
                        TensorOrder::F => TensorIterOrder::F,
                    }
                }
            };
            // generate col-major iterator
            let lc = translate_to_col_major_unary(lc, order)?;
            let la = translate_to_col_major_unary(la, order)?;
            let iter_c = IterLayoutColMajor::new(&lc)?;
            let iter_a = IterLayoutColMajor::new(&la)?;
            // iterate and assign
            for (idx_c, idx_a) in izip!(iter_c, iter_a) {
                c.rawvec[idx_c] = a.rawvec[idx_a].clone();
            }
        }
        return Ok(());
    }
}

impl<T, D> OpAssignAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
    D: DimAPI,
{
    fn assign(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<D>,
        a: &Storage<T, Self>,
        la: &Layout<D>,
    ) -> Result<()> {
        let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
        let layouts_full_ref = layouts_full.iter().collect_vec();
        let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

        if size_contig >= CONTIG_SWITCH {
            let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
            let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
            for (idx_c, idx_a) in izip!(iter_c, iter_a) {
                for i in 0..size_contig {
                    c.rawvec_mut()[idx_c + i] = a.rawvec()[idx_a + i].clone();
                }
            }
        } else {
            let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
            let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
            for (idx_c, idx_a) in izip!(iter_c, iter_a) {
                c.rawvec_mut()[idx_c] = a.rawvec()[idx_a].clone();
            }
        }
        return Ok(());
    }

    fn fill(&self, c: &mut Storage<T, Self>, lc: &Layout<D>, fill: T) -> Result<()> {
        let layouts_full = [translate_to_col_major_unary(lc, TensorIterOrder::G)?];
        let layouts_full_ref = layouts_full.iter().collect_vec();
        let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

        if size_contig > CONTIG_SWITCH {
            let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
            for idx_c in iter_c {
                for i in 0..size_contig {
                    c.rawvec_mut()[idx_c + i] = fill.clone();
                }
            }
        } else {
            let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
            for idx_c in iter_c {
                c.rawvec_mut()[idx_c] = fill.clone();
            }
        }
        return Ok(());
    }
}
