use crate::prelude_dev::*;

impl<T, D> DeviceOpBinary<T, D> for CpuDevice
where
    T: Clone,
    D: DimAPI,
{
    fn assign(
        &self,
        a: &mut Storage<T, CpuDevice>,
        la: &Layout<D>,
        b: &Storage<T, CpuDevice>,
        lb: &Layout<D>,
    ) -> Result<()> {
        rstsr_assert_eq!(la.size(), lb.size(), InvalidLayout)?;
        if la.is_c_contig() && lb.is_c_contig() || la.is_f_contig() && lb.is_f_contig() {
            let it_a = IterLayoutMemNonStrided::new_it(la).unwrap();
            let it_b = IterLayoutMemNonStrided::new_it(lb).unwrap();
            for (ia, ib) in it_a.zip(it_b) {
                a.rawvec[ia] = b.rawvec[ib].clone();
            }
        } else {
            match Order::default() {
                Order::C => {
                    let it_a = IterLayoutC::new_it(la).unwrap();
                    let it_b = IterLayoutC::new_it(lb).unwrap();
                    for (ia, ib) in it_a.zip(it_b) {
                        a.rawvec[ia] = b.rawvec[ib].clone();
                    }
                },
                Order::F => {
                    let it_a = IterLayoutF::new_it(la).unwrap();
                    let it_b = IterLayoutF::new_it(lb).unwrap();
                    for (ia, ib) in it_a.zip(it_b) {
                        a.rawvec[ia] = b.rawvec[ib].clone();
                    }
                },
            }
        }
        Ok(())
    }
}
