use crate::storage::{DataAPI, StorageAPI, StorageToCpuAPI};
use crate::TensorBase;
use crate::{DimAPI, Layout};
use core::fmt::{Debug, Display, Formatter, Write};

pub static MIN_PRINT: usize = 3;
pub static MAX_PRINT: usize = 8;

pub struct FnPrintVecWithLayout<'v, 'l, 'f1, 'f2, T, D>
where
    T: Clone,
    D: DimAPI,
{
    vec: &'v [T],
    layout: &'l Layout<D>,
    offset: usize,
    idx_prev: Vec<usize>,
    max_print: usize,
    min_print: usize,
    fmt: &'f1 mut Formatter<'f2>,
}

pub fn print_vec_with_layout_dfs<T, D>(c: &mut FnPrintVecWithLayout<T, D>) -> core::fmt::Result
where
    T: Clone + Display,
    D: DimAPI,
{
    let FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt } = c;
    let max_print = max_print.clone();
    let min_print = min_print.clone();
    let ndim = layout.ndim();
    let offset = offset.clone();
    let len_prev = idx_prev.len();
    let shape = layout.shape_ref().as_ref();
    let stride = layout.stride_ref().as_ref();

    // special case
    if ndim == 0 {
        return write!(fmt, "[]");
    }

    if idx_prev.last().is_some_and(|&v| v == shape[len_prev - 1]) {
        // hit shape boundary

        // special case: zero shape quick return
        if idx_prev.last().is_some_and(|&v| v == 0) {
            return write!(fmt, "[]");
        }

        // pop last index, increase previous index by 1 or skip
        if len_prev == 1 {
            // multiple dimension exit
            return Ok(());
        } else {
            let p1_prev = idx_prev.pop().unwrap();
            let p2_prev = idx_prev.pop().unwrap();
            let offset = offset as isize
                - p1_prev as isize * stride[len_prev - 1]
                - p2_prev as isize * stride[len_prev - 2];
            if shape[len_prev - 2] < max_print
                || p2_prev + min_print >= shape[len_prev - 2]
                || p2_prev < min_print
            {
                let p2 = p2_prev + 1;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                return print_vec_with_layout_dfs(c);
            } else {
                write!(fmt, "{}...\n\n", " ".repeat(len_prev - 1))?;
                let p2 = shape[len_prev - 2] - min_print;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                return print_vec_with_layout_dfs(c);
            }
        }
    } else {
        // not hit shape boundary
        if len_prev + 1 != ndim {
            // new index can still be pushed
            idx_prev.push(0);
            return print_vec_with_layout_dfs(c);
        } else {
            // last line

            // number of last dimension
            let nlast = shape.last().unwrap().clone();
            let stride_last = stride.last().unwrap().clone();

            // special case: zero shape quick return
            if nlast == 0 {
                return write!(fmt, "[]");
            }

            // prefix: "  [[["
            // count zeros from last element as numbers of "["
            if idx_prev.is_empty() {
                write!(fmt, "[")?;
            } else {
                let mut nbra = 1;
                for &idx in idx_prev.iter().rev() {
                    if idx == 0 {
                        nbra += 1;
                    } else {
                        break;
                    }
                }
                write!(fmt, "{:}", " ".repeat(ndim - nbra))?;
                write!(fmt, "{:}", "[".repeat(nbra))?;
            }

            // values: " 1.23 4.56 ... 7.89 10.11"
            if nlast <= max_print.max(2 * min_print + 1) {
                // all elements in last dimension should be printed
                for i in 0..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Display::fmt(&vec[offset_i], fmt)?;
                }
            } else {
                // only print the first/last min_print elements
                for i in 0..min_print {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Display::fmt(&vec[offset_i], fmt)?;
                }
                write!(fmt, " ...")?;
                for i in (nlast - min_print)..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Display::fmt(&vec[offset_i], fmt)?;
                }
            };

            // suffix: "]]]"
            // count (if index = shape) from last element as numbers of "["
            if idx_prev.is_empty() {
                write!(fmt, "]")?;
            } else {
                let mut nket = 1;
                for i in (0..idx_prev.len()).rev() {
                    if idx_prev[i] == shape[i] - 1 {
                        nket += 1;
                    } else {
                        break;
                    }
                }
                write!(fmt, "{:}", "]".repeat(nket))?;
                if nket != ndim {
                    // last line should not add new-line character
                    if nket > 1 {
                        write!(fmt, "\n\n")?;
                    } else {
                        write!(fmt, "\n")?;
                    }
                }
            }

            // pop last index, increase previous index by 1 or skip
            if len_prev == 0 {
                // one-dimension exit
                return Ok(());
            } else {
                let p1_prev = idx_prev.pop().unwrap();
                let offset = offset as isize - p1_prev as isize * stride[len_prev - 1];
                if shape[len_prev - 1] < max_print
                    || p1_prev + min_print >= shape[len_prev - 1]
                    || p1_prev < min_print
                {
                    let p1 = p1_prev + 1;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    return print_vec_with_layout_dfs(c);
                } else {
                    write!(fmt, "{}...\n", " ".repeat(len_prev))?;
                    let p1 = shape[len_prev - 1] - min_print;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    return print_vec_with_layout_dfs(c);
                }
            }
        }
    }
}

pub fn print_vec_with_layout<'v, 'l, 'f1, 'f2, T, D>(
    fmt: &'f1 mut Formatter<'f2>,
    vec: &'v [T],
    layout: &'l Layout<D>,
    max_print: usize,
    min_print: usize,
) -> core::fmt::Result
where
    T: Clone + Display,
    D: crate::layout::DimAPI,
{
    let idx_prev = vec![];
    let offset = layout.offset;
    let mut config =
        FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt };
    print_vec_with_layout_dfs(&mut config)
}

impl<S, D> Display for TensorBase<S, D>
where
    S: DataAPI,
    D: DimAPI,
    S::Data: StorageAPI + StorageToCpuAPI,
    <S::Data as StorageAPI>::DType: Clone + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vec = self.data().as_storage().to_cpu_vec().unwrap();
        let layout = &self.layout();
        let max_print = MAX_PRINT;
        let min_print = MIN_PRINT;
        print_vec_with_layout(f, &vec, layout, max_print, min_print)
    }
}

impl<S, D> Debug for TensorBase<S, D>
where
    S: DataAPI,
    D: DimAPI,
    S::Data: StorageAPI + StorageToCpuAPI,
    <S::Data as StorageAPI>::DType: Clone + Display,
    <S::Data as StorageAPI>::Device: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Debug Tensor Print ===")?;
        Display::fmt(self, f)?;
        writeln!(f)?;
        Debug::fmt(&self.data().as_storage().device(), f)?;
        writeln!(f)?;
        Debug::fmt(&self.layout(), f)?;
        writeln!(f)?;
        writeln!(f, "===========================")
    }
}

#[cfg(test)]
mod playground {
    use super::*;

    struct VL<T, D>(Vec<T>, Layout<D>)
    where
        T: Clone + Display,
        D: DimAPI;

    impl<T, D> Display for VL<T, D>
    where
        T: Clone + Display,
        D: DimAPI,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let vec = self.0.as_slice();
            let layout = &self.1;
            print_vec_with_layout(f, vec, layout, MAX_PRINT, MIN_PRINT)
        }
    }

    #[test]
    fn playground() {
        use crate::layout::*;

        let mut s = String::new();

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [].into();
        println!("{:}", VL(vec, layout));

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [2, 0, 4].into();
        println!("{:}", VL(vec, layout));

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [2, 4, 0].into();
        println!("{:}", VL(vec, layout));
        println!("{:}", s);

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<Ix2> = [3, 5].into();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[4:13].reshape(3, 3)
        c = b.T[::2, ::-1]
        */
        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new(Shape([2, 3]), Stride([2, -3]), 10);
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[2:14].reshape(3, 4)
        c = b.T[:, ::-1]
        */
        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new(Shape([4, 3]), Stride([1, -4]), 10);
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[2:14].reshape(3, 4)
        c = b.T[:, ::-1]
        */
        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new(Shape([4, 3]), Stride([1, -4]), 10);
        println!("{:}", VL(vec, layout));

        s.clear();
        let vec = (0..1800).collect::<Vec<usize>>();
        let layout = Layout::<Ix3>::new(Shape([15, 12, 10]), Stride([1, 150, 15]), 0);
        println!("{:4}", VL(vec, layout));
    }
}