use crate::{DimAPI, Layout};
use core::fmt::{Display, Write};

pub static MIN_PRINT: usize = 3;
pub static MAX_PRINT: usize = 10;

pub struct FnPrintVecWithLayout<'v, 'l, 'f, T, D>
where
    T: Clone,
    D: DimAPI,
{
    vec: &'v Vec<T>,
    layout: &'l Layout<D>,
    offset: usize,
    idx_prev: Vec<usize>,
    max_print: usize,
    min_print: usize,
    fmt: &'f mut (dyn Write + 'f),
}

pub fn print_vec_with_layout_dfs<T, D>(c: &mut FnPrintVecWithLayout<T, D>)
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
        write!(fmt, "[]").unwrap();
        return;
    }

    if idx_prev.last().is_some_and(|&v| v == shape[len_prev - 1]) {
        // hit shape boundary

        // special case: zero shape quick return
        if idx_prev.last().is_some_and(|&v| v == 0) {
            write!(fmt, "[]").unwrap();
            return;
        }

        // pop last index, increase previous index by 1 or skip
        if len_prev == 1 {
            // multiple dimension exit
            return;
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
                write!(fmt, "{}...\n\n", " ".repeat(len_prev - 1)).unwrap();
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
                write!(fmt, "[]").unwrap();
                return;
            }

            // prefix: "  [[["
            // count zeros from last element as numbers of "["
            let mut prefix = String::new();
            if idx_prev.is_empty() {
                write!(prefix, "[").unwrap();
            } else {
                let mut nbra = 1;
                for &idx in idx_prev.iter().rev() {
                    if idx == 0 {
                        nbra += 1;
                    } else {
                        break;
                    }
                }
                write!(prefix, "{:}", " ".repeat(ndim - nbra)).unwrap();
                write!(prefix, "{:}", "[".repeat(nbra)).unwrap();
            }

            // values: " 1.23 4.56 ... 7.89 10.11"
            let mut values = String::new();
            if nlast <= max_print.max(2 * min_print + 1) {
                // all elements in last dimension should be printed
                for i in 0..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(values, " {:}", vec[offset_i]).unwrap();
                }
            } else {
                // only print the first/last min_print elements
                for i in 0..min_print {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(values, " {:}", vec[offset_i]).unwrap();
                }
                write!(values, " ...").unwrap();
                for i in (nlast - min_print)..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(values, " {:}", vec[offset_i]).unwrap();
                }
            };

            // suffix: "]]]"
            // count (if index = shape) from last element as numbers of "["
            let mut suffix = String::new();
            if idx_prev.is_empty() {
                write!(suffix, "]").unwrap();
            } else {
                let mut nket = 1;
                for i in (0..idx_prev.len()).rev() {
                    if idx_prev[i] == shape[i] - 1 {
                        nket += 1;
                    } else {
                        break;
                    }
                }
                write!(suffix, "{:}", "]".repeat(nket)).unwrap();
                if nket != ndim {
                    // last line should not add new-line character
                    if nket > 1 {
                        write!(suffix, "\n\n").unwrap();
                    } else {
                        write!(suffix, "\n").unwrap();
                    }
                }
            }

            // write to display
            write!(fmt, "{:}{:}{:}", prefix, values, suffix).unwrap();

            // pop last index, increase previous index by 1 or skip
            if len_prev == 0 {
                // one-dimension exit
                return;
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
                    write!(fmt, "{}...\n", " ".repeat(len_prev)).unwrap();
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

pub fn print_vec_with_layout<'v, 'l, 'f, T, D>(
    fmt: &'f mut (dyn Write + 'f),
    vec: &'v Vec<T>,
    layout: &'l Layout<D>,
    max_print: usize,
    min_print: usize,
) where
    T: Clone + Display,
    D: crate::layout::DimAPI,
{
    let idx_prev = vec![];
    let offset = layout.offset;
    let mut config = FnPrintVecWithLayout {
        vec: &vec,
        layout: &layout,
        offset,
        idx_prev,
        max_print,
        min_print,
        fmt,
    };
    print_vec_with_layout_dfs(&mut config);
}

#[test]
fn playground() {
    use crate::layout::*;

    let mut s = String::new();

    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    print_vec_with_layout(&mut s, &vec, &[].into(), 10, 3);
    println!("{:}", s);

    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    print_vec_with_layout(&mut s, &vec, &[].into(), 10, 3);
    println!("{:}", s);

    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    print_vec_with_layout(&mut s, &vec, &[].into(), 10, 3);
    println!("{:}", s);

    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    print_vec_with_layout(&mut s, &vec, &[].into(), 10, 3);
    println!("{:}", s);

    /* Python code
       a = np.arange(15)
       b = a[4:13].reshape(3, 3)
       c = b.T[::2, ::-1]
    */
    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let layout = Layout::<Ix2>::new(Shape([2, 3]), Stride([2, -3]), 10);
    print_vec_with_layout(&mut s, &vec, &layout, 10, 3);
    println!("{:}", s);

    /* Python code
       a = np.arange(15)
       b = a[2:14].reshape(3, 4)
       c = b.T[:, ::-1]
    */
    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let layout = Layout::<Ix2>::new(Shape([4, 3]), Stride([1, -4]), 10);
    print_vec_with_layout(&mut s, &vec, &layout, 10, 3);
    println!("{:}", s);

    /* Python code
       a = np.arange(15)
       b = a[2:14].reshape(3, 4)
       c = b.T[:, ::-1]
    */
    s.clear();
    let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let layout = Layout::<Ix2>::new(Shape([4, 3]), Stride([1, -4]), 10);
    print_vec_with_layout(&mut s, &vec, &layout, 10, 3);
    println!("{:}", s);

    s.clear();
    let vec = (0..1800).collect::<Vec<usize>>();
    let layout = Layout::<Ix3>::new(Shape([15, 12, 10]), Stride([1, 150, 15]), 0);
    print_vec_with_layout(&mut s, &vec, &layout, 8, 3);
    println!("{:}", s);
}
