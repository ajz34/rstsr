//! Host-side tensor formatting kernel.
//!
//! Prints a host slice through a tensor layout, truncating large axes. The
//! device layer chooses the element formatter (`Display`, `Debug`, ...).

use crate::prelude_dev::*;

struct FnPrintVecWithLayout<'v, 'l, 'f1, 'f2, T, D>
where
    D: DimAPI,
{
    vec: &'v [T],
    layout: &'l Layout<D>,
    offset: usize,
    idx_prev: Vec<usize>,
    max_print: usize,
    min_print: usize,
    fmt: &'f1 mut core::fmt::Formatter<'f2>,
    fmt_elem: fn(&T, &mut core::fmt::Formatter<'_>) -> core::fmt::Result,
}

fn print_vec_with_layout_dfs_inner<T, D>(c: &mut FnPrintVecWithLayout<T, D>) -> core::fmt::Result
where
    D: DimAPI,
{
    let FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt, fmt_elem } = c;
    let max_print = *max_print;
    let min_print = *min_print;
    let ndim = layout.ndim();
    let offset = *offset;
    let len_prev = idx_prev.len();
    let shape = layout.shape().as_ref();
    let stride = layout.stride().as_ref();
    let fmt_elem = *fmt_elem;

    // special case
    if ndim == 0 {
        return fmt_elem(&vec[layout.offset()], fmt);
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
            Ok(())
        } else {
            let p1_prev = idx_prev.pop().unwrap();
            let p2_prev = idx_prev.pop().unwrap();
            let offset =
                offset as isize - p1_prev as isize * stride[len_prev - 1] - p2_prev as isize * stride[len_prev - 2];
            if shape[len_prev - 2] < max_print || p2_prev + min_print >= shape[len_prev - 2] || p2_prev + 1 < min_print
            {
                let p2 = p2_prev + 1;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                print_vec_with_layout_dfs_inner(c)
            } else {
                write!(fmt, "{}...\n\n", " ".repeat(len_prev - 1))?;
                let p2: usize = shape[len_prev - 2] - min_print;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                print_vec_with_layout_dfs_inner(c)
            }
        }
    } else {
        // not hit shape boundary
        if len_prev + 1 != ndim {
            // new index can still be pushed
            idx_prev.push(0);
            print_vec_with_layout_dfs_inner(c)
        } else {
            // last line

            // number of last dimension
            let nlast = *shape.last().unwrap();
            let stride_last = *stride.last().unwrap();

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
                    fmt_elem(&vec[offset_i], fmt)?;
                }
            } else {
                // only print the first/last min_print elements
                for i in 0..min_print {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    fmt_elem(&vec[offset_i], fmt)?;
                }
                write!(fmt, " ...")?;
                for i in (nlast - min_print)..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    fmt_elem(&vec[offset_i], fmt)?;
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
                        writeln!(fmt, "\n")?;
                    } else {
                        writeln!(fmt)?;
                    }
                }
            }

            // pop last index, increase previous index by 1 or skip
            if len_prev == 0 {
                // one-dimension exit
                Ok(())
            } else {
                let p1_prev = idx_prev.pop().unwrap();
                let offset = offset as isize - p1_prev as isize * stride[len_prev - 1];
                if shape[len_prev - 1] < max_print
                    || p1_prev + min_print >= shape[len_prev - 1]
                    || p1_prev + 1 < min_print
                {
                    let p1 = p1_prev + 1;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    print_vec_with_layout_dfs_inner(c)
                } else {
                    writeln!(fmt, "{}...", " ".repeat(len_prev))?;
                    let p1: usize = shape[len_prev - 1] - min_print;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    print_vec_with_layout_dfs_inner(c)
                }
            }
        }
    }
}

/// Print a host slice through `layout`, formatting each element with `fmt_elem`.
pub fn format_tensor_cpu_serial<T, D>(
    fmt: &mut core::fmt::Formatter<'_>,
    vec: &[T],
    layout: &Layout<D>,
    max_print: usize,
    min_print: usize,
    fmt_elem: fn(&T, &mut core::fmt::Formatter<'_>) -> core::fmt::Result,
) -> core::fmt::Result
where
    D: DimAPI,
{
    let idx_prev = vec![];
    let offset = layout.offset();
    let mut config = FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt, fmt_elem };
    print_vec_with_layout_dfs_inner(&mut config)
}
