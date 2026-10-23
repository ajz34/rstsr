//! Element addressing for printing CUDA tensors.
//!
//! The traversal must mirror `rstsr-native-impl`'s
//! `print_vec_with_layout_dfs_inner` exactly (row-major DFS with
//! `min_print`/`max_print` truncation on every axis). The printed
//! elements are gathered in one batched D2H and staged compactly; the
//! host holds only those values, never a full-tensor copy.

use rstsr_core::prelude_dev::*;

/// Indices visited on a non-last axis: mirror of the walker's advance
/// rule (`shape < max_print`, else head `min_print` + tail `min_print`).
fn prefix_axis_visited(n: usize, max_print: usize, min_print: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut p = 0usize;
    if n == 0 {
        return out;
    }
    loop {
        out.push(p);
        let next = if n < max_print || p + min_print >= n || p + 1 < min_print {
            if p + 1 < n { Some(p + 1) } else { None }
        } else {
            Some(n - min_print)
        };
        match next {
            Some(v) => p = v,
            None => break,
        }
    }
    out
}

/// Indices printed on the last axis (mirror of the walker's line printer).
fn last_axis_visited(n: usize, max_print: usize, min_print: usize) -> Vec<usize> {
    if n <= max_print.max(2 * min_print + 1) {
        (0..n).collect()
    } else {
        (0..min_print).chain((n - min_print)..n).collect()
    }
}

/// Absolute buffer indices (offset included) the CPU walker prints for
/// `layout` under the given limits, in visit order.
pub(crate) fn touched_element_indices<D>(layout: &Layout<D>, max_print: usize, min_print: usize) -> Vec<usize>
where
    D: DimAPI,
{
    let offset = layout.offset() as isize;
    let shape = layout.shape().as_ref();
    let stride = layout.stride().as_ref();
    let ndim = shape.len();
    if ndim == 0 {
        return vec![layout.offset()];
    }
    let last = last_axis_visited(shape[ndim - 1], max_print, min_print);
    if last.is_empty() {
        return vec![];
    }
    if ndim == 1 {
        return last.iter().map(|&i| (offset + i as isize * stride[0]) as usize).collect();
    }
    let prefix: Vec<Vec<usize>> =
        (0..ndim - 1).map(|ax| prefix_axis_visited(shape[ax], max_print, min_print)).collect();
    if prefix.iter().any(|v| v.is_empty()) {
        return vec![];
    }
    let mut out = vec![];
    let mut pos = vec![0usize; ndim - 1];
    loop {
        let mut base = offset;
        for (ax, &p) in pos.iter().enumerate() {
            base += prefix[ax][p] as isize * stride[ax];
        }
        for &i in &last {
            out.push((base + i as isize * stride[ndim - 1]) as usize);
        }
        // advance the prefix multi-index (row-major), or stop
        let mut ax = ndim - 2;
        loop {
            pos[ax] += 1;
            if pos[ax] < prefix[ax].len() {
                break;
            }
            pos[ax] = 0;
            if ax == 0 {
                return out;
            }
            ax -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touched_count_matches_walker_truncation() {
        // every axis exceeds MAX_PRINT (8) -> head 3 + tail 3 per axis
        let layout: Layout<IxD> = Layout::new(vec![100, 100, 100], vec![10000, 100, 1], 0).unwrap();
        let idxs = touched_element_indices(&layout, 8, 3);
        assert_eq!(idxs.len(), 6 * 6 * (3 + 3));
    }

    #[test]
    fn touched_covers_offset_and_strides() {
        // 2-D transposed view with an offset: indices must be absolute
        let layout: Layout<IxD> = Layout::new(vec![9, 12], vec![1, 9], 5).unwrap();
        let idxs = touched_element_indices(&layout, 8, 3);
        assert_eq!(idxs.len(), (3 + 3) * (3 + 3));
        assert!(idxs.iter().all(|&i| i >= 5));
    }

    #[test]
    fn touched_zero_size_is_empty() {
        let layout: Layout<IxD> = Layout::new(vec![4, 0, 9], vec![0, 0, 1], 0).unwrap();
        assert!(touched_element_indices(&layout, 8, 3).is_empty());
        let scalar: Layout<IxD> = Layout::new(vec![], vec![], 7).unwrap();
        assert_eq!(touched_element_indices(&scalar, 8, 3), vec![7]);
    }
}
