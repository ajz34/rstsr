//! Doc-draft twins of the creation_from_tensor docstrings.

#[allow(unused_imports)]
use crate::test_utils::*;
use rstsr::prelude::*;

use super::CATEGORY;
use crate::TESTCFG;

#[cfg(test)]
mod doc_diag {
    use super::*;
    static FUNC: &str = "doc_diag";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `diag(tensor)` extracts the main diagonal from a 2-D input
        let a = rt::arange((9, &device)).into_shape([3, 3]);
        println!("{}", rt::diag(&a));
        // [ 0 4 8]
        assert_eq!(format!("{}", rt::diag(&a)), "[ 0 4 8]");

        // `diag((tensor, offset))`: 2-D input, offset = 1 (above the main diagonal)
        println!("{}", rt::diag((&a, 1)));
        // [ 1 5]
        assert_eq!(format!("{}", rt::diag((&a, 1))), "[ 1 5]");

        // ... or constructs a diagonal matrix from a 1-D input
        let v = rt::arange((3, &device));
        println!("{}", rt::diag(&v));
        // [[ 0 0 0]
        //  [ 0 1 0]
        //  [ 0 0 2]]
        assert_eq!(format!("{}", rt::diag(&v)), "[[ 0 0 0]\n [ 0 1 0]\n [ 0 0 2]]");

        // `diag((tensor, offset))`: 1-D input, offset = -1 (below the main diagonal)
        println!("{}", rt::diag((&v, -1)));
        // [[ 0 0 0 0]
        //  [ 0 0 0 0]
        //  [ 0 1 0 0]
        //  [ 0 0 2 0]]
        assert_eq!(format!("{}", rt::diag((&v, -1))), "[[ 0 0 0 0]\n [ 0 0 0 0]\n [ 0 1 0 0]\n [ 0 0 2 0]]");

        // column-major default order: same logical content
        let mut device_c = TESTCFG.device.clone();
        device_c.set_default_order(ColMajor);
        let v_c = rt::arange((3, &device_c));
        let m_c = rt::diag((&v_c, -1));
        assert_eq!(format!("{m_c}"), "[[ 0 0 0 0]\n [ 0 0 0 0]\n [ 0 1 0 0]\n [ 0 0 2 0]]");
        assert!(m_c.f_contig());
        let a_f = a.to_contig(ColMajor);
        assert_eq!(format!("{}", rt::diag(&a_f)), "[ 0 4 8]");
        assert_eq!(format!("{}", rt::diag((&a_f, 1))), "[ 1 5]");
    }
}

#[cfg(test)]
mod doc_meshgrid {
    use super::*;
    static FUNC: &str = "doc_meshgrid";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        let x = rt::arange((3, &device));
        let y = rt::arange((2, &device));

        // `meshgrid((tensors, indexing))`, reference inputs, "ij": default copy = true
        let grids = rt::meshgrid(([&x, &y], "ij"));
        println!("{}", grids[0]);
        // [[ 0 0]
        //  [ 1 1]
        //  [ 2 2]]
        println!("{}", grids[1]);
        // [[ 0 1]
        //  [ 0 1]
        //  [ 0 1]]
        assert_eq!(format!("{}", grids[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");
        assert_eq!(format!("{}", grids[1]), "[[ 0 1]\n [ 0 1]\n [ 0 1]]");
        assert!(grids.iter().all(|g| g.is_owned()));

        // `meshgrid(tensors)`, reference inputs: implicit "xy" and copy = true
        let grids_xy = rt::meshgrid([&x, &y]);
        println!("{}", grids_xy[0]);
        // [[ 0 1 2]
        //  [ 0 1 2]]
        println!("{}", grids_xy[1]);
        // [[ 0 0 0]
        //  [ 1 1 1]]
        assert_eq!(format!("{}", grids_xy[0]), "[[ 0 1 2]\n [ 0 1 2]]");
        assert_eq!(format!("{}", grids_xy[1]), "[[ 0 0 0]\n [ 1 1 1]]");

        // `meshgrid((tensors, copy))`, reference inputs, copy = false: views
        // (implicit "xy", so the output shape is [2, 3])
        let grids_v = rt::meshgrid(([&x, &y], false));
        assert!(grids_v.iter().all(|g| !g.is_owned()));
        println!("{:?}", grids_v[0].layout());
        // 2-Dim (dyn), contiguous: Custom
        // shape: [2, 3], stride: [0, 1], offset: 0
        assert_eq!(
            format!("{:?}", grids_v[0].layout()),
            "2-Dim (dyn), contiguous: Custom\nshape: [2, 3], stride: [0, 1], offset: 0"
        );

        // `meshgrid((tensors, indexing, copy))`, reference inputs, "ij", copy = false
        let grids_ij_v = rt::meshgrid(([&x, &y], "ij", false));
        assert!(grids_ij_v.iter().all(|g| !g.is_owned()));
        println!("{:?}", grids_ij_v[0].layout());
        // 2-Dim (dyn), contiguous: Custom
        // shape: [3, 2], stride: [1, 0], offset: 0
        assert_eq!(
            format!("{:?}", grids_ij_v[0].layout()),
            "2-Dim (dyn), contiguous: Custom\nshape: [3, 2], stride: [1, 0], offset: 0"
        );
        assert_eq!(format!("{}", grids_ij_v[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");

        // `meshgrid((tensors, indexing, copy))`, reference inputs, copy = true
        let grids_xy_c = rt::meshgrid(([&x, &y], "xy", true));
        assert_eq!(format!("{}", grids_xy_c[0]), "[[ 0 1 2]\n [ 0 1 2]]");

        // reference inputs as `Vec<&TensorAny>` and `&Vec<&TensorAny>` behave the same
        let grids_vec = rt::meshgrid(vec![&x, &y]);
        assert_eq!(format!("{}", grids_vec[0]), "[[ 0 1 2]\n [ 0 1 2]]");
        let grids_refvec = rt::meshgrid((&vec![&x, &y], "ij"));
        assert_eq!(format!("{}", grids_refvec[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");

        // `meshgrid(tensors)`, owned inputs: owned grids (Vec<Tensor>)
        let grids_own = rt::meshgrid(vec![rt::arange((3, &device)), rt::arange((2, &device))]);
        println!("{}", grids_own[0]);
        // [[ 0 1 2]
        //  [ 0 1 2]]
        assert_eq!(format!("{}", grids_own[0]), "[[ 0 1 2]\n [ 0 1 2]]");

        // `meshgrid((tensors, indexing))`, owned inputs
        let grids_own_ij = rt::meshgrid((vec![rt::arange((3, &device)), rt::arange((2, &device))], "ij"));
        assert_eq!(format!("{}", grids_own_ij[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");

        // owned `[Tensor; N]` array with the full three-tuple
        let grids_arr = rt::meshgrid(([rt::arange((3, &device)), rt::arange((2, &device))], "xy", true));
        assert_eq!(format!("{}", grids_arr[0]), "[[ 0 1 2]\n [ 0 1 2]]");

        // owned `Vec<Tensor>` with the full three-tuple, copy = false: the grids are
        // owned tensors with stride-0 layouts aliasing the inputs' own storages
        let grids_own_3 = rt::meshgrid((vec![rt::arange((3, &device)), rt::arange((2, &device))], "ij", false));
        println!("{}", grids_own_3[0]);
        // [[ 0 0]
        //  [ 1 1]
        //  [ 2 2]]
        println!("{:?}", grids_own_3[0].layout());
        // 2-Dim (dyn), contiguous: Custom
        // shape: [3, 2], stride: [1, 0], offset: 0
        assert_eq!(format!("{}", grids_own_3[0]), "[[ 0 0]\n [ 1 1]\n [ 2 2]]");
        assert_eq!(
            format!("{:?}", grids_own_3[0].layout()),
            "2-Dim (dyn), contiguous: Custom\nshape: [3, 2], stride: [1, 0], offset: 0"
        );
    }
}

#[cfg(test)]
mod doc_concat {
    use super::*;
    static FUNC: &str = "doc_concat";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `concat(tensors)`, implicit axis = 0
        let a = rt::arange((3, &device));
        let b = rt::arange((3, 6, &device));
        println!("{}", rt::concat([&a, &b]));
        // [ 0 1 2 3 4 5]
        assert_eq!(format!("{}", rt::concat([&a, &b])), "[ 0 1 2 3 4 5]");

        // `concat((tensors, axis))`, explicit axis = 1
        let m = rt::arange((6, &device)).into_shape([2, 3]);
        let n = rt::full(([2, 2], 9, &device));
        println!("{}", rt::concat(([&m, &n], 1)));
        // [[ 0 1 2 9 9]
        //  [ 3 4 5 9 9]]
        assert_eq!(format!("{}", rt::concat(([&m, &n], 1))), "[[ 0 1 2 9 9]\n [ 3 4 5 9 9]]");

        // negative axis counts from the back; owned Vec and usize / i32 axis forms
        let m2 = rt::arange((6, &device)).into_shape([2, 3]);
        let n2 = rt::full(([2, 2], 9, &device));
        assert_eq!(format!("{}", rt::concat(([&m2, &n2], -1))), "[[ 0 1 2 9 9]\n [ 3 4 5 9 9]]");
        let m3 = rt::arange((6, &device)).into_shape([2, 3]);
        let n3 = rt::full(([2, 2], 9, &device));
        assert_eq!(format!("{}", rt::concat((vec![m3, n3], 1usize))), "[[ 0 1 2 9 9]\n [ 3 4 5 9 9]]");
        let m4 = rt::arange((6, &device)).into_shape([2, 3]);
        let n4 = rt::full(([2, 2], 9, &device));
        assert_eq!(format!("{}", rt::concat(([m4, n4], 1i32))), "[[ 0 1 2 9 9]\n [ 3 4 5 9 9]]");
    }
}

#[cfg(test)]
mod doc_hstack {
    use super::*;
    static FUNC: &str = "doc_hstack";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // `hstack(tensors)`: 1-D inputs concatenate along axis 0
        let a = rt::arange((3, &device));
        let b = rt::arange((3, 6, &device));
        println!("{}", rt::hstack([&a, &b]));
        // [ 0 1 2 3 4 5]
        assert_eq!(format!("{}", rt::hstack([&a, &b])), "[ 0 1 2 3 4 5]");

        // 2-D inputs concatenate along axis 1; owned Vec form behaves the same
        let a = rt::arange((4, &device)).into_shape([2, 2]);
        let b = rt::full(([2, 1], 9, &device));
        println!("{}", rt::hstack([&a, &b]));
        // [[ 0 1 9]
        //  [ 2 3 9]]
        assert_eq!(format!("{}", rt::hstack([&a, &b])), "[[ 0 1 9]\n [ 2 3 9]]");
        let a2 = rt::arange((4, &device)).into_shape([2, 2]);
        let b2 = rt::full(([2, 1], 9, &device));
        assert_eq!(format!("{}", rt::hstack(vec![a2, b2])), "[[ 0 1 9]\n [ 2 3 9]]");
    }
}

#[cfg(test)]
mod doc_vstack {
    use super::*;
    static FUNC: &str = "doc_vstack";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // `vstack(tensors)`: 1-D inputs are promoted to rows
        let a = rt::arange((3, &device));
        let b = rt::arange((3, 6, &device));
        println!("{}", rt::vstack([&a, &b]));
        // [[ 0 1 2]
        //  [ 3 4 5]]
        assert_eq!(format!("{}", rt::vstack([&a, &b])), "[[ 0 1 2]\n [ 3 4 5]]");

        // 2-D inputs stack along axis 0; owned Vec form behaves the same
        let a = rt::arange((4, &device)).into_shape([2, 2]);
        let b = rt::full(([2, 2], 9, &device));
        println!("{}", rt::vstack([&a, &b]));
        // [[ 0 1]
        //  [ 2 3]
        //  [ 9 9]
        //  [ 9 9]]
        assert_eq!(format!("{}", rt::vstack([&a, &b])), "[[ 0 1]\n [ 2 3]\n [ 9 9]\n [ 9 9]]");
        let a2 = rt::arange((4, &device)).into_shape([2, 2]);
        let b2 = rt::full(([2, 2], 9, &device));
        assert_eq!(format!("{}", rt::vstack(vec![a2, b2])), "[[ 0 1]\n [ 2 3]\n [ 9 9]\n [ 9 9]]");
    }
}

#[cfg(test)]
mod doc_stack {
    use super::*;
    static FUNC: &str = "doc_stack";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `stack(tensors)`, implicit axis = 0 (a new leading axis)
        let a = rt::arange((4, &device));
        let b = rt::full(([4], 9, &device));
        println!("{}", rt::stack([&a, &b]));
        // [[ 0 1 2 3]
        //  [ 9 9 9 9]]
        assert_eq!(format!("{}", rt::stack([&a, &b])), "[[ 0 1 2 3]\n [ 9 9 9 9]]");

        // `stack((tensors, axis))`, explicit axis = 1: owned Vec form
        let a2 = rt::arange((4, &device));
        let b2 = rt::full(([4], 9, &device));
        println!("{}", rt::stack((vec![a2, b2], 1)));
        // [[ 0 9]
        //  [ 1 9]
        //  [ 2 9]
        //  [ 3 9]]
        let a3 = rt::arange((4, &device));
        let b3 = rt::full(([4], 9, &device));
        assert_eq!(format!("{}", rt::stack((vec![a3, b3], 1))), "[[ 0 9]\n [ 1 9]\n [ 2 9]\n [ 3 9]]");

        // 0-D inputs stack into a 1-D tensor
        let s0 = rt::full(([], 3, &device));
        let t0 = rt::full(([], 7, &device));
        println!("{}", rt::stack([&s0, &t0]));
        // [ 3 7]
        assert_eq!(format!("{}", rt::stack([&s0, &t0])), "[ 3 7]");
    }
}

#[cfg(test)]
mod doc_unstack {
    use super::*;
    static FUNC: &str = "doc_unstack";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `unstack(tensor)`, implicit axis = 0
        let a = rt::arange((6, &device)).into_shape([2, 3]);
        let v = rt::unstack(&a);
        println!("{}", v.len());
        // 2
        println!("{}", v[0]);
        // [ 0 1 2]
        println!("{}", v[1]);
        // [ 3 4 5]
        assert_eq!(v.len(), 2);
        assert_eq!(format!("{}", v[0]), "[ 0 1 2]");
        assert_eq!(format!("{}", v[1]), "[ 3 4 5]");

        // `unstack((tensor, axis))`, explicit axis = 1 (negative axis works too)
        let v1 = rt::unstack((&a, 1));
        println!("{}", v1.len());
        // 3
        println!("{}", v1[0]);
        // [ 0 3]
        assert_eq!(v1.len(), 3);
        assert_eq!(format!("{}", v1[0]), "[ 0 3]");
        let v_neg = rt::unstack((&a, -1));
        assert_eq!(v_neg.len(), 3);

        // `unstack(tensor)` with a view input
        let vv = rt::unstack(a.view());
        assert_eq!(vv.len(), 2);
        assert_eq!(format!("{}", vv[0]), "[ 0 1 2]");
    }
}

#[cfg(test)]
mod doc_atleast_1d {
    use super::*;
    static FUNC: &str = "doc_atleast_1d";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `atleast_1d(tensor)` promotes a 0-D input to (1,)
        let s = rt::full(([], 5, &device));
        println!("{}", rt::atleast_1d(&s));
        // [ 5]
        assert_eq!(format!("{}", rt::atleast_1d(&s)), "[ 5]");

        // ndim >= 1 inputs are returned unchanged (a view)
        let a = rt::arange((3, &device));
        println!("{}", rt::atleast_1d(&a));
        // [ 0 1 2]
        assert_eq!(format!("{}", rt::atleast_1d(&a)), "[ 0 1 2]");
    }
}

#[cfg(test)]
mod doc_atleast_2d {
    use super::*;
    static FUNC: &str = "doc_atleast_2d";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `atleast_2d(tensor)` promotes 0-D to (1, 1) ...
        let s = rt::full(([], 5, &device));
        println!("{}", rt::atleast_2d(&s));
        // [[ 5]]
        assert_eq!(format!("{}", rt::atleast_2d(&s)), "[[ 5]]");

        // ... and 1-D (N,) to (1, N)
        let a = rt::arange((3, &device));
        println!("{}", rt::atleast_2d(&a));
        // [[ 0 1 2]]
        assert_eq!(format!("{}", rt::atleast_2d(&a)), "[[ 0 1 2]]");

        // ndim >= 2 inputs are returned unchanged
        let m = rt::arange((4, &device)).into_shape([2, 2]);
        assert_eq!(format!("{}", rt::atleast_2d(&m)), "[[ 0 1]\n [ 2 3]]");
    }
}

#[cfg(test)]
mod doc_atleast_3d {
    use super::*;
    static FUNC: &str = "doc_atleast_3d";

    #[test]
    fn test_doc() {
        crate::specify_test!("test_doc");
        let mut device = TESTCFG.device.clone();
        device.set_default_order(RowMajor);

        // # Overload List: `atleast_3d(tensor)` promotes 0-D to (1, 1, 1) ...
        let s = rt::full(([], 5, &device));
        println!("{}", rt::atleast_3d(&s));
        // [[[ 5]]]
        assert_eq!(format!("{}", rt::atleast_3d(&s)), "[[[ 5]]]");

        // ... 1-D (N,) to (1, N, 1) ...
        let a = rt::arange((3, &device));
        let w = rt::atleast_3d(&a);
        println!("{w}");
        // [[[ 0]
        //   [ 1]
        //   [ 2]]]
        assert_eq!(format!("{w}"), "[[[ 0]\n  [ 1]\n  [ 2]]]");
        assert_eq!(w.shape(), &[1, 3, 1]);

        // ... and 2-D (M, N) to (M, N, 1)
        let m = rt::arange((4, &device)).into_shape([2, 2]);
        let m3 = rt::atleast_3d(&m);
        assert_eq!(m3.shape(), &[2, 2, 1]);
        println!("{m3}");
        // [[[ 0]
        //   [ 1]]
        //
        //  [[ 2]
        //   [ 3]]]
        assert_eq!(format!("{m3}"), "[[[ 0]\n  [ 1]]\n\n [[ 2]\n  [ 3]]]");
    }
}
