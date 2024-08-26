//! Layout broadcasting.
//!
//! We refer to documentation of Python array API: [broadcasting](https://data-apis.org/array-api/2023.12/API_specification/broadcasting.html).

// use super::DimMaxAPI;
use crate::prelude_dev::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastType {
    Upcast,
    Expand,
    Preserve,
    Undefined,
}

/// Shape broadcasting.
///
/// # See also
///
/// [broadcasting](https://data-apis.org/array-api/2023.12/API_specification/broadcasting.html)
#[allow(clippy::type_complexity)]
pub fn broadcast_shape<D1, D2>(
    shape1: &D1,
    shape2: &D2,
) -> Result<(<D1 as DimMaxAPI<D2>>::Max, Vec<BroadcastType>, Vec<BroadcastType>)>
where
    D1: DimDevAPI + DimMaxAPI<D2>,
    D2: DimDevAPI,
    <D1 as DimMaxAPI<D2>>::Max: TryFrom<IxD>,
{
    // step 1-6: determine maximum shape
    let (n1, n2) = (shape1.ndim(), shape2.ndim());
    let n = usize::max(n1, n2);
    // step 7: declare result shape and corresponding broadcast type
    let mut shape = vec![0; n];
    let mut tp1 = vec![BroadcastType::Undefined; n];
    let mut tp2 = vec![BroadcastType::Undefined; n];
    // step 8-10: iterate over the shape
    for i in (0..n).rev() {
        let in1 = (n1 + i) as isize - n as isize;
        let in2 = (n2 + i) as isize - n as isize;

        let d1 = if in1 >= 0 { shape1[in1 as usize] } else { 1 };
        let d2 = if in2 >= 0 { shape2[in2 as usize] } else { 1 };

        match (d1 == 1, d2 == 1) {
            (true, true) => {
                tp1[i] = BroadcastType::Preserve;
                tp2[i] = BroadcastType::Preserve;
                shape[i] = 1;
            },
            (false, true) => {
                tp1[i] = BroadcastType::Preserve;
                tp2[i] = BroadcastType::Upcast;
                shape[i] = d1;
            },
            (true, false) => {
                tp1[i] = BroadcastType::Upcast;
                tp2[i] = BroadcastType::Preserve;
                shape[i] = d2;
            },
            (false, false) => {
                rstsr_assert_eq!(d1, d2, InvalidLayout, "Broadcasting failed.")?;
                tp1[i] = BroadcastType::Preserve;
                tp2[i] = BroadcastType::Preserve;
                shape[i] = d1;
            },
        }

        if in1 < 0 {
            tp1[i] = BroadcastType::Expand;
        }
        if in2 < 0 {
            tp2[i] = BroadcastType::Expand;
        }
    }

    let shape = TryInto::<<D1 as DimMaxAPI<D2>>::Max>::try_into(shape);
    let shape = shape.map_err(|_| Error::InvalidLayout("Type cast error.".to_string()))?;

    return Ok((shape, tp1, tp2));
}

/// Layout broadcasting.
///
/// Dimensions that to be upcasted or expanded will have stride length of zero.
///
/// Note that zero stride length is generally not accepted, since different
/// indices will point to the same memory, which is not expected in most cases
/// for this library. But this will be convenient when we need to broadcast.
///
/// # See also
///
/// [broadcasting](https://data-apis.org/array-api/2023.12/API_specification/broadcasting.html)
#[allow(clippy::type_complexity)]
pub fn broadcast_layout<D1, D2>(
    layout1: &Layout<D1>,
    layout2: &Layout<D2>,
) -> Result<(Layout<<D1 as DimMaxAPI<D2>>::Max>, Layout<<D1 as DimMaxAPI<D2>>::Max>)>
where
    D1: DimDevAPI + DimMaxAPI<D2>,
    D2: DimDevAPI,
    <D1 as DimMaxAPI<D2>>::Max: DimDevAPI + TryFrom<IxD>,
    <<D1 as DimMaxAPI<D2>>::Max as DimBaseAPI>::Stride: TryFrom<Vec<isize>>,
{
    let shape1 = layout1.shape();
    let shape2 = layout2.shape();
    let (shape, tp1, tp2) = broadcast_shape(shape1, shape2)?;
    let layout1 = update_layout_by_shape(layout1, &shape, &tp1)?;
    let layout2 = update_layout_by_shape(layout2, &shape, &tp2)?;
    return Ok((layout1, layout2));
}

fn update_layout_by_shape<D, DMax>(
    layout: &Layout<D>,
    shape: &DMax,
    broadcast_type: &[BroadcastType],
) -> Result<Layout<DMax>>
where
    D: DimDevAPI,
    DMax: DimDevAPI,
    DMax::Stride: TryFrom<Vec<isize>>,
{
    let n_old = layout.ndim();
    let stride_old = layout.stride();
    let n = shape.ndim();
    let mut stride = vec![0; n];
    stride[n - n_old..n].copy_from_slice(stride_old.as_ref());
    for i in 0..n {
        match broadcast_type[i] {
            BroadcastType::Expand | BroadcastType::Upcast => {
                stride[i] = 0;
            },
            _ => {},
        }
    }
    let stride = stride.try_into();
    let stride = stride.map_err(|_| Error::InvalidLayout("Type cast error.".to_string()))?;
    unsafe { Ok(Layout::new_unchecked(shape.clone(), stride, layout.offset())) }
}

#[cfg(test)]
mod test {
    use super::*;
    use BroadcastType::*;

    #[test]
    fn test_broadcast_shape() {
        // A      (4d array):  8 x 1 x 6 x 1
        // B      (3d array):      7 x 1 x 5
        // ---------------------------------
        // Result (4d array):  8 x 7 x 6 x 5
        let shape1 = [8, 1, 6, 1];
        let shape2 = [7, 1, 5];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [8, 7, 6, 5]);
        assert_eq!(broadcast.1, [Preserve, Upcast, Preserve, Upcast]);
        assert_eq!(broadcast.2, [Expand, Preserve, Upcast, Preserve]);
        // A      (2d array):  5 x 4
        // B      (1d array):      1
        // -------------------------
        // Result (2d array):  5 x 4
        let shape1 = [5, 4];
        let shape2 = [1];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [5, 4]);
        assert_eq!(broadcast.1, [Preserve, Preserve]);
        assert_eq!(broadcast.2, [Expand, Upcast]);
        // A      (2d array):  5 x 4
        // B      (1d array):      4
        // -------------------------
        // Result (2d array):  5 x 4
        let shape1 = [5, 4];
        let shape2 = [4];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [5, 4]);
        assert_eq!(broadcast.1, [Preserve, Preserve]);
        assert_eq!(broadcast.2, [Expand, Preserve]);
        // A      (3d array):  15 x 3 x 5
        // B      (3d array):  15 x 1 x 5
        // ------------------------------
        // Result (3d array):  15 x 3 x 5
        let shape1 = [15, 3, 5];
        let shape2 = [15, 1, 5];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [15, 3, 5]);
        assert_eq!(broadcast.1, [Preserve, Preserve, Preserve]);
        assert_eq!(broadcast.2, [Preserve, Upcast, Preserve]);
        // A      (3d array):  15 x 3 x 5
        // B      (2d array):       3 x 5
        // ------------------------------
        // Result (3d array):  15 x 3 x 5
        let shape1 = [15, 3, 5];
        let shape2 = [3, 5];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [15, 3, 5]);
        assert_eq!(broadcast.1, [Preserve, Preserve, Preserve]);
        assert_eq!(broadcast.2, [Expand, Preserve, Preserve]);
        // A      (3d array):  15 x 3 x 5
        // B      (2d array):       3 x 1
        // ------------------------------
        // Result (3d array):  15 x 3 x 5
        let shape1 = [15, 3, 5];
        let shape2 = [3, 1];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [15, 3, 5]);
        assert_eq!(broadcast.1, [Preserve, Preserve, Preserve]);
        assert_eq!(broadcast.2, [Expand, Preserve, Upcast]);

        // other test cases
        let shape1 = [1, 1, 2];
        let shape2 = [1, 2];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [1, 1, 2]);
        assert_eq!(broadcast.1, [Preserve, Preserve, Preserve]);
        assert_eq!(broadcast.2, [Expand, Preserve, Preserve]);

        // other test cases
        let shape1 = [1, 2];
        let shape2 = [1, 1, 2];
        let broadcast = broadcast_shape(&shape1, &shape2).unwrap();
        assert_eq!(broadcast.0, [1, 1, 2]);
        assert_eq!(broadcast.1, [Expand, Preserve, Preserve]);
        assert_eq!(broadcast.2, [Preserve, Preserve, Preserve]);
    }

    #[test]
    fn test_broadcast_shape_fail() {
        // A      (1d array):  3
        // B      (1d array):  4           # dimension does not match
        let shape1 = [3];
        let shape2 = [4];
        let broadcast = broadcast_shape(&shape1, &shape2);
        assert!(broadcast.is_err());
        // A      (2d array):      2 x 1
        // B      (3d array):  8 x 4 x 3   # second dimension does not match
        let shape1 = [2, 1];
        let shape2 = [8, 4, 3];
        let broadcast = broadcast_shape(&shape1, &shape2);
        assert!(broadcast.is_err());
        // A      (3d array):  15 x 3 x 5
        // B      (2d array):  15 x 3
        // # singleton dimensions can only be prepended, not appended
        let shape1 = [15, 3, 5];
        let shape2 = [15, 3];
        let broadcast = broadcast_shape(&shape1, &shape2);
        assert!(broadcast.is_err());
    }

    #[test]
    fn test_broadcast_layout() {
        // A      (4d array):  8 x 1 x 6 x 3 x 1
        // B      (3d array):      7 x 1 x 3 x 5
        // -------------------------------------
        // Result (4d array):  8 x 7 x 6 x 3 x 5
        let shape1 = [8, 1, 6, 3, 1];
        let shape2 = [7, 1, 3, 5];
        let layout1 = shape1.c();
        let layout2 = shape2.f();
        let (layout1, layout2) = broadcast_layout(&layout1, &layout2).unwrap();
        assert_eq!(layout1.shape(), &[8, 7, 6, 3, 5]);
        assert_eq!(layout2.shape(), &[8, 7, 6, 3, 5]);
        assert_eq!(layout1.stride(), &[18, 0, 3, 1, 0]);
        assert_eq!(layout2.stride(), &[0, 1, 0, 7, 21]);
    }
}
