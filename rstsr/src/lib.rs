#![doc = include_str!("../notes_to_api_doc.md")]
#![doc = include_str!("../readme.md")]
// the crate-root table of contents links the `cheatsheet::linalg` subpage,
// which does not exist without the `linalg` feature
#![cfg_attr(not(feature = "linalg"), allow(rustdoc::broken_intra_doc_links))]
#![cfg_attr(not(test), no_std)]
pub mod prelude;

// re-exported so the crate-level doc pages can link `crate::order_semantics`
pub use rstsr_core::order_semantics;

#[allow(unused_imports)]
use crate as rstsr;

pub mod cheatsheet {
    #![doc = include_str!("docs/cheatsheet.md")]
    #![allow(unused_imports)]
    // the core page links to the linalg subpage, which is absent without `linalg`
    #![cfg_attr(not(feature = "linalg"), allow(rustdoc::broken_intra_doc_links))]

    use crate::prelude::*;
    use rt::*;

    #[cfg(feature = "linalg")]
    pub mod linalg {
        #![doc = include_str!("docs/cheatsheet_linalg.md")]
        #![allow(unused_imports)]

        use crate::prelude::*;
        use rt::linalg::*;
        use rt::*;
    }
}

pub mod api_specification {
    #![doc = include_str!("docs/api_specification.md")]
    #![allow(unused_imports)]

    use crate::prelude::*;
    use rstsr_dtype_traits::*;
    use rt::*;
    extern crate alloc;
    use rstsr_dtype_traits::*;
}

// re-export the order-semantics page so the shared spec pages' `crate::order_semantics`
// links resolve in this crate too (they are symlinked from rstsr-core)
pub use rstsr_core::order_semantics;

pub mod array_api_standard {
    #![doc = include_str!("docs/array_api_standard.md")]
    #![doc = include_str!("docs/array_api_standard_extension.md")]
    #![allow(unused_imports)]

    use crate::prelude::*;
    use core::ops::*;
    use num::complex::ComplexFloat;
    use num::{pow::Pow, Complex, Float, Integer, Num, Signed};
    use rstsr_core::prelude_dev::Indexer;
    use rstsr_dtype_traits::*;
    use rt::*;
}

#[test]
fn test() {
    use crate::prelude::*;

    // 3x2 matrix with c-contiguous memory layout
    let a = rt::asarray((vec![6., 2., 7., 4., 8., 5.], [3, 2].c()));

    // 2x4x3 matrix by arange and reshaping
    let b = rt::arange(24.);
    let b = b.reshape((-1, 4, 3));
    // in one line, you can also
    // let b = rt::arange(24.).into_shape((-1, 4, 3));

    // broadcasted matrix multiplication
    let c = b % a;

    // print layout of the result
    println!("{:?}", c.layout());
    // output:
    // 3-Dim (dyn), contiguous: Cc
    // shape: [2, 4, 2], stride: [8, 2, 1], offset: 0

    // print the result
    println!("{c:6.1}");
    // output:
    // [[[   23.0   14.0]
    //   [   86.0   47.0]
    //   [  149.0   80.0]
    //   [  212.0  113.0]]
    //
    //  [[  275.0  146.0]
    //   [  338.0  179.0]
    //   [  401.0  212.0]
    //   [  464.0  245.0]]]
}

#[cfg(feature = "dlpack")]
#[test]
fn test_dlpack_prelude() {
    use crate::prelude::*;

    // the bridge crate is surfaced through the prelude namespace `rt::dlpack`
    let device = DeviceCpuSerial::default();
    let tensor: Tensor<f64, DeviceCpuSerial, IxD> = rt::arange_f((0.0, 5.0, 1.0, &device)).unwrap();
    let shared = rt::dlpack::into_shared_dlpack_f(tensor).unwrap();
    let export: dlpack::DlpackExport = rt::dlpack::to_dlpack_shared(&shared);
    assert_eq!(export.flags(), rstsr_cpu_dlpack::dlpark::DlpackFlags::READ_ONLY.bits());

    // a second function of the namespace, pinned by signature
    let _: fn(Tensor<f64, DeviceCpuSerial, IxD>) -> rt::dlpack::DlpackExport = rt::dlpack::into_dlpack;
}
