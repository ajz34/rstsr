pub mod rstsr_traits {
    pub use crate::traits_def::{
        BatchedMatmulAPI, BatchedMatmulStridedAPI, CholeskyAPI, DetAPI, EighAPI, EigvalshAPI, InvAPI, PinvAPI,
        SLogDetAPI, SVDvalsAPI, SolveGeneralAPI, SolveSymmetricAPI, SolveTriangularAPI, SVDAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{
        batched_matmul, batched_matmul_f, batched_matmul_strided, batched_matmul_strided_f, cholesky, cholesky_f,
        det, det_f, eigh, eigh_f, eigvalsh, eigvalsh_f, inv, inv_f, pinv, pinv_f, slogdet, slogdet_f,
        solve_general, solve_general_f, solve_symmetric, solve_symmetric_f, solve_triangular, solve_triangular_f,
        svd, svd_f, svdvals, svdvals_f,
    };
}

pub mod rstsr_structs {
    pub use crate::traits_def::{
        EighArgs, EighArgs_, EighArgs_Builder, EighResult, SLogDetResult, SVDArgs, SVDArgs_, SVDArgs_Builder, SVDResult,
    };
}
